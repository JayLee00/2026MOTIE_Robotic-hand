#!/usr/bin/env python3
"""SAM3 후보 → Qwen2-VL 선택 → 파지.

run_sam3_gui_select.py 의 형제. 마우스 클릭 대신 **Qwen2-VL 이 자연어 지시로 자동 선택**한다.
  ① SAM3 로 query 물체 전부 검출 (정밀 마스크)
  ② 번호 박스 overlay + 자연어 지시를 Qwen2-VL 에 전달 → 잡을 index
  ③ 그 index 의 SAM3 마스크 → run_topdown_grasp(PCA) → (옵션) send_to_robot_scenario1

'SAM3=정밀 후보 / Qwen2-VL=판단' 구조. 이미지+텍스트 둘 다 근거 → 외형 판단 가능.
기존 grasp/robot 경로는 그대로 재사용 (원본 파일 안 건드림).

용법 (grasp_fruit conda python, ROS 오염 제거 필수):
  env -u PYTHONPATH -u AMENT_PREFIX_PATH -u LD_LIBRARY_PATH \
    ~/miniconda3/envs/grasp_fruit/bin/python scripts/run_sam3_qwen_select.py \
      --input data/raw/kiwi_probe_000.npz \
      --query kiwi \
      --instruction "왼쪽에 있는 잘 익은 것" \
      --calibration configs/calibration/extrinsic_20260612_170053.json
  # --execute_robot 붙이면 pose 확인(Enter) 후 실제 로봇 파지 (호스트 py3.10 직접 실행)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cv2

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

# run_sam3_gui_select 의 SAM3 검출/결과창 로직 재사용 (import 만으로 창은 안 뜸)
from run_sam3_gui_select import detect_all, report_pose, BOX_COLORS, SEL_COLOR
from utils.qwen_select import QwenSelector
from utils.select_router import hybrid_select


def build_overlay(base_bgr, dets):
    """번호 박스를 그린 overlay(BGR) + Qwen 용 detections 메타 반환."""
    canvas = base_bgr.copy()
    meta = []
    for i, d in enumerate(dets):
        x0, y0, x1, y1 = d["box"]
        color = BOX_COLORS[i % len(BOX_COLORS)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, f"#{i} {d['score']:.2f}", (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        meta.append({"index": i, "score": d["score"], "box": d["box"]})
    return canvas, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="npz (keys: rgb[BGR], depth, K)")
    ap.add_argument("--query", required=True, help="SAM3 검출용 물체명 (예: kiwi)")
    ap.add_argument("--instruction", required=True,
                    help="선택 지시 (예: '왼쪽에 있는 잘 익은 것')")
    ap.add_argument("--calibration", default=None, help="extrinsic json — 파지 단계로 전달")
    ap.add_argument("--sam3_model_id", default="facebook/sam3")
    ap.add_argument("--sam3_threshold", type=float, default=0.5)
    ap.add_argument("--sam3_mask_threshold", type=float, default=0.5)
    ap.add_argument("--qwen_model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--output_dir", default=str(ROOT / "data" / "interim"))
    ap.add_argument("--no_grasp", action="store_true",
                    help="mask 만 저장하고 파지 pose 계산 생략")
    # ── 로봇 실행 (시나리오1: 물체 쥔 채 유지) ──
    ap.add_argument("--execute_robot", action="store_true",
                    help="파지 pose 후 send_to_robot_scenario1.py 로 실제 로봇 실행. "
                         "없으면 pose 계산까지만 (안전).")
    ap.add_argument("--execute_mode", default="direct_franka_topic",
                    choices=["trajectory_forwarder", "direct_franka_topic"])
    ap.add_argument("--speed_factor", type=float, default=0.1)
    ap.add_argument("--approach_offset", type=float, default=0.10)
    ap.add_argument("--disable_collision", action="store_true")
    ap.add_argument("--no_record", action="store_true")
    ap.add_argument("--show_sec", type=float, default=0.0,
                    help="선택 결과 창 자동 닫힘 시간(초). 0=Enter 까지 대기 (기본)")
    ap.add_argument("--no_show_result", action="store_true",
                    help="파지 결과 overlay 창을 띄우지 않음")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    npz = np.load(str(input_path))
    rgb = npz["rgb"]
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    base_bgr = rgb.copy()                              # npz 는 BGR 저장
    image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)   # SAM3/Qwen 입력용 RGB

    # ── ① SAM3: 모든 후보 검출 ──
    dets = detect_all(image_rgb, args.query, args.sam3_model_id,
                      args.sam3_threshold, args.sam3_mask_threshold)
    if not dets:
        print(f"[QWEN-SEL] {args.query!r} 검출 없음 — 종료.")
        sys.exit(1)
    print(f"[QWEN-SEL] {len(dets)}개 검출. Qwen2-VL 로 선택 중...")

    overlay_bgr, meta = build_overlay(base_bgr, dets)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # ── ② 하이브리드: VLM(외형 필터) → 규칙(위치 확정) ──
    selector = QwenSelector(model_id=args.qwen_model)
    result = hybrid_select(args.instruction, meta, overlay_rgb, selector,
                           base_rgb=image_rgb, query=args.query)
    selector.close()   # VRAM 반환 (SAM3 와 16GB 공유)
    sel = result["index"]
    print(f"\n[QWEN-SEL] 선택 #{sel}  (경로={result['source']})")
    print(f"           {result['reason']}")

    # 선택 시각화 저장 (사람 확인용)
    sel_vis = overlay_bgr.copy()
    ov = sel_vis.copy()
    ov[dets[sel]["mask"]] = SEL_COLOR
    sel_vis = cv2.addWeighted(ov, 0.45, sel_vis, 0.55, 0.0)
    x0, y0, x1, y1 = dets[sel]["box"]
    cv2.rectangle(sel_vis, (x0, y0), (x1, y1), SEL_COLOR, 3)
    cv2.putText(sel_vis, f"SELECTED #{sel}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, SEL_COLOR, 2)
    sel_path = out_dir / f"{stem}_qwen_select.png"
    cv2.imwrite(str(sel_path), sel_vis)
    print(f"[QWEN-SEL] 선택 시각화: {sel_path}")

    # 무엇을 골랐는지 창으로 표시. --show_sec N 이면 N초 후 자동 진행, 0이면 Enter 대기.
    # --no_show_result 로 창 자체를 끔.
    if not args.no_show_result:
        win = "qwen_select"
        if args.show_sec > 0:
            print(f"[QWEN-SEL] 선택 결과 창 — {args.show_sec:.0f}초 후 자동 진행 (Enter 로 즉시).")
        else:
            print("[QWEN-SEL] 선택 결과 창 — Enter 로 닫고 계속 (q/ESC 도 가능).")
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win, sel_vis)
        cv2.waitKey(1)
        t0 = time.time()
        while True:
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 10, ord('q'), 27):
                break
            if args.show_sec > 0 and time.time() - t0 >= args.show_sec:
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # ── 선택 마스크 저장 (기존 파이프라인과 동일 이름/포맷) ──
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), (dets[sel]["mask"].astype(np.uint8) * 255))
    print(f"[QWEN-SEL] mask 저장: {mask_path}")

    if args.no_grasp:
        print("[QWEN-SEL] --no_grasp: 파지 pose 계산 생략.")
        return

    # ── ③ 파지 pose 계산 (run_topdown_grasp) ──
    grasp_cmd = [sys.executable, str(SCRIPTS / "run_topdown_grasp.py"),
                 "--input", str(input_path),
                 "--mask", str(mask_path),
                 "--query", args.query]
    if args.calibration:
        grasp_cmd += ["--calibration", args.calibration]
    print("[QWEN-SEL] 파지 pose 계산:\n  " + " ".join(grasp_cmd))
    rc = subprocess.run(grasp_cmd).returncode
    if rc != 0:
        print(f"[QWEN-SEL] 파지 pose 실패 (rc={rc}) — 로봇 실행 안 함.")
        sys.exit(rc)

    # pose 로그 + 결과 overlay 창 (execute 모드: Enter=진행, q/ESC=취소)
    summary_path = ROOT / "data" / "outputs" / f"{stem}_topdown_summary.json"
    overlay_path = ROOT / "data" / "outputs" / f"{stem}_topdown_overlay.png"
    proceed = report_pose(summary_path, overlay_path,
                          show_result=not args.no_show_result,
                          execute=args.execute_robot)

    if not args.execute_robot:
        print("[QWEN-SEL] --execute_robot 없음: pose 계산까지만 (로봇 실행 안 함).")
        return
    if not proceed:
        print("[QWEN-SEL] 사용자 취소 (q/ESC) — 로봇 실행 안 함.")
        return

    # ── ④ 실제 로봇 실행 (시나리오1, 호스트 py3.10 — run_scenario1_host 와 동일 경로) ──
    if not summary_path.exists():
        print(f"[QWEN-SEL] summary 없음 → 로봇 실행 불가: {summary_path}")
        sys.exit(1)
    executor = str(SCRIPTS / "robot_executor_scenario1.py")
    extra = (f"--mode grasp --execute_mode {args.execute_mode} "
             f"--speed_factor {args.speed_factor} --approach_offset {args.approach_offset}")
    if args.disable_collision:
        extra += " --disable_collision"
    bash = (
        "unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER && "
        "source /opt/ros/humble/setup.bash && "
        "source /home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/fr_ws/install/setup.bash && "
        "source /home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/dex_ros/isaac-ros/kistar_ws/install/setup.bash && "
        "export ROS_DOMAIN_ID=9 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && "
        "export ROS_LOCALHOST_ONLY=0 && "
        f"exec /usr/bin/python3 {executor} --summary_json {summary_path} {extra}"
    )
    print("[QWEN-SEL] 로봇 실행 (시나리오1, 물체 쥔 채 유지 — 호스트 직접 실행)")
    rc = subprocess.run(["bash", "-c", bash]).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
