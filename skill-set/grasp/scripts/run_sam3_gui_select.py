#!/usr/bin/env python3
"""SAM3 GUI 선택 → 파지.

검출된 **모든** 인스턴스를 OpenCV 창에 번호+score 로 표시하고,
사용자가 **마우스로 박스를 클릭**해 하나를 고르면 그 마스크를
{output_dir}/{stem}_mask.png (기존 파이프라인과 동일 이름/포맷, 255=object)로 저장한 뒤
**run_topdown_grasp.py 를 자동 실행**해 파지 pose 까지 낸다.

기존 run_sam3_only_stage.py 는 argmax(score) 1개만 골랐지만, 이 스크립트는
사람이 직접 고른다 (점수가 비등해 자동선택이 통제 안 될 때 유용).

용법 (grasp_fruit conda python):
  ~/miniconda3/envs/grasp_fruit/bin/python scripts/run_sam3_gui_select.py \
      --input data/kiwi_test/kiwi_probe_003.npz \
      --query kiwi \
      --calibration configs/calibration/extrinsic_20260612_170053.json

조작:
  - 박스 클릭      : 그 인스턴스 선택(초록 강조). 여러 박스 겹치면 가장 작은 것.
  - Enter/Space   : 선택 확정 → mask 저장 → 파지 실행
  - r             : 선택 취소(초기화)
  - q / ESC       : 전체 취소(파지 안 함)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from transformers import Sam3Processor, Sam3Model

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

# 인스턴스 박스 색 (BGR) — 선택되지 않은 것들
BOX_COLORS = [(0, 180, 255), (255, 180, 0), (200, 0, 200), (0, 200, 220),
              (120, 120, 255), (255, 120, 0), (0, 160, 120), (160, 0, 255)]
SEL_COLOR = (0, 255, 0)   # 선택된 것


def detect_all(image_rgb, query, model_id, threshold, mask_threshold):
    """SAM3 실행 → 검출된 모든 인스턴스 리스트 반환 (argmax 안 함)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SAM3] 로딩: {model_id}  ({device})")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    img_in = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        vis = model.get_vision_features(pixel_values=img_in.pixel_values)
    txt_in = processor(text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(vision_embeds=vis, **txt_in)

    h, w = image_rgb.shape[:2]
    res = processor.post_process_instance_segmentation(
        out, threshold=threshold, mask_threshold=mask_threshold,
        target_sizes=[(h, w)])[0]

    masks = res.get("masks", [])
    scores = res.get("scores", [])
    boxes = res.get("boxes", [])

    dets = []
    for i in range(len(masks)):
        m = np.array(masks[i].cpu()).astype(bool)
        b = [int(v) for v in boxes[i].cpu().tolist()]
        dets.append({"mask": m, "score": float(scores[i]), "box": b})

    del model
    torch.cuda.empty_cache()
    # 점수 내림차순 정렬 → 번호가 신뢰도 순
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def report_pose(summary_path, overlay_path, show_result, execute=False):
    """파지 pose 를 로그로 출력하고 결과 overlay 를 창으로 띄운다.
    반환 proceed(bool): execute 모드에서 Enter=True(로봇 진행), q/ESC/X=False(취소).
    비-execute 모드에선 반환값 무시됨(창은 Enter 로 닫힘)."""
    import json
    if not summary_path.exists():
        print(f"[GUI] summary 없음 — pose 출력 불가: {summary_path}")
        return False
    d = json.load(open(summary_path))
    g = d["grasps"][0]
    bp = g["base_pose"]
    x, y, z = bp["xyz"]
    qx, qy, qz, qw = bp["quat_xyzw"]
    print("\n" + "=" * 52)
    print("  [파지 pose 계산 결과]  (coordinate_frame = world)")
    print(f"   position   x={x:+.4f}  y={y:+.4f}  z={z:+.4f}   [m]")
    print(f"   quat xyzw  [{qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f}]")
    ja = g.get("joint_angles_rad")
    if ja:
        print("   joints[rad] [" + ", ".join(f"{v:+.3f}" for v in ja) + "]")
    print(f"   approach_offset={d.get('approach_offset')}  z_offset={d.get('z_offset')}")
    print(f"   summary : {summary_path}")
    print(f"   overlay : {overlay_path}")
    print("=" * 52 + "\n")

    if not show_result:
        return True

    img = cv2.imread(str(overlay_path))
    if img is None:
        print(f"[GUI] overlay 이미지 없음: {overlay_path}")
        return True

    rwin = "grasp_result"
    if execute:
        print("[GUI] 결과 overlay 창 — Enter=로봇 실행, q/ESC=취소 (창 X버튼=취소).")
    else:
        print("[GUI] 결과 overlay 창 — Enter 를 누르면 닫힘 (창 X버튼도 가능).")
    cv2.namedWindow(rwin, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(rwin, img)
    proceed = False
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):        # Enter / LF → 진행
            proceed = True
            break
        if key in (ord('q'), 27):  # q / ESC → 취소
            proceed = False
            break
        # 창 X버튼으로 닫힌 경우 → 취소
        if cv2.getWindowProperty(rwin, cv2.WND_PROP_VISIBLE) < 1:
            proceed = False
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)   # Qt: 창을 실제로 닫으려면 이벤트 루프 1회 더
    return proceed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="npz (keys: rgb[BGR], depth, K)")
    ap.add_argument("--query", required=True)
    ap.add_argument("--calibration", default=None,
                    help="extrinsic json — 파지 단계로 전달")
    ap.add_argument("--sam3_model_id", default="facebook/sam3")
    ap.add_argument("--sam3_threshold", type=float, default=0.5)
    ap.add_argument("--sam3_mask_threshold", type=float, default=0.5)
    ap.add_argument("--output_dir", default=str(ROOT / "data" / "interim"))
    ap.add_argument("--no_grasp", action="store_true",
                    help="mask 만 저장하고 파지 pose 계산 생략")
    # ── 로봇 실행 (시나리오1: 물체 쥔 채 유지 → Inhand 넘김) ──
    ap.add_argument("--execute_robot", action="store_true",
                    help="파지 pose 계산 후 send_to_robot_scenario1.py 로 실제 로봇 실행. "
                         "없으면 pose 계산까지만 (안전, 시뮬 확인용).")
    ap.add_argument("--execute_mode", default="direct_franka_topic",
                    choices=["trajectory_forwarder", "direct_franka_topic"])
    ap.add_argument("--speed_factor", type=float, default=0.1)
    ap.add_argument("--approach_offset", type=float, default=0.10)
    ap.add_argument("--disable_collision", action="store_true")
    ap.add_argument("--no_record", action="store_true",
                    help="로봇 실행 시 녹화 여부 묻지 않고 건너뜀")
    ap.add_argument("--no_show_result", action="store_true",
                    help="파지 pose 결과 overlay 창을 띄우지 않음 (기본: 띄우고 키 입력 시 닫힘)")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    npz = np.load(str(input_path))
    rgb = npz["rgb"]
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    # npz 는 BGR 저장 → cv2 표시는 그대로, SAM3 입력은 RGB
    base_bgr = rgb.copy()
    image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    dets = detect_all(image_rgb, args.query, args.sam3_model_id,
                      args.sam3_threshold, args.sam3_mask_threshold)
    if not dets:
        print(f"[GUI] {args.query!r} 검출 없음 — 종료.")
        sys.exit(1)
    print(f"[GUI] {len(dets)}개 검출. 창에서 박스를 클릭해 선택하세요.")

    state = {"sel": None}

    def hit_test(x, y):
        """(x,y)를 포함하는 박스 중 가장 작은 것의 인덱스."""
        cands = []
        for i, d in enumerate(dets):
            x0, y0, x1, y1 = d["box"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                cands.append((i, (x1 - x0) * (y1 - y0)))
        if not cands:
            return None
        return min(cands, key=lambda c: c[1])[0]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            i = hit_test(x, y)
            if i is not None:
                state["sel"] = i
                print(f"  선택: #{i}  score={dets[i]['score']:.3f}  box={dets[i]['box']}")

    def render():
        canvas = base_bgr.copy()
        # 선택된 것 마스크 반투명 강조
        sel = state["sel"]
        if sel is not None:
            ov = canvas.copy()
            ov[dets[sel]["mask"]] = SEL_COLOR
            canvas = cv2.addWeighted(ov, 0.45, canvas, 0.55, 0.0)
        for i, d in enumerate(dets):
            x0, y0, x1, y1 = d["box"]
            selected = (i == sel)
            color = SEL_COLOR if selected else BOX_COLORS[i % len(BOX_COLORS)]
            th = 3 if selected else 2
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, th)
            label = f"#{i} {d['score']:.2f}"
            cv2.putText(canvas, label, (x0, max(y0 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # 안내문
        msg1 = "click=select  Enter=confirm  r=reset  q=cancel"
        msg2 = (f"selected #{sel} (score {dets[sel]['score']:.2f})"
                if sel is not None else "no selection")
        cv2.putText(canvas, msg1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.putText(canvas, msg2, (8, base_bgr.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return canvas

    win = "sam3_select"   # ASCII 전용 (유니코드 창이름은 Qt setMouseCallback 조회 실패)
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    # Qt 백엔드: setMouseCallback 전에 창을 실제로 realize 해야 핸들이 생김
    cv2.imshow(win, render())
    cv2.waitKey(1)
    cv2.setMouseCallback(win, on_mouse)

    confirmed = False
    while True:
        cv2.imshow(win, render())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):          # Enter / Space
            if state["sel"] is not None:
                confirmed = True
                break
            print("  [안내] 아직 선택 안 함 — 박스를 먼저 클릭하세요.")
        elif key == ord('r'):
            state["sel"] = None
            print("  선택 초기화.")
        elif key in (ord('q'), 27):  # q / ESC
            print("  [GUI] 취소됨 — 파지 안 함.")
            break
        # 창 X 버튼으로 닫힌 경우
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            print("  [GUI] 창 닫힘 — 취소.")
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)   # Qt: 선택창을 실제로 닫으려면 이벤트 루프 1회 더
    if not confirmed:
        sys.exit(1)

    sel = state["sel"]
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), (dets[sel]["mask"].astype(np.uint8) * 255))
    print(f"[GUI] 선택 #{sel} (score {dets[sel]['score']:.3f}) → mask 저장: {mask_path}")

    if args.no_grasp:
        print("[GUI] --no_grasp: 파지 pose 계산 생략.")
        return

    # ── 1) 파지 pose 계산 (run_topdown_grasp) ──
    grasp_cmd = [sys.executable, str(SCRIPTS / "run_topdown_grasp.py"),
                 "--input", str(input_path),
                 "--mask", str(mask_path),
                 "--query", args.query]
    if args.calibration:
        grasp_cmd += ["--calibration", args.calibration]
    print("[GUI] 파지 pose 계산:\n  " + " ".join(grasp_cmd))
    rc = subprocess.run(grasp_cmd).returncode
    if rc != 0:
        print(f"[GUI] 파지 pose 계산 실패 (rc={rc}) — 로봇 실행 안 함.")
        sys.exit(rc)

    # 파지 pose 로그 출력 + 결과 overlay 창 (execute 모드: Enter=진행, q/ESC=취소)
    summary_path = ROOT / "data" / "outputs" / f"{stem}_topdown_summary.json"
    overlay_path = ROOT / "data" / "outputs" / f"{stem}_topdown_overlay.png"
    proceed = report_pose(summary_path, overlay_path,
                          show_result=not args.no_show_result,
                          execute=args.execute_robot)

    if not args.execute_robot:
        print("[GUI] --execute_robot 없음: pose 계산까지만 (로봇 실행 안 함).")
        return

    if not proceed:
        print("[GUI] 사용자 취소 (q/ESC) — 로봇 실행 안 함.")
        return

    # ── 2) 실제 로봇 실행 (시나리오1: 물체 쥔 채 유지 → Inhand) ──
    if not summary_path.exists():
        print(f"[GUI] summary 없음 → 로봇 실행 불가: {summary_path}")
        sys.exit(1)
    robot_cmd = [sys.executable, str(SCRIPTS / "send_to_robot_scenario1.py"),
                 "--summary_json", str(summary_path),
                 "--execute_mode", args.execute_mode,
                 "--speed_factor", str(args.speed_factor),
                 "--approach_offset", str(args.approach_offset)]
    if args.disable_collision:
        robot_cmd += ["--disable_collision"]
    if args.no_record:
        robot_cmd += ["--no_record"]
    print("[GUI] 로봇 실행 (시나리오1, 물체 쥔 채 유지):\n  " + " ".join(robot_cmd))
    rc = subprocess.run(robot_cmd).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
