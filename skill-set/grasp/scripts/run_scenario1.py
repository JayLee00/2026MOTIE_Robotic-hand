#!/usr/bin/env python3
"""
시나리오1 — Pick + 시퀀스 제어권 (Pick DONE → Inhand 이어받기).

run_pipeline_interactive.py 와 동일한 파이프라인(ROS 카메라 → SAM3 → PCA → 로봇 pick)을
sequence_client 로 감싼 것. Pick 1회 성공 시 제어권을 반납(End=DONE)해서
다음 시퀀스(Inhand=2)로 넘긴다. (원본 run_pipeline_interactive.py 는 안 건드림)

시퀀스 규칙 (Dual_Arm_Hand_Ctrl/docs_dev/SEQUENCE_GUIDE.md):
  1=Pick, 2=Inhand, 3=Stiffness, 4=Place
  with SequenceClient(SEQ_PICK): 진입=Start(제어권+하트비트), 정상탈출=End(DONE), 예외=abort(IDLE)
  Pick 은 첫 시퀀스라 wait_for_previous_done 없음.

전제:
  - 제어 PC(메인 PC)에서 sequence_arbiter 실행 + require_control:=true 로 launch
  - dual_arm_msgs + sequence_client 빌드/소싱 (colcon build --packages-select dual_arm_msgs sequence_client)
  - export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
    source /opt/ros/<distro>/setup.bash

실행:
    python scripts/run_scenario1.py --camera_source ros --execute_robot \
        --calibration configs/calibration/extrinsic_20260612_170053.json
    # Query> orange  → pick 1회 성공 → Pick DONE → Inhand 차례
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(ROOT / "src"))

from affordance_grasp.io.realsense import RealSenseSession
from pipeline_core import python_bin, stage_grasp, run_stage
from run_pipeline_interactive import Sam3Session, build_parser   # 원본 재사용

from dual_arm_msgs.msg import SequenceState
from sequence_client import SequenceClient, SequenceError


def stage_robot_scenario1(python, args, grasp_json, on_error='continue'):
    """send_to_robot_scenario1.py 호출 — Pick 후 물체 쥔 채 유지 (place 안 씀, Pick 전용)."""
    robot_args = [
        "--summary_json",    str(grasp_json),
        "--execute_mode",    args.execute_mode,
        "--speed_factor",    str(args.speed_factor),
        "--approach_offset", str(args.approach_offset),
        "--kistar_ws",       args.kistar_ws,
    ]
    if getattr(args, 'disable_collision', False):
        robot_args += ["--disable_collision"]
    return run_stage(python, SCRIPTS / "send_to_robot_scenario1.py",
                     robot_args, "Robot (Scenario1-Pick, 물체유지)", on_error=on_error)


def main():
    args    = build_parser().parse_args()
    args.place = False   # 시나리오1은 Pick 전용 — place 강제 OFF (물체 쥔 채 Inhand가 이어받음)
    python  = python_bin(Path(args.conda_base), args.env)
    interim = Path(args.interim_dir)
    outputs = Path(args.output_dir)
    raw_dir = Path(args.camera_raw_dir)
    interim.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    sam3 = Sam3Session(
        args.sam3_model_id, args.sam3_threshold, args.sam3_mask_threshold)

    # 카메라 소스 (원본과 동일 — ROS / RealSense)
    if args.camera_source == "ros":
        from affordance_grasp.io.ros_camera import ROSCameraSource
        print(f"[Camera] ROS 토픽 구독 모드: {args.ros_color_topic}")
        cam_session = ROSCameraSource(
            color_topic=args.ros_color_topic,
            depth_topic=args.ros_depth_topic,
            info_topic=args.ros_info_topic,
        )
    else:
        cam_session = RealSenseSession(
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_frames=args.warmup_frames,
        )

    capture_idx = 0

    print("\n" + "=" * 60)
    print("  Scenario 1 — Pick (시퀀스 제어권)")
    print("  Pick 1회 성공 시 제어권 반납 → Inhand(2) 차례")
    print("  'exit'/'quit'/'q' = Pick 취소(abort)")
    print("=" * 60)

    rclpy.init()   # SequenceClient + ROSCameraSource 공용 (ROSCameraSource는 init-safe)
    seq = SequenceClient(SequenceState.SEQ_PICK)   # Pick=1, 첫 시퀀스라 wait 없음

    try:
        print("\n[Scenario1] 제어권 획득 중 (Start)...")
        with seq:                          # Start: request_control + 하트비트 자동
            with cam_session as cam:
                while True:
                    try:
                        query = input("\n  Query> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        raise SequenceError("Pick 취소 (사용자 중단)")

                    if query.lower() in ("exit", "quit", "q"):
                        raise SequenceError("Pick 취소 (사용자 종료)")   # → abort (DONE 아님)
                    if not query:
                        continue

                    stem = f"scenario1_{capture_idx:03d}"
                    capture_idx += 1

                    # ── Stage 0: 캡처 ──
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    input_path = cam.capture(raw_dir, stem, capture_idx - 1)

                    # ── Stage 1: SAM3 (in-process) ──
                    npz = np.load(str(input_path))
                    rgb = npz["rgb"]
                    if rgb.dtype != np.uint8:
                        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                    image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                    result = sam3.segment(image_rgb, query)
                    if result is None or not result["mask"].any():
                        print("  [WARN] 마스크 없음 — 다시 입력하세요.")
                        continue

                    # 마스크/overlay 저장 (원본과 동일)
                    input_stem   = input_path.stem
                    mask_path    = interim / f"{input_stem}_mask.png"
                    overlay_path = interim / f"{input_stem}_overlay.png"
                    cv2.imwrite(str(mask_path),
                                (result["mask"].astype(np.uint8) * 255))
                    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    ov     = canvas.copy()
                    ov[result["mask"]] = (0, 0, 220)
                    cv2.imwrite(str(overlay_path),
                                cv2.addWeighted(ov, 0.45, canvas, 0.55, 0.0))

                    sam3_json = interim / f"{input_stem}_sam3.json"
                    with open(sam3_json, "w") as f:
                        json.dump({
                            "stem":  input_stem,
                            "query": query,
                            "sam3":  {
                                "model":       args.sam3_model_id,
                                "used_query":  query,
                                "score":       result["score"],
                                "box_xyxy":    result["box_xyxy"],
                                "mask_pixels": int(result["mask"].sum()),
                            },
                        }, f, indent=2, ensure_ascii=False)

                    # ── Stage 2: Grasp ──
                    grasp_json = stage_grasp(
                        python, args, input_path, mask_path, outputs,
                        query=query, on_error='continue')
                    if grasp_json is None:
                        print("  [WARN] 파지 계산 실패 — 다시 입력하세요.")
                        continue

                    # ── Stage 3: Robot (선택) — 물체 쥔 채 유지 (scenario1) ──
                    if args.execute_robot:
                        stage_robot_scenario1(python, args, grasp_json, on_error='continue')

                    print(f"  ✓ Pick 성공: query={query!r}  →  {grasp_json.name}")
                    break   # 1회 성공 → with 정상 탈출 → End(DONE) → Inhand 이어받음

        # with seq 정상 탈출 = End (release_control → DONE)
        print("\n[Scenario1] ✅ Pick DONE — 제어권 반납. Inhand(2) 차례.")

    except SequenceError as e:
        # 예외 탈출 = abort (release 없이 → 3초 후 IDLE 회수, DONE 아님)
        print(f"\n[Scenario1] ⚠️ Pick 중단: {e}  (abort → IDLE 회수, Inhand 진행 안 함)")
    finally:
        seq.shutdown()
        sam3.close()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
