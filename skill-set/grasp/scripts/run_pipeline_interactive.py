#!/usr/bin/env python3
"""
Interactive pipeline — SAM3 모델을 한 번 로드 후 query를 반복 입력받아 실행.

매 반복:
  Stage 0: RealSense 캡처
  Stage 1: SAM3 추론 (in-process, 재로드 없음)
  Stage 2: Top-down grasp 계산 (subprocess)
  Stage 3: 로봇 실행 (subprocess, --execute_robot 시)

'exit' 입력 시 종료.

Usage:
    conda activate pipeline_all
    cd HARILAB/Grasp_fruit

    python scripts/run_pipeline_interactive.py \\
        --calibration configs/calibration/extrinsic_20260612_170053.json

    python scripts/run_pipeline_interactive.py \\
        --calibration configs/calibration/extrinsic_20260612_170053.json \\
        --execute_robot --place
"""

import argparse
import json

import cv2
import numpy as np
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT    = SCRIPTS.parent
import sys; sys.path.insert(0, str(ROOT / "src"))
from affordance_grasp.io.realsense import RealSenseSession

from pipeline_core import (
    ROOT,
    python_bin,
    add_conda_args, add_camera_args, add_sam3_args, add_grasp_args, add_robot_args,
    stage_grasp, stage_robot,
)


# ---------------------------------------------------------------------------
# SAM3 세션 (in-process, 모델 상주)
# ---------------------------------------------------------------------------

class Sam3Session:
    """SAM3 모델을 메모리에 상주시켜 반복 추론 (subprocess 오버헤드 없음)."""

    def __init__(self, model_id: str, threshold: float, mask_threshold: float):
        import torch
        from transformers import Sam3Processor, Sam3Model

        self.threshold      = threshold
        self.mask_threshold = mask_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SAM3] 로딩 중: {model_id}  ({self.device})")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model     = Sam3Model.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()
        print("[SAM3] 로딩 완료.")

    def segment(self, image_rgb: np.ndarray, query: str) -> 'dict | None':
        import torch

        img_in = self.processor(
            images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vis = self.model.get_vision_features(
                pixel_values=img_in.pixel_values)

        txt_in = self.processor(text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(vision_embeds=vis, **txt_in)

        h, w = image_rgb.shape[:2]
        res = self.processor.post_process_instance_segmentation(
            out,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(h, w)],
        )[0]

        masks  = res.get("masks", [])
        scores = res.get("scores", [])
        boxes  = res.get("boxes", [])

        if not len(masks):
            print(f"  [SAM3] {query!r} — 검출 없음")
            return None

        best    = int(np.argmax([float(s) for s in scores]))
        mask_np = np.array(masks[best].cpu()).astype(bool)
        score   = float(scores[best])
        print(f"  [SAM3] {query!r}  score={score:.3f}  px={mask_np.sum()}")
        return {
            "mask":     mask_np,
            "score":    score,
            "box_xyxy": boxes[best].cpu().tolist(),
        }

    def close(self):
        import torch
        del self.model
        torch.cuda.empty_cache()
        print("[SAM3] 모델 해제 완료.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_conda_args(p)
    add_camera_args(p)
    add_sam3_args(p)
    add_grasp_args(p)
    add_robot_args(p)
    p.add_argument("--interim_dir", default=str(ROOT / "data" / "interim"))
    p.add_argument("--output_dir",  default=str(ROOT / "data" / "outputs"))
    # 카메라 소스: realsense(USB 직접) | ros(토픽 구독)  #추가
    p.add_argument("--camera_source", choices=["realsense", "ros"], default="realsense")  #추가
    p.add_argument("--ros_color_topic", default="/front_cam/front/color/image_raw")            #추가
    p.add_argument("--ros_depth_topic", default="/front_cam/front/aligned_depth_to_color/image_raw")  #추가
    p.add_argument("--ros_info_topic",  default="/front_cam/front/color/camera_info")          #추가
    return p


def main():
    args    = build_parser().parse_args()
    python  = python_bin(Path(args.conda_base), args.env)
    interim = Path(args.interim_dir)
    outputs = Path(args.output_dir)
    raw_dir = Path(args.camera_raw_dir)
    interim.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    sam3 = Sam3Session(
        args.sam3_model_id, args.sam3_threshold, args.sam3_mask_threshold)

    capture_idx = 0

    print("\n" + "="*60)
    print("  Interactive Pipeline 준비 완료")
    print("  'exit' / 'quit' / 'q' 입력 시 종료")
    print("="*60)

    # 비디오 녹화 여부 (세션 시작 전 1회 질문)
    video_path = None
    try:
        ans = input("\n  비디오 녹화 하시겠습니까? (yes/no) > ").strip().lower()
        if ans in ("yes", "y"):
            video_path = outputs / "interactive_session.mp4"
            print(f"  [녹화] {video_path}")
    except (EOFError, KeyboardInterrupt):
        pass

    if args.camera_source == "ros":                                    #추가
        from affordance_grasp.io.ros_camera import ROSCameraSource      #추가
        print(f"[Camera] ROS 토픽 구독 모드: {args.ros_color_topic}")   #추가
        cam_session = ROSCameraSource(                                  #추가
            color_topic=args.ros_color_topic,                          #추가
            depth_topic=args.ros_depth_topic,                          #추가
            info_topic=args.ros_info_topic,                            #추가
            video_path=video_path,                                     #추가
        )                                                              #추가
    else:                                                              #추가
        cam_session = RealSenseSession(
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_frames=args.warmup_frames,
            video_path=video_path,
        )

    try:
        with cam_session as cam:
            while True:
                print()
                try:
                    query = input("  Query> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Interactive] 종료.")
                    break

                if query.lower() in ("exit", "quit", "q"):
                    print("[Interactive] 종료.")
                    break
                if not query:
                    continue

                stem = f"orange_{capture_idx:03d}"
                #stem = f"interactive_{capture_idx:03d}"
                capture_idx += 1

                # ── Stage 0: 캡처 (파이프라인 유지 — AE/AWB 재수렴 없음) ──
                raw_dir.mkdir(parents=True, exist_ok=True)
                input_path = cam.capture(raw_dir, stem, capture_idx - 1)

                # ── Stage 1: SAM3 (in-process) ─────────────────────────
                npz = np.load(str(input_path))
                rgb = npz["rgb"]
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                result = sam3.segment(image_rgb, query)
                if result is None or not result["mask"].any():
                    print("  [WARN] 마스크 없음 — 다음 query를 입력하세요.")
                    continue

                # 마스크/overlay 저장
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

                # ── Stage 2: Grasp ─────────────────────────────────────────
                grasp_json = stage_grasp(
                    python, args, input_path, mask_path, outputs,
                    query=query, on_error='continue')
                if grasp_json is None:
                    continue

                # ── Stage 3: Robot (선택) ──────────────────────────────────
                if args.execute_robot:
                    stage_robot(python, args, grasp_json, on_error='continue',
                                no_record=(video_path is not None))

                print(f"  ✓ 완료: query={query!r}  →  {grasp_json.name}")

    finally:
        sam3.close()


if __name__ == "__main__":
    main()
