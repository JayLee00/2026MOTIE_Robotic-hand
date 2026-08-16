#!/usr/bin/env python3
"""
Grasp_fruit pipeline: (RealSense 캡처 또는 파일 입력) → SAM3 → Top-down Grasp → (로봇 실행)

[파일 입력]
    python scripts/run_pipeline.py \\
        --input data/raw/scene.npz \\
        --query "apple" \\
        --calibration configs/calibration/extrinsic_20260612_170053.json

[카메라 캡처 + 로봇]
    python scripts/run_pipeline.py \\
        --capture \\
        --query "apple" \\
        --calibration configs/calibration/extrinsic_20260612_170053.json \\
        --execute_robot

[카메라 캡처 + Place]
    python scripts/run_pipeline.py \\
        --capture \\
        --query "apple" \\
        --calibration configs/calibration/extrinsic_20260612_170053.json \\
        --execute_robot --place
"""

import argparse
import json
from pathlib import Path

from pipeline_core import (
    ROOT,
    python_bin,
    add_conda_args, add_camera_args, add_sam3_args,
    add_grasp_args, add_robot_args,
    stage_capture, stage_sam3_only, stage_grasp, stage_robot,
)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",   metavar="NPZ",
                     help="RGB-D NPZ 파일 (오프라인 모드)")
    src.add_argument("--capture", action="store_true",
                     help="RealSense로 직접 촬영")
    p.add_argument("--camera_stem", default="capture",
                   help="캡처 파일 stem (default: capture → capture_000.npz)")

    p.add_argument("--query", required=True, help="SAM3 텍스트 쿼리 (예: apple)")

    add_conda_args(p)
    add_camera_args(p)
    add_sam3_args(p)
    add_grasp_args(p)
    add_robot_args(p)
    p.add_argument("--interim_dir", default=str(ROOT / "data" / "interim"))
    p.add_argument("--output_dir",  default=str(ROOT / "data" / "outputs"))
    return p


def main():
    args    = build_parser().parse_args()
    python  = python_bin(Path(args.conda_base), args.env)
    interim = Path(args.interim_dir)
    outputs = Path(args.output_dir)

    # ── Stage 0: 입력 소스 ──────────────────────────────────────────────────
    if args.capture:
        raw_dir    = Path(args.camera_raw_dir)
        input_path = stage_capture(python, args, args.camera_stem, raw_dir)
    else:
        input_path = Path(args.input)

    # ── Stage 1: 비전 ───────────────────────────────────────────────────────
    mask_path = stage_sam3_only(python, args, input_path, interim)

    # ── Stage 2: Grasp ──────────────────────────────────────────────────────
    grasp_json = stage_grasp(python, args, input_path, mask_path, outputs)

    # ── Stage 3: Robot (선택) ───────────────────────────────────────────────
    if args.execute_robot:
        stage_robot(python, args, grasp_json)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Pipeline complete")
    print(f"  Input  : {input_path}")
    print(f"  Query  : {args.query!r}")
    print(f"  Output : {grasp_json}")

    if grasp_json.exists():
        with open(grasp_json) as f:
            ginfo = json.load(f)
        grasps = ginfo.get("grasps", [])
        if grasps:
            bp  = grasps[0].get("base_pose", {})
            xyz = bp.get("xyz", [])
            enc = grasps[0].get("joint_angles_enc", [])
            print(f"  EE pos : {[round(v, 3) for v in xyz]}")
            print(f"  Hand   : {enc}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
