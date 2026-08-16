#!/usr/bin/env python3
"""Shared utilities for Grasp_fruit pipeline scripts.

Provides:
  - python_bin / run_stage   : subprocess helpers
  - add_*_args               : argparse group builders
  - stage_capture            : RealSense → NPZ
  - stage_sam3_only          : SAM3 text query → mask
  - stage_grasp              : top-down grasp → summary JSON
  - stage_robot              : Docker exec → robot executor
"""

import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent
ROOT    = SCRIPTS.parent

sys.path.insert(0, str(SCRIPTS))
from utils.paths import CONDA_BASE as DEFAULT_CONDA_BASE, CONDA_ENV as DEFAULT_ENV, KISTAR_WS as DEFAULT_KISTAR_WS
from utils.arm import (
    APPROACH_OFFSET_M as DEFAULT_APPROACH_OFFSET,
    PLACE_Z_DESCENT_M as DEFAULT_PLACE_Z_DESCENT,
    GRASP_Z_OFFSET_M  as DEFAULT_GRASP_Z_OFFSET,
)

# configs/camera/realsense.yaml
_rs_cfg = yaml.safe_load((ROOT / "configs" / "camera" / "realsense.yaml").read_text())
_CAM_WIDTH   = int(_rs_cfg["width"])
_CAM_HEIGHT  = int(_rs_cfg["height"])
_CAM_FPS     = int(_rs_cfg["fps"])
_CAM_WARMUP  = int(_rs_cfg["warmup_frames"])


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def python_bin(conda_base: Path, env_name: str) -> Path:
    p = conda_base / "envs" / env_name / "bin" / "python"
    if not p.exists():
        raise FileNotFoundError(
            f"Python not found at {p}.\n"
            f"  bash setup_pipeline_all.sh")
    return p


def run_stage(python: Path, script: Path, extra_args: list[str],
              stage_name: str, on_error: str = 'exit') -> bool:
    """Run one pipeline stage as a subprocess.

    on_error:
      'exit'     → sys.exit(returncode) on failure  (default)
      'continue' → return False on failure
    Returns True on success.
    """
    cmd = [str(python), str(script)] + extra_args
    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] {stage_name} 실패 (exit {result.returncode}).")
        if on_error == 'exit':
            sys.exit(result.returncode)
        return False
    return True


# ---------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------

def add_conda_args(p) -> None:
    p.add_argument("--conda_base", default=str(DEFAULT_CONDA_BASE))
    p.add_argument("--env",        default=DEFAULT_ENV)


def add_camera_args(p) -> None:
    cam = p.add_argument_group("Camera (RealSense) — defaults from configs/camera/realsense.yaml")
    cam.add_argument("--warmup_frames",  type=int, default=_CAM_WARMUP)
    cam.add_argument("--camera_width",   type=int, default=_CAM_WIDTH)
    cam.add_argument("--camera_height",  type=int, default=_CAM_HEIGHT)
    cam.add_argument("--camera_fps",     type=int, default=_CAM_FPS)
    cam.add_argument("--camera_raw_dir", default=str(ROOT / "data" / "raw"))


def add_sam3_args(p) -> None:
    p.add_argument("--sam3_model_id",       default="facebook/sam3")
    p.add_argument("--sam3_threshold",      type=float, default=0.5)
    p.add_argument("--sam3_mask_threshold", type=float, default=0.5)


def add_grasp_args(p, z_offset: float = DEFAULT_GRASP_Z_OFFSET) -> None:
    gsp = p.add_argument_group("Grasp")
    gsp.add_argument("--depth_scale", type=float, default=1.0)
    gsp.add_argument("--z_offset",    type=float, default=z_offset)
    gsp.add_argument("--x_offset",    type=float, default=None)  #추가
    gsp.add_argument("--y_offset",    type=float, default=None)  #추가
    gsp.add_argument("--hand_pose",   default=None, metavar="JSON")
    gsp.add_argument("--calibration", default=None, metavar="JSON")
    gsp.add_argument("--robot_base_x",     type=float, default=None)
    gsp.add_argument("--robot_base_y",     type=float, default=None)
    gsp.add_argument("--robot_base_z",     type=float, default=None)
    gsp.add_argument("--robot_base_roll",  type=float, default=None)
    gsp.add_argument("--robot_base_pitch", type=float, default=None)
    gsp.add_argument("--robot_base_yaw",   type=float, default=None)
    gsp.add_argument("--preview",     action="store_true")


def add_robot_args(p) -> None:
    rob = p.add_argument_group("Robot")
    rob.add_argument("--execute_robot",   action="store_true")
    rob.add_argument("--execute_mode",    default="direct_franka_topic",
                     choices=["trajectory_forwarder", "direct_franka_topic"])
    rob.add_argument("--speed_factor",    type=float, default=0.1)
    rob.add_argument("--approach_offset", type=float, default=DEFAULT_APPROACH_OFFSET)
    rob.add_argument("--place", action="store_true",
                     help=f"pick+place 모드 활성화. 하강 거리는 configs/arm.yaml "
                          f"place_z_descent_m={DEFAULT_PLACE_Z_DESCENT} m 사용")
    rob.add_argument("--kistar_ws", default=DEFAULT_KISTAR_WS)
    rob.add_argument("--disable_collision", action="store_true",
                     help="MoveIt collision 검사 비활성화 (base 이동 후 임시 테스트용)")


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_capture(python: Path, args, stem: str, output_dir: Path,
                  warmup_frames: int = None,
                  on_error: str = 'exit') -> 'Path | None':
    """RealSense 캡처 → NPZ. 성공 시 NPZ Path, 실패 시 None (on_error='continue') 또는 sys.exit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wf = warmup_frames if warmup_frames is not None else args.warmup_frames
    ok = run_stage(python, SCRIPTS / "capture_realsense_once.py", [
        "--output_dir",    str(output_dir),
        "--stem",          stem,
        "--width",         str(args.camera_width),
        "--height",        str(args.camera_height),
        "--fps",           str(args.camera_fps),
        "--warmup_frames", str(wf),
    ], f"Capture: {stem}", on_error=on_error)
    if not ok:
        return None
    npz = output_dir / f"{stem}_000.npz"
    if not npz.exists():
        print(f"[ERROR] 캡처된 NPZ 없음: {npz}")
        if on_error == 'exit':
            sys.exit(1)
        return None
    return npz


def stage_sam3_only(python: Path, args, input_path: Path, output_dir: Path,
                    query: str = None,
                    on_error: str = 'exit') -> 'Path | None':
    """SAM3 text query → mask PNG. 실패 시 None 또는 sys.exit."""
    q = query if query is not None else args.query
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = run_stage(python, SCRIPTS / "run_sam3_only_stage.py", [
        "--input",               str(input_path),
        "--query",               q,
        "--sam3_model_id",       args.sam3_model_id,
        "--sam3_threshold",      str(args.sam3_threshold),
        "--sam3_mask_threshold", str(args.sam3_mask_threshold),
        "--output_dir",          str(output_dir),
    ], f"SAM3-only: {q!r}", on_error=on_error)
    if not ok:
        return None
    mask = output_dir / f"{input_path.stem}_mask.png"
    if not mask.exists():
        print(f"[ERROR] 마스크 없음: {mask}")
        if on_error == 'exit':
            sys.exit(1)
        return None
    return mask


def stage_grasp(python: Path, args, input_path: Path,
                mask_path: Path, output_dir: Path,
                query: str = None,
                on_error: str = 'exit') -> 'Path | None':
    """Top-down grasp 계산 → summary JSON. 실패 시 None 또는 sys.exit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_args = [
        "--input",       str(input_path),
        "--mask",        str(mask_path),
        "--depth_scale", str(args.depth_scale),
        "--z_offset",    str(args.z_offset),
        "--output",      str(output_dir),
    ]
    if getattr(args, 'x_offset', None) is not None:        #추가
        stage_args += ["--x_offset", str(args.x_offset)]   #추가
    if getattr(args, 'y_offset', None) is not None:        #추가
        stage_args += ["--y_offset", str(args.y_offset)]   #추가
    q = query or getattr(args, 'query', None)
    if q:
        stage_args += ["--query", q]
    if args.calibration:
        stage_args += ["--calibration", args.calibration]
    if getattr(args, 'hand_pose', None):
        stage_args += ["--hand_pose", args.hand_pose]
    rbase = {
        "--robot_base_x":     getattr(args, 'robot_base_x', None),
        "--robot_base_y":     getattr(args, 'robot_base_y', None),
        "--robot_base_z":     getattr(args, 'robot_base_z', None),
        "--robot_base_roll":  getattr(args, 'robot_base_roll', None),
        "--robot_base_pitch": getattr(args, 'robot_base_pitch', None),
        "--robot_base_yaw":   getattr(args, 'robot_base_yaw', None),
    }
    if all(v is not None for v in rbase.values()):
        for flag, val in rbase.items():
            stage_args += [flag, str(val)]
    if getattr(args, 'preview', False):
        stage_args += ["--preview"]

    ok = run_stage(python, SCRIPTS / "run_topdown_grasp.py",
                   stage_args, f"Grasp: {input_path.stem}", on_error=on_error)
    if not ok:
        return None
    grasp_json = output_dir / f"{input_path.stem}_topdown_summary.json"
    if not grasp_json.exists():
        print(f"[ERROR] grasp summary 없음: {grasp_json}")
        if on_error == 'exit':
            sys.exit(1)
        return None
    return grasp_json


def stage_robot(python: Path, args, grasp_json: Path,
                label: str = '', on_error: str = 'exit',
                no_record: bool = False) -> bool:
    """로봇 실행 (Docker exec → send_to_robot.py --mode grasp|place)."""
    use_place = getattr(args, 'place', False)
    mode      = 'place' if use_place else 'grasp'

    robot_args = [
        "--summary_json",    str(grasp_json),
        "--execute_mode",    args.execute_mode,
        "--speed_factor",    str(args.speed_factor),
        "--approach_offset", str(args.approach_offset),
        "--kistar_ws",       args.kistar_ws,
    ]
    if use_place:
        robot_args += ["--place"]
    if getattr(args, 'disable_collision', False):
        robot_args += ["--disable_collision"]
    if no_record:
        robot_args += ["--no_record"]

    name = f"Robot ({mode.capitalize()})" + (f" {label}" if label else "")
    return run_stage(python, SCRIPTS / "send_to_robot.py",
                     robot_args, name, on_error=on_error)
