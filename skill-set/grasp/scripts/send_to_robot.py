#!/usr/bin/env python3
"""
Send grasp or pick-place task to robot via Docker container.

Usage:
    # grasp only (default)
    python scripts/send_to_robot.py \\
        --summary_json data/outputs/scene_topdown_summary.json

    # pick + place
    python scripts/send_to_robot.py \\
        --summary_json data/outputs/scene_topdown_summary.json \\
        --mode place \\
        --place_z_descent 0.15
"""

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from utils.paths import KISTAR_WS as DEFAULT_KISTAR_WS
from utils.arm import PLACE_Z_DESCENT_M

from docker_runner import (
    DOCKER_CONTAINER,
    to_container_path, ensure_running,
    ros_exec_cmd, run_in_container,
    ask_and_record, stop_recording,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary_json",   required=True,
                   help="Path to topdown_summary.json (호스트 경로)")
    p.add_argument("--place", action="store_true",
                   help=f"pick+place 모드. 하강 거리는 arm.yaml ({PLACE_Z_DESCENT_M} m) 사용")
    p.add_argument("--execute_mode",   default="direct_franka_topic",
                   choices=["trajectory_forwarder", "direct_franka_topic"])
    p.add_argument("--speed_factor",   type=float, default=0.1)
    p.add_argument("--approach_offset", type=float, default=0.10)
    p.add_argument("--disable_collision", action="store_true",
                   help="MoveIt collision 검사 비활성화 (base 이동 후 임시 테스트용)")
    p.add_argument("--container",  default=DOCKER_CONTAINER)
    p.add_argument("--kistar_ws",  default=DEFAULT_KISTAR_WS)
    p.add_argument("--no_record",  action="store_true",
                   help="녹화 여부 묻지 않고 건너뜀 (세션 녹화 중일 때 pipeline이 설정)")
    return p.parse_args()


def main():
    args = parse_args()

    mode = "place" if args.place else "grasp"

    summary_host  = str(Path(args.summary_json).resolve())
    executor_ctr  = to_container_path(str(SCRIPTS / "robot_executor.py"))
    summary_ctr   = to_container_path(summary_host)
    kistar_ws_ctr = to_container_path(args.kistar_ws)

    print(f"[send_to_robot] Docker exec → {args.container}  mode={mode}")
    print(f"  summary (host): {summary_host}")
    print(f"  summary (ctr) : {summary_ctr}")

    if not Path(summary_host).exists():
        print(f"[ERROR] summary_json 없음: {summary_host}")
        sys.exit(1)

    extra = (
        f"--mode {mode} "
        f"--execute_mode {args.execute_mode} "
        f"--speed_factor {args.speed_factor} "
        f"--approach_offset {args.approach_offset}"
    )
    if args.place:
        extra += f" --place_z_descent {PLACE_Z_DESCENT_M}"
    if args.disable_collision:
        extra += " --disable_collision"

    ensure_running(args.container)
    if args.no_record:
        stop_event, thread = None, None
    else:
        stop_event, thread, _ = ask_and_record()

    bash_cmd = ros_exec_cmd(executor_ctr, summary_ctr, kistar_ws_ctr,
                            extra_args=extra)
    rc = run_in_container(args.container, bash_cmd)
    stop_recording(stop_event, thread)

    if rc == 0:
        print(f"\n[send_to_robot] {mode} 완료.")
    else:
        print(f"\n[send_to_robot] 종료 코드: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
