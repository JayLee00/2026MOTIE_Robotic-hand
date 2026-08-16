#!/usr/bin/env python3
"""
Robot executor entry point — dispatches to GraspExecutor or PlaceExecutor.

Usage (must be run with ROS2 Humble sourced):
    source /opt/ros/humble/setup.bash
    source kistar_ws/install/local_setup.bash

    # grasp only
    python3 scripts/robot_executor.py \\
        --summary_json data/outputs/scene_topdown_summary.json \\
        --mode grasp

    # pick + place
    python3 scripts/robot_executor.py \\
        --summary_json data/outputs/scene_topdown_summary.json \\
        --mode place \\
        --place_z_descent 0.15

Exits with code 0 on success, 1 on failure.
"""

import argparse
import json
import sys
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from utils.grasp import GraspExecutor
from utils.place import PlaceExecutor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary_json',   required=True,
                   help='Path to {stem}_topdown_summary.json')
    p.add_argument('--mode',           default='grasp',
                   choices=['grasp', 'place'],
                   help='grasp: grasp only  |  place: pick+place via HOME')
    p.add_argument('--execute_mode',   default='direct_franka_topic',
                   choices=['trajectory_forwarder', 'direct_franka_topic'])
    p.add_argument('--speed_factor',   type=float, default=0.1)
    p.add_argument('--approach_offset', type=float, default=0.10)
    p.add_argument('--place_z_descent', type=float, default=None,
                   help='[place mode] HOME EE Z 에서 내려갈 거리 (m). 양수 = 아래 방향.')
    p.add_argument('--disable_collision', action='store_true',
                   help='MoveIt collision 검사 비활성화 (base 이동 후 scene 재설정 전 임시 사용)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'place' and args.place_z_descent is None:
        print('[ERROR] --mode place 사용 시 --place_z_descent 가 필요합니다.')
        sys.exit(1)

    with open(args.summary_json) as f:
        summary = json.load(f)

    rclpy.init()

    if args.mode == 'grasp':
        node = GraspExecutor(
            summary,
            args.execute_mode,
            args.speed_factor,
            args.approach_offset,
            summary_json_path=args.summary_json,
            disable_collision=args.disable_collision,
        )
    else:
        node = PlaceExecutor(
            summary,
            args.execute_mode,
            args.speed_factor,
            args.approach_offset,
            place_z_descent=args.place_z_descent,
            summary_json_path=args.summary_json,
            disable_collision=args.disable_collision,
        )

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node._hold_hand_position(duration=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(0 if node._success else 1)


if __name__ == '__main__':
    main()
