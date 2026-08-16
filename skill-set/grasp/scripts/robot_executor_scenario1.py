#!/usr/bin/env python3
"""
시나리오1용 Robot executor 진입점 — 원본 robot_executor.py 의 복사본.
grasp 모드에서 GraspExecutor 대신 GraspScenario1Executor(물체 쥔 채 유지)를 쓴다.
원본 robot_executor.py 는 건드리지 않는다.

Usage (ROS2 Humble sourced):
    python3 scripts/robot_executor_scenario1.py --summary_json ... --mode grasp
"""

import argparse
import json
import sys
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from utils.grasp_scenario1 import GraspScenario1Executor   # ← 원본과 다른 부분
from utils.place import PlaceExecutor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary_json',   required=True,
                   help='Path to {stem}_topdown_summary.json')
    p.add_argument('--mode',           default='grasp',
                   choices=['grasp', 'place'],
                   help='grasp: grasp only (물체 유지)  |  place: pick+place')
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
        node = GraspScenario1Executor(     # ← 원본과 다른 부분 (물체 쥔 채 유지)
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
