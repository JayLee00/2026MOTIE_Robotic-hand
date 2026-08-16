#!/usr/bin/env python3
"""시퀀스 1(Pick)의 Start/End 신호 최소 예제 — GPU PC 개발자용 템플릿.

Pick은 첫 시퀀스이므로 선행 대기(wait_for_previous_done)가 없다.
2번(Inhand) 이후의 체이닝 예시는 run_sequence.py 참조.

사전 조건: 제어 PC에서 sequence_arbiter 실행 중
    (control_pc.launch.py 또는 ros2 run trajectory_receiver sequence_arbiter_node)

사용 예:
    ros2 run sequence_client pick_sequence_example [--work-sec 3]

NOTE: 현재는 각 시퀀스가 latched /sequence_state를 보고 스스로 체이닝하지만,
추후 오케스트레이터 노드가 순서를 지시하는 방식으로 대체될 수 있음.
"""

import argparse
import sys
import time

import rclpy

from dual_arm_msgs.msg import SequenceState
from sequence_client import SequenceClient, SequenceError


def main():
    parser = argparse.ArgumentParser(
        description="시퀀스 1(Pick) Start/End 예제")
    parser.add_argument("--work-sec", type=float, default=3.0,
                        help="실제 동작을 대신하는 placeholder 시간 [s]")
    args = parser.parse_args()

    rclpy.init()
    client = SequenceClient(SequenceState.SEQ_PICK)  # client_id = seq_id = 1
    try:
        # with 진입 = Start(S): request_control 승인 + 하트비트 자동 발행
        # 정상 탈출 = End(E): release_control → DONE (다음 시퀀스가 이어받음)
        # 예외 탈출 = abort(): release 없이 하트비트 정지 → 3초 후 자동 회수(IDLE)
        with client:
            # ── 여기서 실제 Pick 동작 수행 ──
            # 타겟 입구는 docs_dev/ROS2_TOPIC_GUIDE.md §2 참조
            # (/franka/<side>/ee_target_world, /hand/<side>/q_target 등)
            time.sleep(args.work_sec)
        print("sq1(Pick) 완료: End(E) 전송됨")
    except SequenceError as e:
        print(f"sq1(Pick) 실패: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
