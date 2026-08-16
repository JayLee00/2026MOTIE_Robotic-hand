#!/usr/bin/env python3
"""inhand_sequence.py 테스트용 가짜 sequence 상태 publisher.

실제 sequence_arbiter 없이 /sequence/shm_state (std_msgs/Int32MultiArray,
data=[seq_id, state, owner])를 arbiter와 동일한 latched(TRANSIENT_LOCAL) QoS로
발행한다. SEQUENCE_GUIDE.md §3의 대체 토픽과 동일 포맷.

배정 번호: 1=Pick, 2=Inhand, 3=Stiffness, 4=Place
상태값   : 0=IDLE · 1=RUNNING · 2=DONE

기본 동작:
    Inhand 시작 상태 [seq_id=2, state=1(RUNNING), owner=2]를 계속 발행
    → inhand_sequence.py가 "in-hand manipulation start"를 트리거해야 함.

사용 예:
    source /opt/ros/humble/setup.bash

    # 1) 기본: Inhand RUNNING 상태를 계속 발행 (Ctrl+C로 종료)
    /usr/bin/python3 scripts/fake_sequence_publisher.py

    # 2) 임의 상태 지정
    /usr/bin/python3 scripts/fake_sequence_publisher.py --seq-id 2 --state 1 --owner 2

    # 3) 체이닝 시뮬레이션: Pick DONE → (지연) → Inhand RUNNING
    /usr/bin/python3 scripts/fake_sequence_publisher.py --sequence
"""

import argparse

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Int32MultiArray

# arbiter가 쓰는 latched QoS와 동일해야 구독자가 마지막 샘플을 받는다
LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

IDLE, RUNNING, DONE = 0, 1, 2
SEQ_PICK, SEQ_INHAND = 1, 2


class FakeSequencePublisher(Node):
    def __init__(self, rate_hz=2.0):
        super().__init__('fake_sequence_publisher')
        self._pub = self.create_publisher(
            Int32MultiArray, '/sequence/shm_state', LATCHED_QOS)
        self._current = None
        self._period = 1.0 / rate_hz
        # latched지만 늦게 뜨는 구독자를 위해 주기적으로도 재발행
        self.create_timer(self._period, self._republish)

    def publish_state(self, seq_id, state, owner):
        self._current = [int(seq_id), int(state), int(owner)]
        self._pub.publish(Int32MultiArray(data=self._current))
        self.get_logger().info(
            f'/sequence/shm_state <- [seq_id={seq_id}, state={state}, owner={owner}]')

    def _republish(self):
        if self._current is not None:
            self._pub.publish(Int32MultiArray(data=self._current))


def main():
    parser = argparse.ArgumentParser(description='가짜 sequence 상태 publisher')
    parser.add_argument('--seq-id', type=int, default=SEQ_INHAND)
    parser.add_argument('--state', type=int, default=RUNNING)
    parser.add_argument('--owner', type=int, default=SEQ_INHAND)
    parser.add_argument('--rate', type=float, default=2.0, help='재발행 주기 [Hz]')
    parser.add_argument('--sequence', action='store_true',
                        help='Pick DONE → 2초 후 Inhand RUNNING 순으로 발행')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='--sequence 모드에서 Pick DONE 유지 시간 [s]')
    args = parser.parse_args()

    rclpy.init()
    node = FakeSequencePublisher(rate_hz=args.rate)
    try:
        if args.sequence:
            # 1) 직전 시퀀스 Pick 정상 종료
            node.publish_state(SEQ_PICK, DONE, 0)
            # delay 동안 spin (재발행 타이머 동작)
            end = node.get_clock().now().nanoseconds + int(args.delay * 1e9)
            while rclpy.ok() and node.get_clock().now().nanoseconds < end:
                rclpy.spin_once(node, timeout_sec=0.1)
            # 2) Inhand 시작 → inhand_sequence.py 트리거 대상
            node.publish_state(SEQ_INHAND, RUNNING, SEQ_INHAND)
        else:
            node.publish_state(args.seq_id, args.state, args.owner)

        node.get_logger().info('발행 중... (Ctrl+C로 종료)')
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
