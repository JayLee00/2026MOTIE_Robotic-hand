#!/usr/bin/env python3
"""테스트용 최소 sequence arbiter — inhand_sequence.py(SequenceClient) 검증용.

실제 sequence_arbiter 없이 시퀀스 프로토콜의 happy-path를 흉내낸다:
  - /sequence_state (SequenceState, latched TRANSIENT_LOCAL) 게시
  - /sequence/request_control (RequestControl) 서비스 = Start → RUNNING 게시
  - /sequence/release_control (ReleaseControl) 서비스 = End   → DONE 게시
  - 시작 시 Pick(1) DONE을 게시해 Inhand(2)가 바로 이어받게 한다

주의: 하트비트 타임아웃/실패 회수는 구현하지 않은 happy-path 전용 스텁이다.
(정식 arbiter는 하트비트가 끊기면 IDLE로 회수한다 — SEQUENCE_GUIDE.md 참조.)

사용 (dual_arm_msgs source 필요, /usr/bin/python3):
    source /opt/ros/humble/setup.bash
    source ~/isaac_ws/dex_soldering/dex_ros/isaac-ros/kistar_ws/install/setup.bash
    /usr/bin/python3 scripts/fake_arbiter.py
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from dual_arm_msgs.msg import SequenceState
from dual_arm_msgs.srv import ReleaseControl, RequestControl

LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class FakeArbiter(Node):
    def __init__(self):
        super().__init__('fake_arbiter')
        self._pub = self.create_publisher(SequenceState, '/sequence_state', LATCHED_QOS)
        self.create_service(RequestControl, '/sequence/request_control', self._on_request)
        self.create_service(ReleaseControl, '/sequence/release_control', self._on_release)
        self._owner = 0
        # 시작 상태: Pick(1) 정상 종료 → Inhand(2)가 이어받을 수 있음
        self._publish(SequenceState.SEQ_PICK, SequenceState.DONE, 0)
        self.get_logger().info('fake arbiter up (Pick(1) DONE 게시, request/release 대기)')

    def _publish(self, seq_id, state, owner):
        msg = SequenceState()
        msg.seq_id = int(seq_id)
        msg.state = int(state)
        msg.owner = int(owner)
        msg.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)
        self.get_logger().info(
            f'/sequence_state <- seq_id={seq_id} state={state} owner={owner}')

    def _on_request(self, req, resp):
        # 다른 owner가 잡고 있으면 거부, 아니면 승인 → RUNNING 게시
        if self._owner not in (0, req.client_id):
            resp.granted = False
            resp.current_owner = self._owner
            resp.message = f'busy (owner={self._owner})'
            self.get_logger().warn(
                f'request 거부: client={req.client_id} seq={req.seq_id} (owner={self._owner})')
            return resp
        self._owner = req.client_id
        self._publish(req.seq_id, SequenceState.RUNNING, req.client_id)
        resp.granted = True
        resp.current_owner = req.client_id
        resp.message = 'granted'
        return resp

    def _on_release(self, req, resp):
        # owner 반납 → DONE 게시 (다음 시퀀스가 이어받음)
        self._publish(req.seq_id, SequenceState.DONE, 0)
        self._owner = 0
        resp.released = True
        resp.message = 'released'
        return resp


def main():
    rclpy.init()
    node = FakeArbiter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
