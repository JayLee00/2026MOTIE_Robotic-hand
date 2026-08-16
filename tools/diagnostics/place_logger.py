#!/usr/bin/env python3
"""물체 내려놓기(Place) 진행 상황 로거 — 산업부 PC 터미널용.

Current PC의 place skill 서버(vision_pipeline/skill_server.py)가 `/place/status`
(std_msgs/String)로 발행하는 진행 로그를 구독해 타임스탬프와 함께 출력한다.

cd ~/motie_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=9 python3 place_logger.py

- 같은 ROS_DOMAIN_ID(=9)라야 토픽이 보인다.
- 의존성은 ROS2 Humble core(rclpy + std_msgs)뿐 — dual_arm_msgs/sequence_client 불필요.
- 로거를 늦게 켜면 그 이후 메시지만 보인다(진행 이벤트라 latch 안 함). 서버보다 먼저 켜 두면 처음부터 다 보인다.
"""
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TOPIC = "/place/status"


class PlaceLogger(Node):
    def __init__(self):
        super().__init__("place_logger")
        self.create_subscription(String, TOPIC, self._on_status, 10)
        self._count = 0
        dom = os.environ.get("ROS_DOMAIN_ID", "(unset)")
        print(f"[place_logger] subscribe {TOPIC}  (ROS_DOMAIN_ID={dom})", flush=True)
        print("[place_logger] 물체 내려놓기 진행 로그 대기 중 ... (Ctrl+C 종료)", flush=True)

    def _on_status(self, msg):
        self._count += 1
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {self._count:2d}. {msg.data}", flush=True)


def main():
    rclpy.init()
    node = PlaceLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[place_logger] 종료", flush=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
