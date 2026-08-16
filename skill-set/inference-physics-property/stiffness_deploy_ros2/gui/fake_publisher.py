#!/usr/bin/env python3
"""fake_publisher.py — 로봇 없이 GUI(stiffness_gui.py) 를 미리 보는 개발용 가짜 퍼블리셔.

실제 deploy_task3_ros2 대신, /stiffness/result 에 measuring→done 을 과일별로 번갈아
발행한다. 사진 배치/막대 색을 로봇 실행 전에 확인할 때 사용.

사용 (터미널 2개):
  # 터미널 A — GUI
  source env.sh && python3 stiffness_deploy_ros2/gui/stiffness_gui.py
  # 터미널 B — 가짜 데이터
  source env.sh && python3 stiffness_deploy_ros2/gui/fake_publisher.py
(같은 ROS_DOMAIN_ID 여야 함. env.sh 로 두 창 모두 세팅하면 동일.)
"""
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

# 실제 퍼블리셔(토픽/QoS/JSON 스키마)를 그대로 재사용 → GUI 와 100% 동일 경로.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "launch"))
from stiffness_result_pub import StiffnessResultPublisher  # noqa: E402

CLASS_NAMES = ["soft", "mid", "hard"]
# (norm_min, norm_max, [경계1, 경계2], 데모로 보여줄 강성값들)  — 실제 yaml 값과 동일.
FRUITS = {
    "plum":   (0.0, 7.65,  [1.761, 3.84],  [1.0, 2.8, 5.0]),
    "kiwi":   (0.0, 7.63,  [2.07, 3.672],  [1.2, 3.0, 5.5]),
    "tomato": (0.0, 6.06,  [1.905, 3.075], [1.0, 2.5, 4.5]),
    "lemon":  (0.0, 10.14, [3.416, 4.894], [2.5, 4.0, 7.0]),
}


def cls_of(stiffness, bounds):
    return sum(1 for b in bounds if stiffness >= b)


def main():
    rclpy.init()
    pub = StiffnessResultPublisher()
    ex = SingleThreadedExecutor()
    ex.add_node(pub)
    threading.Thread(target=ex.spin, daemon=True).start()

    print("[fake] /stiffness/result 로 데모 발행 중 (Ctrl+C 종료)…")
    try:
        while True:
            for fruit, (lo, hi, bounds, samples) in FRUITS.items():
                for s in samples:
                    pub.set_measuring(fruit, lo, hi, bounds, CLASS_NAMES)
                    print(f"[fake] {fruit} 측정 중…")
                    time.sleep(2.0)
                    cls = cls_of(s, bounds)
                    pub.set_result(fruit, s, cls, CLASS_NAMES[cls], lo, hi, bounds, CLASS_NAMES)
                    print(f"[fake] {fruit} 결과 강성={s} 등급={CLASS_NAMES[cls]}")
                    time.sleep(3.0)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
