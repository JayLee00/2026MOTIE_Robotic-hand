#!/usr/bin/env python3
"""3구 USB 풋스위치(PCsensor) → 글러브 핸드 텔레옵 제어권 + 데이터 로깅 토글.

매핑(사용자 선택: 독립형):
  오른쪽 = 텔레옵 ENGAGE   → /teleop/hand_engage/<side> = true  (글러브가 q_target 스트리밍)
  왼쪽   = 텔레옵 DISENGAGE → /teleop/hand_engage/<side> = false (홀드, 손 안 풀림)
  중간   = 로깅 S/E 토글    → /record/enable 토글 (true=에피소드 시작, false=저장)

입력: /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd 직접 read + EVIOCGRAB(독점).
권한: 최초 1회  sudo usermod -aG input $USER  후 재로그인 (또는 newgrp input).
     (glove_teleop.py 는 /teleop/hand_engage/<side> 를 구독해 게이팅, ros2_hdf5_recorder.py 는 /record/enable 을 구독.)

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 record/foot_pedal_glove.py
  python3 record/foot_pedal_glove.py --side left
"""
import argparse
import fcntl
import os
import select
import struct
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool

DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"

# evdev input_event: struct timeval(long,long) + type(H) + code(H) + value(i)
EV_KEY = 1
KEY_A, KEY_B, KEY_C = 30, 48, 46      # 왼쪽 / 중간 / 오른쪽 페달
EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)
EVIOCGRAB = 0x40044590


class PedalGlove(Node):
    def __init__(self, side: str) -> None:
        super().__init__("foot_pedal_glove")
        self.side = side
        rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        # engage 는 TRANSIENT_LOCAL: 글러브/레코더가 나중에 떠도 마지막 상태를 받음
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_engage = self.create_publisher(Bool, f"/teleop/hand_engage/{side}", latched)
        self.pub_record = self.create_publisher(Bool, "/record/enable", rel)
        self.recording = False
        # 시작 상태: disengage(안전) + 로깅 off
        self._engage(False)
        self.get_logger().info(
            f"발판 준비 (side={side}). 오른=ENGAGE 왼=DISENGAGE 중간=로깅토글. Ctrl-C 종료")

    def _engage(self, on: bool) -> None:
        self.pub_engage.publish(Bool(data=on))
        self.get_logger().info(f"텔레옵 {'ENGAGE(제어권 ON)' if on else 'DISENGAGE(홀드)'}")

    def on_right(self) -> None:
        self._engage(True)

    def on_left(self) -> None:
        self._engage(False)

    def on_mid(self) -> None:
        self.recording = not self.recording
        self.pub_record.publish(Bool(data=self.recording))
        self.get_logger().info(f"로깅 {'시작(S) — 에피소드 수집' if self.recording else '종료(E) — 저장'}")


def open_pedal():
    try:
        fd = os.open(DEVICE, os.O_RDONLY | os.O_NONBLOCK)
    except PermissionError:
        user = os.environ.get("USER", "$USER")
        print(f"[pedal] 권한 없음: {DEVICE}\n"
              f"  최초 1회:  sudo usermod -aG input {user}   → 재로그인(또는 newgrp input)")
        return None
    except FileNotFoundError:
        print(f"[pedal] 장치 없음: {DEVICE}\n  풋스위치 USB 연결/재연결 확인")
        return None
    try:
        fcntl.ioctl(fd, EVIOCGRAB, 1)
    except OSError as e:
        print(f"[pedal] EVIOCGRAB 실패(계속 진행): {e}")
    return fd


def main() -> None:
    ap = argparse.ArgumentParser(description="풋스위치 → 글러브 텔레옵 engage + 로깅 토글")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    args = ap.parse_args()

    rclpy.init()
    node = PedalGlove(args.side)
    fd = open_pedal()
    if fd is None:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print("[pedal] 오른=ENGAGE  왼=DISENGAGE  중간=로깅 S/E토글  |  Ctrl-C 종료")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                continue
            data = os.read(fd, EVENT_SIZE * 64)
            for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                _, _, etype, code, val = struct.unpack(EVENT_FMT, data[off:off + EVENT_SIZE])
                if etype == EV_KEY and val == 1:      # 눌림(press)만
                    if code == KEY_C:
                        node.on_right()
                    elif code == KEY_A:
                        node.on_left()
                    elif code == KEY_B:
                        node.on_mid()
    except KeyboardInterrupt:
        pass
    finally:
        node._engage(False)                            # 종료 시 안전하게 disengage
        if node.recording:
            node.pub_record.publish(Bool(data=False))  # 열려있던 에피소드 저장
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.01)
        try:
            fcntl.ioctl(fd, EVIOCGRAB, 0)
        except OSError:
            pass
        os.close(fd)
        node.destroy_node()
        rclpy.shutdown()
        print("\n[pedal] 종료 (DISENGAGE)")


if __name__ == "__main__":
    main()
