#!/usr/bin/env python3
"""과일 종류 라벨 노드 — /fruit/type 발행 + FoundationPose CAD 교체.

과일은 종류당 대표 CAD 1개를 쓰고, 종류는 정수 라벨로 구분한다(레몬=1, 자두=2 …).
카탈로그는 fruits.yaml 이 단일 출처다.

발행 (기존 /fruit/* 네임스페이스에 맞춤):
    /fruit/type       std_msgs/Int32     과일 종류 id (0 = 미지정)
    /fruit/type_name  std_msgs/String    사람이 읽을 이름 (디버깅·오버레이용)
    /fruit/reset      std_msgs/String    선택된 CAD 경로 → fp_ros_node 가 메시 교체

구독:
    /fruit/set_type   std_msgs/String    과일 이름("lemon") 또는 id("1") 로 전환

레코더에는 33_fruit_type 으로 넣으면 30~32(pos/quat/size) 와 나란히 놓인다.

실행:
    python3 fruit_label_node.py --fruit lemon
    ros2 topic pub --once /fruit/set_type std_msgs/String '{data: "plum"}'
"""
from __future__ import annotations

import argparse
import os
import sys

import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

HERE = os.path.dirname(os.path.abspath(__file__))

# 라벨은 늦게 뜬 구독자도 반드시 받아야 한다(HDF5 에 들어가는 값이므로)
# → TRANSIENT_LOCAL 로 마지막 값을 붙잡아 둔다.
LATCH = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                   durability=DurabilityPolicy.TRANSIENT_LOCAL,
                   history=HistoryPolicy.KEEP_LAST, depth=1)


def load_catalog(path: str) -> list[dict]:
    with open(path) as f:
        cat = yaml.safe_load(f).get("fruits", [])
    ids = [c["id"] for c in cat]
    if len(ids) != len(set(ids)):
        raise ValueError(f"fruits.yaml 에 중복 id 가 있습니다: {ids}")
    if 0 in ids:
        raise ValueError("id 0 은 '미지정' 예약값입니다")
    return cat


class FruitLabelNode(Node):
    def __init__(self, catalog: list[dict], start: str | None, rate: float):
        super().__init__("fruit_label")
        self.cat = catalog
        self.cur = None

        self.pub_type = self.create_publisher(Int32, "/fruit/type", LATCH)
        self.pub_name = self.create_publisher(String, "/fruit/type_name", LATCH)
        # fp_ros_node 가 듣는 재등록 토픽 — CAD 를 같이 갈아준다
        self.pub_reset = self.create_publisher(String, "/fruit/reset", 10)
        self.create_subscription(String, "/fruit/set_type", self._on_set, 10)

        names = ", ".join(f"{c['id']}={c['name']}" for c in self.cat)
        self.get_logger().info(f"카탈로그: {names}")
        self.get_logger().info("발행: /fruit/type, /fruit/type_name  구독: /fruit/set_type")

        if start:
            self._select(start, swap_mesh=True)
        else:
            self._publish(0, "none")
            self.get_logger().warn("과일 미지정 (0) — /fruit/set_type 으로 골라주세요")

        if rate > 0:
            self.create_timer(1.0 / rate, self._republish)

    def _find(self, key: str) -> dict | None:
        key = key.strip()
        if key.isdigit():
            return next((c for c in self.cat if c["id"] == int(key)), None)
        return next((c for c in self.cat if c["name"].lower() == key.lower()), None)

    def _select(self, key: str, swap_mesh: bool) -> bool:
        c = self._find(key)
        if c is None:
            self.get_logger().error(
                f"모르는 과일: '{key}' — 가능: "
                f"{[x['name'] for x in self.cat]}")
            return False
        self.cur = c
        self._publish(c["id"], c["name"])
        self.get_logger().info(f"과일 = {c['name']}({c.get('name_ko','')}) id={c['id']}")

        if swap_mesh:
            mesh = c["mesh"]
            if not os.path.isabs(mesh):
                mesh = os.path.join(HERE, mesh)
            if os.path.isfile(mesh):
                self.pub_reset.publish(String(data=mesh))
                self.get_logger().info(f"CAD 교체 요청: {mesh}")
            else:
                # CAD 가 없어도 라벨은 계속 나가야 한다 — 자세만 못 잡을 뿐이다
                self.get_logger().warn(
                    f"CAD 파일이 없습니다: {mesh} — 라벨만 발행하고 자세는 이전 메시 유지. "
                    f"prepare_mesh.py 로 만들어 assets/ 에 두세요")
        return True

    def _publish(self, fid: int, name: str):
        self.pub_type.publish(Int32(data=int(fid)))
        self.pub_name.publish(String(data=name))

    def _republish(self):
        """주기 재발행 — 늦게 붙은 구독자·재시작한 레코더가 놓치지 않게."""
        if self.cur is None:
            self._publish(0, "none")
        else:
            self._publish(self.cur["id"], self.cur["name"])

    def _on_set(self, msg: String):
        self._select(msg.data, swap_mesh=True)


def main():
    ap = argparse.ArgumentParser(description="과일 종류 라벨 발행 + CAD 교체")
    ap.add_argument("--fruit", default=None,
                    help="시작 과일 (이름 또는 id). 없으면 0(미지정)")
    ap.add_argument("--catalog", default=os.path.join(HERE, "fruits.yaml"))
    ap.add_argument("--rate", type=float, default=2.0,
                    help="라벨 재발행 주기 [Hz], 0=한 번만")
    ap.add_argument("--list", action="store_true", help="카탈로그만 출력하고 종료")
    a = ap.parse_args()

    cat = load_catalog(a.catalog)
    if a.list:
        print(f"{'id':>3}  {'name':<10} {'한글':<8} {'nominal[m]':<22} mesh")
        for c in cat:
            mesh = c["mesh"] if os.path.isabs(c["mesh"]) else os.path.join(HERE, c["mesh"])
            mark = "✓" if os.path.isfile(mesh) else "✗ 없음"
            print(f"{c['id']:>3}  {c['name']:<10} {c.get('name_ko',''):<8} "
                  f"{str(c.get('nominal','')):<22} {mark} {c['mesh']}")
        return

    rclpy.init()
    node = FruitLabelNode(cat, a.fruit, a.rate)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
