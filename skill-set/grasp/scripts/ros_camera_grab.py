#!/usr/bin/env python3
"""
ROS 카메라 프레임 1장 → NPZ 저장 (rclpy 그래버, 파이프라인과 분리 실행).

grasp_fruit(conda, Python 3.12)에서는 Humble rclpy(3.10)를 import 할 수 없으므로,
이 스크립트는 **시스템 Python 3.10**(= Humble native)으로 실행하여 카메라 토픽을
구독하고 NPZ 만 저장한다. 이후 SAM3/PCA 는 grasp_fruit 이 오프라인으로 처리한다:

    # 1) 카메라 → NPZ  (Humble python, ROS 소싱 필요)
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
    /usr/bin/python3 scripts/ros_camera_grab.py --stem scene

    # 2) NPZ → SAM3 → PCA  (grasp_fruit, rclpy 불필요)
    /home/user/miniconda3/envs/grasp_fruit/bin/python scripts/run_pipeline.py \
        --input data/raw/scene_000.npz --query orange \
        --calibration configs/calibration/extrinsic_20260612_170053.json

저장 포맷은 src/affordance_grasp/io/dataset_io.py save_rgbd_bundle 과 동일:
    NPZ keys: rgb (BGR uint8), depth (meters float32), K (float32 3x3)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo

DEFAULT_COLOR = "/front_cam/front/color/image_raw"
DEFAULT_DEPTH = "/front_cam/front/aligned_depth_to_color/image_raw"
DEFAULT_INFO  = "/front_cam/front/color/camera_info"

_SPIN_DT = 0.05   # 느린 aligned_depth 를 놓치지 않으려면 촘촘히 spin


class _CamNode(Node):
    def __init__(self, color_topic, depth_topic, info_topic):
        super().__init__("ros_camera_grab")
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.color = None
        self.depth = None
        self.K = None
        self.create_subscription(Image, color_topic, self._color_cb, qos)
        self.create_subscription(Image, depth_topic, self._depth_cb, qos)
        self.create_subscription(CameraInfo, info_topic, self._info_cb, qos)

    def _color_cb(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
        a = np.ascontiguousarray(a[:, : m.width * 3]).reshape(m.height, m.width, 3)
        if m.encoding == "rgb8":          # RGB → BGR (파이프라인은 BGR 저장)
            a = a[..., ::-1]
        self.color = np.ascontiguousarray(a)

    def _depth_cb(self, m):               # 16UC1 (mm)
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
        a = np.ascontiguousarray(a[:, : m.width * 2])
        self.depth = a.view(np.uint16).reshape(m.height, m.width)

    def _info_cb(self, m):
        self.K = np.array(m.k, dtype=np.float64).reshape(3, 3)

    def ready(self):
        return self.color is not None and self.depth is not None and self.K is not None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    p.add_argument("--stem",        default="scene")
    p.add_argument("--index",       type=int, default=0)
    p.add_argument("--color_topic", default=DEFAULT_COLOR)
    p.add_argument("--depth_topic", default=DEFAULT_DEPTH)
    p.add_argument("--info_topic",  default=DEFAULT_INFO)
    p.add_argument("--depth_scale", type=float, default=0.001, help="mm → m")
    p.add_argument("--timeout",     type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = _CamNode(args.color_topic, args.depth_topic, args.info_topic)
    print(f"[grab] 구독 시작: {args.color_topic}")
    t0 = time.time()
    while rclpy.ok() and not node.ready() and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=_SPIN_DT)

    if not node.ready():
        print(f"[grab] {args.timeout:.0f}s 안에 프레임 수신 실패 "
              f"(color={node.color is not None}, depth={node.depth is not None}, "
              f"K={node.K is not None}) — 토픽/QoS/ROS_DOMAIN_ID 확인", file=sys.stderr)
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)

    rgb_bgr = node.color
    depth_m = node.depth.astype(np.float32) * args.depth_scale
    K = node.K.astype(np.float32)

    out_path = out_dir / f"{args.stem}_{args.index:03d}.npz"
    np.savez_compressed(out_path, rgb=rgb_bgr, depth=depth_m, K=K)
    print(f"[grab] 저장 완료: {out_path}")
    print(f"[grab]   rgb {rgb_bgr.shape} {rgb_bgr.dtype} | "
          f"depth {depth_m.shape} {depth_m.dtype} (m) | K\n{K}")

    # 눈으로 확인용 PNG (cv2 있으면)
    try:
        import cv2
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_rgb.png")), rgb_bgr)
        print(f"[grab]   미리보기: {out_path.with_name(out_path.stem + '_rgb.png')}")
    except Exception:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
