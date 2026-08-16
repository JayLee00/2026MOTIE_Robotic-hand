#!/usr/bin/env python3
"""
ROSCameraSource — ROS 카메라 토픽에서 프레임을 받아 RealSenseSession 과
동일한 (__enter__ / capture() / __exit__) 인터페이스로 NPZ 를 저장한다.

run_pipeline_interactive.py 에서 `--camera_source ros` 로 선택해서 사용.
저장 포맷(rgb=BGR uint8, depth=meters float32, K)은 RealSense 캡처와 동일하므로
나머지 파이프라인(SAM3 → PCA → overlay)은 수정 없이 그대로 동작한다.

전제: conda 환경에서 rclpy import 가능 (검증됨). 실행 전 ROS 소싱 필요:
    export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
    source /opt/ros/<distro>/setup.bash

주의: aligned_depth 가 매우 느릴 수 있어(0.2~0.8Hz), spin_once 를 촘촘히(0.05s)
돌려야 느린 depth 프레임을 놓치지 않는다. (0.2s 로 성글게 돌면 depth 를 굶김)
"""
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo

from .realsense import make_capture_path, save_capture_bundle

try:
    import cv2
except Exception:
    cv2 = None

# 실측된 카메라 토픽 (2026-07) — /front_cam
DEFAULT_COLOR = "/front_cam/front/color/image_raw"
DEFAULT_DEPTH = "/front_cam/front/aligned_depth_to_color/image_raw"
DEFAULT_INFO  = "/front_cam/front/color/camera_info"

_SPIN_DT = 0.05   # spin_once 간격 (느린 depth 를 놓치지 않으려면 촘촘히)


class _CamNode(Node):
    def __init__(self, color_topic, depth_topic, info_topic):
        super().__init__("ros_camera_source")
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)  # 이미지=sensor data
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


class ROSCameraSource:
    """RealSenseSession 과 호환되는 ROS 카메라 소스."""

    def __init__(self, color_topic=DEFAULT_COLOR, depth_topic=DEFAULT_DEPTH,
                 info_topic=DEFAULT_INFO, depth_scale=0.001, timeout=30.0,
                 video_path=None, **_ignored):
        self._color_topic = color_topic
        self._depth_topic = depth_topic
        self._info_topic  = info_topic
        self._depth_scale = depth_scale     # mm → m
        self._timeout     = timeout
        self._video_path  = video_path
        self._node   = None
        self._writer = None
        self._owns_rclpy = False

    def __enter__(self):
        # rclpy가 아직 init 안 됐을 때만 우리가 init (SequenceClient 등과 공존)
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._node = _CamNode(self._color_topic, self._depth_topic, self._info_topic)
        print(f"[ROSCamera] 구독 시작: {self._color_topic}")
        t0 = time.time()
        while rclpy.ok() and not self._node.ready() and time.time() - t0 < self._timeout:
            rclpy.spin_once(self._node, timeout_sec=_SPIN_DT)
        if not self._node.ready():
            raise RuntimeError(
                f"[ROSCamera] {self._timeout:.0f}s 안에 프레임 수신 실패 "
                f"(color={self._node.color is not None}, depth={self._node.depth is not None}, "
                f"K={self._node.K is not None}) — 토픽/QoS/네트워크/발행률 확인")
        print("[ROSCamera] 준비 완료")
        return self

    def _grab(self):
        """최신 프레임 1장 획득 (color·depth 를 새로 대기)."""
        self._node.color = None
        self._node.depth = None
        t0 = time.time()
        while rclpy.ok() and not self._node.ready() and time.time() - t0 < self._timeout:
            rclpy.spin_once(self._node, timeout_sec=_SPIN_DT)
        if not self._node.ready():
            raise RuntimeError("[ROSCamera] 프레임 수신 실패 (color/depth)")
        return self._node.color, self._node.depth, self._node.K

    def capture(self, output_dir, stem, capture_idx=0, suffix=".npz",
                save_rgb_png=True, save_depth_png=True):
        rgb_bgr, depth_raw, K = self._grab()
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        if self._video_path and cv2 is not None:
            if self._writer is None:
                h, w = rgb_bgr.shape[:2]
                self._writer = cv2.VideoWriter(
                    str(self._video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            self._writer.write(rgb_bgr)
        output_path = make_capture_path(output_dir, stem, capture_idx, suffix=suffix)
        save_capture_bundle(output_path, rgb_bgr, depth_raw, depth_m, K,
                            save_rgb_png=save_rgb_png, save_depth_png=save_depth_png)
        print(f"[ROSCamera] 캡처 저장: {output_path}")
        return output_path

    def __exit__(self, *_):
        if self._writer is not None:
            self._writer.release()
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_rclpy:   # 우리가 init 했을 때만 shutdown
            try:
                rclpy.shutdown()
            except Exception:
                pass
