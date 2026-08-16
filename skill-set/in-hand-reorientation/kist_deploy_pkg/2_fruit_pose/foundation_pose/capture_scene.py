#!/usr/bin/env python3
"""카메라 RGB-D 를 FoundationPose 레포의 demo_data 형식으로 녹화한다.

왜 필요한가: 라이브로 비교하면 매번 손 움직임이 달라 공정한 비교가 안 된다.
시퀀스를 한 번 떠두면 **같은 입력**으로 설정을 바꿔가며 돌릴 수 있고,
오프라인이라 주기(rate) 변수도 사라져 자세추정 자체의 상한을 볼 수 있다.

만들어지는 것 (YcbineoatReader 가 그대로 읽는 형식):
    <out>/cam_K.txt        3x3 내부 파라미터
    <out>/rgb/000000.png   컬러
    <out>/depth/000000.png 깊이 uint16 [mm]
    <out>/masks/000000.png 첫 프레임 마스크 (레포는 프레임 0 것만 쓴다)

사용:
    python3 capture_scene.py --out /tmp/lemon_scene --sec 15
      → 창에서 물체를 한 번 클릭 → 그 시점부터 녹화 시작

그다음 레포 코드를 **그대로** 돌린다:
    python3 run_demo.py --mesh_file .../lemon.obj --test_scene_dir /tmp/lemon_scene
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

import message_filters

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1)
FM_ROOT = "/home/js/Desktop/vive_franka_teleop/fruit-manipulation"
SAM2_CKPT = os.path.join(FM_ROOT, "sam2.1_hiera_tiny.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
WIN = "capture — 물체를 클릭하면 녹화 시작 (q=중단)"


def main():
    ap = argparse.ArgumentParser(description="demo_data 형식으로 RGB-D 녹화")
    ap.add_argument("--out", default="/tmp/lemon_scene")
    ap.add_argument("--sec", type=float, default=15.0)
    ap.add_argument("--color-topic", default="/front_cam/front/color/image_fast")
    ap.add_argument("--depth-topic", default="/front_cam/front/aligned_depth_to_color/image_raw")
    ap.add_argument("--info-topic", default="/front_cam/front/color/camera_info")
    a = ap.parse_args()

    for sub in ("rgb", "depth", "masks"):
        d = os.path.join(a.out, sub)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    sys.path.insert(0, FM_ROOT)
    sys.path.insert(0, os.path.join(FM_ROOT, "third_party", "sam2"))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM2 로드 ({dev}) …", flush=True)
    sam = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device=dev))
    print("준비 완료 — 창에서 물체를 클릭하세요", flush=True)

    rclpy.init()
    node = rclpy.create_node("capture_scene")
    bridge = CvBridge()
    st = {"K": None, "click": None, "started": False, "n": 0, "t0": None,
          "quit": False, "rgb": None}

    def on_info(m):
        if st["K"] is None:
            st["K"] = np.array(m.k, dtype=np.float64).reshape(3, 3)
    node.create_subscription(CameraInfo, a.info_topic, on_info, QOS)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and not st["started"]:
            st["click"] = (float(x), float(y))
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    def on_rgbd(cm, dm):
        if st["K"] is None or st["quit"]:
            return
        rgb = bridge.imgmsg_to_cv2(cm, "rgb8")
        d = bridge.imgmsg_to_cv2(dm, "passthrough")
        st["rgb"] = rgb
        if d.dtype != np.uint16:                       # 레포는 uint16 mm 를 읽는다
            d = (d * 1000.0).astype(np.uint16)

        if st["click"] is not None and not st["started"]:
            sam.set_image(rgb)
            masks, scores, _ = sam.predict(
                point_coords=np.array([st["click"]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32), multimask_output=True)
            m = masks[int(np.argmax(scores))].astype(np.uint8) * 255
            cv2.imwrite(os.path.join(a.out, "masks", "000000.png"), m)
            np.savetxt(os.path.join(a.out, "cam_K.txt"), st["K"], fmt="%.8f")
            st["started"] = True
            st["t0"] = time.time()
            print(f"마스크 저장 ({int((m > 0).sum())}px) — 녹화 시작 {a.sec:.0f}초", flush=True)

        if st["started"]:
            i = st["n"]
            cv2.imwrite(os.path.join(a.out, "rgb", f"{i:06d}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(a.out, "depth", f"{i:06d}.png"), d)
            st["n"] = i + 1
            if time.time() - st["t0"] >= a.sec:
                st["quit"] = True

    sc = message_filters.Subscriber(node, Image, a.color_topic, qos_profile=QOS)
    sd = message_filters.Subscriber(node, Image, a.depth_topic, qos_profile=QOS)
    message_filters.ApproximateTimeSynchronizer([sc, sd], 5, 0.05).registerCallback(on_rgbd)

    while rclpy.ok() and not st["quit"]:
        rclpy.spin_once(node, timeout_sec=0.05)
        if st["rgb"] is not None:
            img = cv2.cvtColor(st["rgb"], cv2.COLOR_RGB2BGR).copy()
            msg = (f"녹화중 {st['n']}프레임 "
                   f"{time.time()-st['t0']:.1f}/{a.sec:.0f}s" if st["started"]
                   else "물체를 클릭하세요")
            cv2.putText(img, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if st["started"] else (0, 200, 255), 2)
            cv2.imshow(WIN, img)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

    n = st["n"]
    print(f"\n저장: {a.out}  ({n} 프레임, {n/max(a.sec,1e-9):.1f} fps)")
    if n:
        print("레포 코드를 그대로 돌리려면:")
        print(f"  run_demo.py --mesh_file <lemon.obj> --test_scene_dir {a.out}")


if __name__ == "__main__":
    main()
