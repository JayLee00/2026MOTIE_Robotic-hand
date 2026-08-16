"""Grab ONE live front_cam frame (color + aligned depth + K) from the ROS topics and save
it to test_logs/live_frame.npz — for offline inspection / segmentation checks. Run INSIDE
the dex_ros container (has rclpy), domain 9:

    python3 vision_pipeline/tools/grab_frame.py
"""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # host or /work
sys.path.insert(0, REPO)
from vision_pipeline.backends.ros_backend import (  # noqa: E402
    _cimg_to_rgb, _cimg_to_depth_m, TOPIC_RGB, TOPIC_DEPTH, TOPIC_CAMINFO)


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, CameraInfo

    rclpy.init()
    node = Node("grab_frame")
    got = {}
    node.create_subscription(CompressedImage, TOPIC_RGB, lambda m: got.__setitem__("rgb", _cimg_to_rgb(m)),
                             qos_profile_sensor_data)
    node.create_subscription(CompressedImage, TOPIC_DEPTH, lambda m: got.__setitem__("depth", _cimg_to_depth_m(m)),
                             qos_profile_sensor_data)
    node.create_subscription(CameraInfo, TOPIC_CAMINFO,
                             lambda m: got.__setitem__("K", np.asarray(m.k, float).reshape(3, 3)),
                             qos_profile_sensor_data)
    t0 = time.time()
    while not all(k in got for k in ("rgb", "depth", "K")) and time.time() - t0 < 15:
        rclpy.spin_once(node, timeout_sec=0.2)

    if "rgb" not in got:
        print("NO FRAME — camera topic not reachable (domain 9 / align_depth?)")
        return
    out = f"{REPO}/vision_pipeline/test_logs/live_frame.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, rgb=got["rgb"], depth=got.get("depth", np.zeros(0)), K=got.get("K", np.eye(3)))
    d = got.get("depth")
    print(f"saved {out}: rgb {got['rgb'].shape}, depth {None if d is None else d.shape} "
          f"(valid {0 if d is None else int((d > 0).sum())})")


if __name__ == "__main__":
    main()
