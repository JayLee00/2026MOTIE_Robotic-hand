"""Host RGB-D capture service (run on Current PC host, env `anyplace_cu128`).

The front_cam RealSense is a HOST USB device; the dex_ros container has no USB
access. So the camera lives here as an HTTP service and the orchestrator (in the
container, --network host) grabs frames over 127.0.0.1. Depth is aligned to color.

    POST /predict  npz{}  ->  npz{ rgb:(H,W,3)u8, depth:(H,W)f32 meters, K:(3,3) }

Run:  conda activate anyplace_cu128 && python vision_pipeline/services/capture_service.py [--port 8814]
"""
import argparse
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8814)
    a = ap.parse_args()

    import pyrealsense2 as rs
    from vision_pipeline.services.rpc import serve

    # hardware reset (the D435i needs it after a cold start / prior grab)
    for d in rs.context().query_devices():
        d.hardware_reset()
    time.sleep(5)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    lock = threading.Lock()
    for _ in range(40):                              # let auto-exposure settle
        pipe.wait_for_frames(15000)
    print("[capture] ready (front_cam 640x480, depth aligned to color)", flush=True)

    def predict(_inp):
        with lock:
            fr = align.process(pipe.wait_for_frames(15000))
            c = np.asanyarray(fr.get_color_frame().get_data())            # HxWx3 rgb8
            d = np.asanyarray(fr.get_depth_frame().get_data()).astype(np.float32) * scale
            ci = fr.get_color_frame().profile.as_video_stream_profile().intrinsics
            K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]])
        return {"rgb": c.astype(np.uint8), "depth": d.astype(np.float32), "K": K.astype(np.float64)}

    serve(predict, a.port, name="capture")


if __name__ == "__main__":
    main()
