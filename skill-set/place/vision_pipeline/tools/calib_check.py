"""Test A — calibration check (host, env `anyplace_cu128`). Needs capture_service :8814.

Grabs one front_cam frame from the host capture service, back-projects to world via
the offline extrinsic (world_from_cam = tf.txt . URDF base->link0 — the SAME transform
the orchestrator uses, since the camera is fixed to base), and reports whether the
world coords are sane: camera height, scene bbox, and a dominant-plane fit (the table
should be ~horizontal in world -> plane normal ~ +z, small residual).

    conda activate anyplace_cu128 && python vision_pipeline/tools/calib_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")
from vision_pipeline.services.rpc import post_npz
from vision_pipeline.core.pointcloud import backproject
from vision_pipeline.core.extrinsic import world_from_cam

OUT = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place/vision_pipeline/test_logs"


def main():
    os.makedirs(OUT, exist_ok=True)
    out = post_npz("http://127.0.0.1:8814/predict", timeout=30)
    rgb, depth, K = out["rgb"], out["depth"], out["K"]
    T = world_from_cam()
    pc = backproject(depth, K, T)

    c = pc.mean(0)
    _, _, vt = np.linalg.svd(pc - c, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    if n[2] < 0:
        n = -n
    resid = float(np.sqrt(np.mean(((pc - c) @ n) ** 2)))

    np.savez(os.path.join(OUT, "A_scene.npz"), rgb=rgb, depth=depth, K=K, world=pc, T_world_cam=T)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(os.path.join(OUT, "A_scene.png"))
    except Exception:
        pass

    print("=== Test A calibration check ===")
    print("camera world position:", np.round(T[:3, 3], 3).tolist(), " (expect z ~0.7-0.8m)")
    print("scene world bbox  min:", np.round(pc.min(0), 3).tolist(), " max:", np.round(pc.max(0), 3).tolist())
    print("dominant-plane normal (world):", np.round(n, 3).tolist(),
          " cos(normal,+z) = %.3f" % abs(n[2]), " (1.0 = table horizontal -> extrinsic OK)")
    print("plane RMS residual: %.4f m" % resid)
    print("saved:", os.path.join(OUT, "A_scene.npz"), "(+ A_scene.png)")


if __name__ == "__main__":
    main()
