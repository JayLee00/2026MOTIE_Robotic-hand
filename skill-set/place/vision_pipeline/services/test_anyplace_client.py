"""Client test for anyplace_service: waits for readiness, posts dummy clouds,
runs the §9-A ranking + T_act composition end-to-end."""
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")
from vision_pipeline.services.rpc import post_npz
from vision_pipeline.core import geometry as G

URL = "http://127.0.0.1:8801"


def wait_ready(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(URL + "/health", timeout=2).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def main():
    assert wait_ready(), "service not ready"
    rng = np.random.default_rng(1)
    # child already gravity-aligned (child_pc_zalign): flat-ish object near scene
    child = rng.normal([0.35, 0.0, 0.20], [0.03, 0.03, 0.02], size=(2000, 3)).astype(np.float32)
    parent = rng.normal([0.35, 0.0, 0.0], [0.10, 0.10, 0.01], size=(4000, 3)).astype(np.float32)

    t0 = time.time()
    out = post_npz(URL + "/predict", parent_pcd=parent, child_pcd=child)
    dt = time.time() - t0
    T_pred = out["out_tf"]
    print(f"out_tf shape {T_pred.shape}  ({dt:.1f}s)")
    assert T_pred.ndim == 3 and T_pred.shape[1:] == (4, 4)

    # §9-A: pick the most-upright candidate, then compose T_act
    k, scores = G.rank_upright(T_pred)
    print(f"rank_upright -> k={k}  score(R[2,2])={scores[k]:.4f}  (range {scores.min():.3f}..{scores.max():.3f})")
    n_palm = np.array([1.0, 0.0, 0.0])           # live right_palm +x (placeholder)
    T_zalign = G.align_palm_down(child, n_palm)
    T_act = G.compose_T_act(T_pred[k], T_zalign)
    ee_cur = G.homog(R=np.eye(3), t=[0.34, 0.04, 0.46])  # current EE (placeholder)
    ee_goal = G.ee_target(T_act, ee_cur)
    assert np.allclose(ee_goal[3], [0, 0, 0, 1]), "EE goal not a valid homog"
    print("T_act=\n", np.round(T_act, 4))
    print("EE_target translation:", np.round(ee_goal[:3, 3], 4))
    print("PIPELINE STEP-1 OK: AnyPlace service -> rank_upright -> T_act -> EE_target")


if __name__ == "__main__":
    main()
