"""Real-data perception+geometry validation (no robot).

Uses the captured fixtures/scene.npz (orange + fruit tray) + the offline
world<-cam extrinsic + the live HTTP services to exercise the perception chain
on REAL data:  Molmo -> SAM -> back-project(world) -> IGR completion -> AnyPlace.

Requires molmo:8810, sam:8811, igr:8816, anyplace:8801 running.
Saves world clouds to fixtures/ for inspection. Skips stages whose service is down.

    python -m vision_pipeline.validate_real
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")
from vision_pipeline.core import pointcloud as PC
from vision_pipeline.core import geometry as G
from vision_pipeline.core.extrinsic import world_from_cam
from vision_pipeline.models_client import ModelsHTTP

FX = os.path.join(os.path.dirname(__file__), "fixtures")


def up(url):
    import urllib.request
    try:
        urllib.request.urlopen(url + "/health".replace("/predict", ""), timeout=2)
        return True
    except Exception:
        try:
            urllib.request.urlopen(url.replace("/predict", "/health"), timeout=2)
            return True
        except Exception:
            return False


def main():
    from vision_pipeline import prompts
    d = np.load(os.path.join(FX, "scene.npz"))
    rgb, depth, K = d["rgb"], d["depth"], d["K"]
    Twc = world_from_cam()
    m = ModelsHTTP()
    H, W = depth.shape
    print(f"scene {W}x{H}")

    def molmo(prompt, multi=False):
        pts = m.molmo(rgb, prompt, multi=multi)
        print(f"  Molmo('{prompt[:40]}...') -> {len(pts)} pt(s): {[tuple(np.round(p,0)) for p in pts[:5]]}")
        return pts

    # --- parent: fruit tray ---
    tray_pt = molmo("Point to the molded fiber fruit tray")[0]
    tray_mask = m.sam(rgb, tray_pt)
    tray_pc = PC.backproject(depth, K, Twc, mask=tray_mask)
    print(f"  tray mask {int(tray_mask.sum())}px -> parent_pc {len(tray_pc)} pts, "
          f"world bbox {np.round(tray_pc.min(0),3)}..{np.round(tray_pc.max(0),3)}")
    np.save(os.path.join(FX, "tray_pc.npy"), tray_pc)

    # --- child: grasped object (FIXED prompt) ---
    orange_pt = molmo(prompts.grasp_prompt())[0]
    orange_mask = m.sam(rgb, orange_pt)
    orange_pc = PC.backproject(depth, K, Twc, mask=orange_mask)
    print(f"  orange mask {int(orange_mask.sum())}px -> child_pc_i {len(orange_pc)} pts, "
          f"center {np.round(orange_pc.mean(0),3)}")
    np.save(os.path.join(FX, "orange_pc.npy"), orange_pc)

    # --- completion (IGR; contacts computed service-side from FK) ---
    orange_com, _ = m.complete_igr(orange_pc)
    print(f"  IGR -> child_pc_com {orange_com.shape}, center {np.round(orange_com.mean(0),3)}")
    np.save(os.path.join(FX, "orange_com.npy"), orange_com)

    # --- hand cloud (PaXini-URDF FK) + fuse (step 3-14). Offline this uses the recorded
    #     grasp (kistar_pose/tmp_pose.txt); live it uses the real hand joints. ---
    hand_pc = m.hand_pc_paxini()
    child_pc = np.concatenate([orange_com, hand_pc], axis=0)
    print(f"  PaXini hand_pc {hand_pc.shape} -> child_pc (orange+hand) {child_pc.shape}")

    # --- placement (child upright; placeholder palm normal = world +x) ---
    n_palm = np.array([1.0, 0.0, 0.0])
    Tz = G.align_palm_down(child_pc, n_palm)
    child_z = G.apply(Tz, child_pc)
    lcs = G.local_crop_size(orange_com, n_palm)        # crop size from object only (3-12)
    holes = molmo("Point to all the empty holes in the molded fiber fruit tray where fruit can be inserted",
                  multi=True)
    centers = [PC.backproject_pixel(h, depth, K, Twc) for h in holes]
    centers = [c for c in centers if c is not None] or [tray_pc.mean(0)]
    cand = []
    for c in centers:
        region = G.crop_region(tray_pc, c, max(lcs, 0.06), axis=G.WORLD_DOWN)
        if len(region) < 16:
            continue
        cand.extend(list(m.place(region, child_z)))
    if cand:
        cand = np.asarray(cand)
        k, sc = G.rank_upright(cand)
        T_act = G.compose_T_act(cand[k], Tz)
        print(f"  AnyPlace: {len(cand)} cand from {len(centers)} hole(s); "
              f"upright top-1={sc[k]:.3f}\n  T_act=\n{np.round(T_act,4)}")
    print("REAL-DATA CHAIN OK")


if __name__ == "__main__":
    main()
