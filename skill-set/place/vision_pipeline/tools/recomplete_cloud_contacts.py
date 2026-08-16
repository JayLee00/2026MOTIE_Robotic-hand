"""Re-run IGR completion on a saved run using the PaXini fingerprint PAD SURFACES (point
clouds) as contacts instead of 4 fingerprint centres, and write a new run dir whose
child_pc_zalign comes from that completion — so crop_thickness_sweep can sweep it through
AnyPlace.

The 4 fingerprint clouds are anchored to the run's SAVED contact centres (the run used live
hand joints that weren't logged, so FK from the recorded grasp is off by a few cm on one
finger; anchoring keeps each cloud at the true contact position and uses FK only for the
local pad shape).

  conda activate anyplace_cu128
  python vision_pipeline/tools/recomplete_cloud_contacts.py [run_dir] [--per_finger 256]
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from vision_pipeline.core import geometry as G           # noqa: E402
from vision_pipeline.core import igr_complete as IGR     # noqa: E402
from vision_pipeline.tools import hand_fk                # noqa: E402


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--per_finger", type=int, default=256)
    a = ap.parse_args()
    run = a.run or sorted(glob.glob(os.path.join(REPO, "vision_pipeline/test_logs/run_*")))[-1]
    d = np.load(os.path.join(run, "debug.npz"), allow_pickle=True)
    partial = np.asarray(d["child_pc_refined"], np.float64)      # DBSCAN-refined observed object
    saved_c = np.asarray(d["contact_points"], float)             # 4 live-FK fingerprint centres
    hand_pc = np.asarray(d["hand_pc"], float)                    # saved PaXini FK hand cloud
    n_palm = np.asarray(d["n_palm"], float)
    com_pts_saved = np.asarray(d["child_pc_com"], float)         # the run's 4-point completion

    # fingerprint pad SURFACE clouds (FK from the recorded grasp), then anchor each finger's
    # cloud onto that finger's saved (live-FK) contact centre.
    jv = hand_fk.joint_values("child_pose")
    _, per = hand_fk.fingertip_paxini_contact_cloud(jv, per_finger=a.per_finger)
    clouds = [(cloud - centre) + saved_c[i] for i, (f, cloud, centre) in enumerate(per)]
    contacts_cloud = np.concatenate(clouds, 0)
    print(f"contacts: 4 centres -> {len(contacts_cloud)} fingerprint-surface pts "
          f"({a.per_finger}/finger, {len(per)} fingers)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = IGR.load_network(device)
    print("IGR net loaded; completing (cloud contacts, verbatim)...")
    com_cloud = IGR.complete(net, partial, contacts_cloud, device, seed=0,
                             denoise_partial=False, expand_contact_patches=False)
    if com_cloud is None:
        sys.exit("IGR returned no surface")

    def extent(p):
        return (p.max(0) - p.min(0)) * 100
    print(f"child_pc_com  4-point(run): {len(com_pts_saved)} pts extent {np.round(extent(com_pts_saved),1)} cm")
    print(f"child_pc_com  cloud(new)  : {len(com_cloud)} pts extent {np.round(extent(com_cloud),1)} cm")

    # fuse hand + gravity-align (same as orchestrator [3-14]..[3-16])
    child_pc = np.concatenate([com_cloud, hand_pc], 0)
    T_zalign = G.align_palm_down(child_pc, n_palm)
    child_zalign = G.apply(T_zalign, child_pc)

    out = run.rstrip("/") + "_cloudcontact"
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "debug.npz"),
             parent_pc_full=np.asarray(d["parent_pc_full"], np.float32),
             local_crop=float(d["local_crop"]),
             centers=np.asarray(d["centers"], np.float32),
             k_top=int(d["k_top"]) if "k_top" in d.files else 0,
             child_pc_zalign=child_zalign.astype(np.float32),
             child_pc_com=com_cloud.astype(np.float32),
             contacts=contacts_cloud.astype(np.float32),
             child_pc_com_4pt=com_pts_saved.astype(np.float32))
    print(f"\nwrote {out}/debug.npz  (child_pc_zalign from cloud-contact completion)")
    print(f"next: python vision_pipeline/tools/crop_thickness_sweep.py {out} --compute --save <dir>")


if __name__ == "__main__":
    main()
