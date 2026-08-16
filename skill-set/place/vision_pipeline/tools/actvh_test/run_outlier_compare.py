"""Compare point-cloud OUTLIER-REMOVAL methods for the IGR completion input: clean the
orange partial with each method, then run IGR x10 on the cleaned cloud. Saves per-method
cleaned clouds (kept + removed) + 10 completions, and prints a comparison table.

Goal = remove ONLY the far disconnected cluster (segmentation bleed ~2cm off the body),
NOT shave the object surface. The far cluster is DENSE (its local density = the surface),
so density/count methods (sor, ror) can't separate it; connectivity (dbscan, connected)
can, via the spatial gap. Distances to orange_C are used for the REPORT metrics only —
the removal methods themselves never use the object centre.

    conda activate anyplace_cu128
    python vision_pipeline/tools/actvh_test/run_outlier_compare.py [--n_results 10]
"""
import os
import sys
import argparse

import numpy as np
import torch

REPO = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place"
sys.path.insert(0, REPO)
from vision_pipeline.core import igr_complete as IGR       # noqa: E402
from vision_pipeline.core import outlier_removal as OR      # noqa: E402

TEST = f"{REPO}/vision_pipeline/fixtures/actvh_test"
OUT = f"{TEST}/outlier_compare"

# method -> (callable(partial)->keep-mask, short description). Order = good first.
METHODS = {
    "dbscan":    (lambda p: OR.dbscan(p, eps=0.005, min_samples=10),
                  "DBSCAN keep-largest (connectivity) — removes far cluster, keeps surface"),
    "connected": (lambda p: OR.connected(p, radius=0.004),
                  "connected-components keep-largest (connectivity twin of dbscan)"),
    "ror":       (lambda p: OR.ror(p, radius=0.005, min_pts=16),
                  "Radius Outlier Removal (density) — preserves surface but MISSES dense cluster"),
    "sor":       (lambda p: OR.sor(p, nb=20, std_ratio=2.0),
                  "Statistical OR (baseline) — SHAVES surface boundary, misses cluster"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_results", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--resolution", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    d = np.load(f"{TEST}/inputs.npz")
    partial = d["partial"].astype(np.float64)
    contacts = d["contacts"].astype(np.float32)
    C = d["orange_C"].astype(float)
    dist = np.linalg.norm(partial - C, axis=1)
    surface = dist < 0.049          # dense cap — must be preserved (report metric only)
    farclust = dist > 0.055         # the disconnected 108-pt cluster — must be removed
    import trimesh

    net = IGR.load_network(device)
    rows = []
    for name, (fn, desc) in METHODS.items():
        mask = np.asarray(fn(partial), bool)
        kept, removed = partial[mask], partial[~mask]
        shaved = int((~mask & surface).sum())
        fcl = int((~mask & farclust).sum())
        mdir = f"{OUT}/{name}"
        os.makedirs(mdir, exist_ok=True)
        np.savez(f"{mdir}/cleaned.npz", kept=kept.astype(np.float32),
                 removed=removed.astype(np.float32))
        exts = []
        for rep in range(args.n_results):
            world = IGR.complete(net, kept, contacts, device, seed=rep,
                                 iterations=args.iterations, resolution=args.resolution,
                                 denoise_partial=False)          # pre-cleaned; no built-in SOR
            if world is None:
                continue
            trimesh.PointCloud(world).export(f"{mdir}/result_{rep:02d}.ply")
            exts.append((world.max(0) - world.min(0)).max() * 100)
        me = float(np.mean(exts)) if exts else float("nan")
        rows.append((name, len(kept), len(removed), shaved, fcl, me, desc))
        print(f"[{name}] kept {len(kept)} removed {len(removed)} "
              f"(surf_shaved {shaved}, far_cluster {fcl}/{int(farclust.sum())}) "
              f"-> {len(exts)} completions, mean max-extent {me:.2f}cm")

    print("\n================ OUTLIER-REMOVAL COMPARISON (orange, real diam 8.5cm) ================")
    print(f"{'method':>10} {'removed':>8} {'surf_shaved':>12} {'far_clust_rm':>13} {'IGR_extent':>11}")
    for name, nk, nr, sh, fc, me, _ in rows:
        print(f"{name:>10} {nr:>8} {sh:>12} {str(fc)+'/'+str(int(farclust.sum())):>13} {me:>9.2f}cm")
    print("\nlower surf_shaved = better surface preservation; far_clust_rm should = full "
          f"({int(farclust.sum())}); IGR_extent near 8.5cm is best.")


if __name__ == "__main__":
    main()
