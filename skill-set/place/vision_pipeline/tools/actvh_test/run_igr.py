#!/usr/bin/env python
"""Act-VH (visuo-haptic) IGR completion of the real orange, x30 -> igr_results/.

Input = the orange partial + fingertip-PAD contact points (inputs.npz). The 30
outputs differ by optimization + surface-sampling seed. The reconstruction core
lives in vision_pipeline/core/igr_complete.py (shared with the live pipeline
service); this file is just the 30-rep driver + extent report.

Env: conda activate anyplace_cu128  (torch 2.7 + cu128, RTX 5090 sm_120).

    python vision_pipeline/tools/actvh_test/run_igr.py
"""
import os
import sys
import argparse

import numpy as np
import torch

REPO = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place"
sys.path.insert(0, REPO)
from vision_pipeline.core import igr_complete as IGR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=f"{REPO}/vision_pipeline/fixtures/actvh_test/inputs.npz")
    ap.add_argument("--out_dir", default=f"{REPO}/vision_pipeline/fixtures/actvh_test/igr_results")
    ap.add_argument("--n_results", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--no_denoise", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device,
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    d = np.load(args.inputs)
    partial = d["partial"].astype(np.float32)
    contacts = d["contacts"].astype(np.float32)
    orange_R = float(d["orange_R"])
    print(f"partial {partial.shape}, contacts {contacts.shape}, "
          f"orange diameter ref = {2*orange_R*100:.2f} cm")

    if not args.no_denoise:
        kept = IGR.denoise(partial)
        print(f"DBSCAN outlier removal: {len(partial)} -> {len(kept)} "
              f"({len(partial)-len(kept)} outliers removed)")

    net = IGR.load_network(device)
    import trimesh

    extents, produced = [], 0
    for rep in range(args.n_results):
        world = IGR.complete(net, partial, contacts, device,
                             seed=args.base_seed + rep,
                             iterations=args.iterations, resolution=args.resolution,
                             denoise_partial=not args.no_denoise,
                             sor_nb=args.sor_nb, sor_std=args.sor_std)
        if world is None:
            print(f"rep {rep:02d}: no surface extracted, skipping")
            continue
        ext = world.max(axis=0) - world.min(axis=0)
        extents.append(ext)
        out = os.path.join(args.out_dir, f"result_{rep:02d}.ply")
        trimesh.PointCloud(world).export(out)
        produced += 1
        if rep < 3 or rep == args.n_results - 1:
            print(f"rep {rep:02d}: {world.shape[0]} pts, extent(cm)="
                  f"[{ext[0]*100:.2f}, {ext[1]*100:.2f}, {ext[2]*100:.2f}] -> {out}")

    print(f"\nproduced {produced}/{args.n_results} point clouds in {args.out_dir}")
    if extents:
        e = np.array(extents)
        print(f"orange diameter reference: {2*orange_R*100:.2f} cm")
        print(f"result extent mean (cm): "
              f"[{e[:,0].mean()*100:.2f}, {e[:,1].mean()*100:.2f}, {e[:,2].mean()*100:.2f}]")
        print(f"result max-extent per cloud (cm): mean={e.max(axis=1).mean()*100:.2f}, "
              f"min={e.max(axis=1).min()*100:.2f}, max={e.max(axis=1).max()*100:.2f}")


if __name__ == "__main__":
    main()
