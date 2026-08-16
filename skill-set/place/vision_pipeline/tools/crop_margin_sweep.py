"""Sweep the parent-crop MARGIN and RE-RUN AnyPlace at each margin — using a saved run's
debug.npz — to see how the crop size actually changes the placement result.

The pipeline crops the parent point cloud around the placement point with
`geometry.crop_region(parent, center, local_crop, axis=WORLD_DOWN, margin=CROP_MARGIN)`
(an upright cylinder, radius `local_crop/2 + margin`) and feeds that crop to AnyPlace. This
tool, for each margin in {0, 0.5, 1, …, 5} cm, crops the SAME chosen placement point and calls
AnyPlace again (parent crop + the saved child_pc_zalign), so the placement pose (and its
upright cosine) is genuinely re-inferred per margin — NOT the one cached in the run.

Requires the AnyPlace service up:  conda activate anyplace_cu128 &&
  python vision_pipeline/services/anyplace_service.py --port 8801

  python vision_pipeline/tools/crop_margin_sweep.py                 # latest run, infer + view
  python vision_pipeline/tools/crop_margin_sweep.py <run_dir>
  python vision_pipeline/tools/crop_margin_sweep.py --table         # infer + numbers, no GUI
  python vision_pipeline/tools/crop_margin_sweep.py --save out/     # infer + a PNG per margin
  python vision_pipeline/tools/crop_margin_sweep.py --no-infer      # crop only (no AnyPlace)

Interactive keys:  N / P  next/prev margin
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from vision_pipeline.core import geometry as G   # noqa: E402

MARGINS = [0.0] + list(np.round(np.arange(0.005, 0.0500001, 0.005), 3))   # base + 0.5..5 cm
GRAY = [0.72, 0.72, 0.72]
GREEN = [0.10, 0.85, 0.25]        # crop region at the margin
BLUE = [0.20, 0.45, 0.95]         # placed child (re-inferred)


def latest_run():
    runs = sorted(glob.glob(os.path.join(REPO, "vision_pipeline/test_logs/run_*")))
    if not runs:
        sys.exit("no test_logs/run_* found")
    return runs[-1]


def load(run_dir):
    npz = os.path.join(run_dir, "debug.npz")
    if not os.path.exists(npz):
        sys.exit(f"no debug.npz in {run_dir}")
    d = np.load(npz, allow_pickle=True)
    return dict(parent=np.asarray(d["parent_pc_full"], float),
                local_crop=float(d["local_crop"]),
                centers=np.asarray(d["centers"], float).reshape(-1, 3),
                k_top=int(d["k_top"]) if "k_top" in d.files else 0,
                child_z=np.asarray(d["child_pc_zalign"], float),
                run=run_dir)


def crop_at(parent, center, local_crop, margin):
    return G.crop_region(parent, center, local_crop, axis=G.WORLD_DOWN, margin=margin)


def infer_sweep(S):
    """Crop + RE-RUN AnyPlace at each margin on the chosen candidate. Returns a list of
    {margin, region, K, cos, T, placed} (T/cos/placed None on empty crop or failure)."""
    from vision_pipeline.models_client import ModelsHTTP
    from vision_pipeline.services.rpc import post_npz  # noqa: F401 (import checks service module)
    M = ModelsHTTP()
    c, lc, child = S["centers"][S["k_top"]], S["local_crop"], S["child_z"]
    out = []
    for m in MARGINS:
        region = crop_at(S["parent"], c, lc, m)
        rec = {"margin": m, "region": region, "K": 0, "cos": None, "T": None, "placed": None}
        if len(region) >= 16:
            try:
                cand = np.asarray(M.place(region, child))          # (K,4,4) — real AnyPlace call
                k, sc = G.rank_upright(cand)
                rec.update(K=len(cand), cos=float(sc[k]), T=cand[k],
                           placed=G.apply(cand[k], child))
            except Exception as e:                                 # noqa: BLE001
                rec["err"] = f"{type(e).__name__}: {str(e)[:160]}"
        out.append(rec)
        cs = f"cos={rec['cos']:+.3f}" if rec["cos"] is not None else rec.get("err", "empty crop")
        print(f"  [infer] margin {m*100:4.1f}cm  crop {len(region):5d} pts  K={rec['K']:2d}  {cs}")
    return out


def print_table(S, res):
    lc = S["local_crop"]
    print(f"\nrun: {S['run']}")
    print(f"local_crop_size = {lc*100:.2f} cm (base radius {lc/2*100:.2f} cm), "
          f"chosen candidate k={S['k_top']}, center={np.round(S['centers'][S['k_top']],3).tolist()}\n")
    print(f"  {'margin':>9} {'radius':>7} {'crop pts':>9} {'K':>3} {'best cos':>9}  {'placed xyz (m)':>22}")
    for r in res:
        tag = "0.0(base)" if r["margin"] == 0 else f"+{r['margin']*100:.1f}cm"
        cos = f"{r['cos']:+.3f}" if r["cos"] is not None else "   —   "
        xyz = (np.round(r["T"][:3, 3], 3).tolist() if r["T"] is not None else "-")
        print(f"  {tag:>9} {(lc/2+r['margin'])*100:6.1f}c {len(r['region']):9d} "
              f"{r['K']:3d} {cos:>9}  {str(xyz):>22}")
    coss = [r["cos"] for r in res if r["cos"] is not None]
    if coss:
        print(f"\n  best-cos spread across margins: min {min(coss):+.3f}  max {max(coss):+.3f}  "
              f"(Δ={max(coss)-min(coss):.3f})  -> re-inference {'DID' if max(coss)-min(coss)>1e-4 else 'did NOT'} change it")
    print()


def crop_only_sweep(S):
    """No AnyPlace — just the crop regions (fallback / --no-infer)."""
    c, lc = S["centers"][S["k_top"]], S["local_crop"]
    return [{"margin": m, "region": crop_at(S["parent"], c, lc, m),
             "T": None, "cos": None, "K": 0, "placed": None} for m in MARGINS]


def _pcd(pts, color):
    import open3d as o3d
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, float).reshape(-1, 3))
    if len(p.points):
        p.paint_uniform_color(color)
    return p


def interactive(S, res):
    import open3d as o3d
    parent_pcd = _pcd(S["parent"], GRAY)
    region_pcd = _pcd(res[0]["region"], GREEN)
    placed_pcd = _pcd(res[0]["placed"] if res[0]["placed"] is not None else np.empty((0, 3)), BLUE)
    st = {"i": 0}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("parent crop + AnyPlace per margin", 1360, 940)
    for g in (parent_pcd, region_pcd, placed_pcd):
        vis.add_geometry(g)
    vis.get_render_option().point_size = 3.0

    def redraw():
        r = res[st["i"]]
        region_pcd.points = o3d.utility.Vector3dVector(r["region"])
        region_pcd.paint_uniform_color(GREEN)
        placed = r["placed"] if r["placed"] is not None else np.empty((0, 3))
        placed_pcd.points = o3d.utility.Vector3dVector(placed)
        if len(placed):
            placed_pcd.paint_uniform_color(BLUE)
        for g in (region_pcd, placed_pcd):
            vis.update_geometry(g)
        tag = "base" if r["margin"] == 0 else f"+{r['margin']*100:.1f}cm"
        cos = f"cos={r['cos']:+.3f} (K={r['K']})" if r["cos"] is not None else "(no placement)"
        print(f"[margin {tag}]  crop {len(r['region'])} pts  {cos}  green=crop  blue=re-inferred placement")

    def nxt(_):
        st["i"] = (st["i"] + 1) % len(res); redraw(); return False

    def prv(_):
        st["i"] = (st["i"] - 1) % len(res); redraw(); return False

    vis.register_key_callback(ord("N"), nxt)
    vis.register_key_callback(ord("P"), prv)
    print("keys: N/P cycle margin  ·  green=parent crop  ·  blue=AnyPlace placement re-inferred at that margin")
    redraw()
    vis.run()
    vis.destroy_window()


def save_pngs(S, res, out):
    import open3d as o3d
    os.makedirs(out, exist_ok=True)
    parent_pcd = _pcd(S["parent"], GRAY)
    region_pcd = _pcd(res[0]["region"], GREEN)
    placed_pcd = _pcd(res[0]["placed"] if res[0]["placed"] is not None else np.empty((0, 3)), BLUE)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=900)
    for g in (parent_pcd, region_pcd, placed_pcd):
        vis.add_geometry(g)
    vis.get_render_option().point_size = 3.0
    for r in res:
        region_pcd.points = o3d.utility.Vector3dVector(r["region"]); region_pcd.paint_uniform_color(GREEN)
        placed = r["placed"] if r["placed"] is not None else np.empty((0, 3))
        placed_pcd.points = o3d.utility.Vector3dVector(placed)
        if len(placed):
            placed_pcd.paint_uniform_color(BLUE)
        for g in (region_pcd, placed_pcd):
            vis.update_geometry(g)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(os.path.join(out, f"margin_{r['margin']*100:04.1f}cm.png"), do_render=True)
    vis.destroy_window()
    print(f"saved {len(res)} PNGs to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--table", action="store_true", help="numbers only, no GUI")
    ap.add_argument("--save", metavar="DIR", help="render a PNG per margin (offscreen)")
    ap.add_argument("--no-infer", action="store_true", help="crop regions only, skip AnyPlace")
    a = ap.parse_args()
    S = load(a.run or latest_run())
    res = crop_only_sweep(S) if a.no_infer else infer_sweep(S)
    print_table(S, res)
    if a.save:
        save_pngs(S, res, a.save)
    elif not a.table:
        interactive(S, res)


if __name__ == "__main__":
    main()
