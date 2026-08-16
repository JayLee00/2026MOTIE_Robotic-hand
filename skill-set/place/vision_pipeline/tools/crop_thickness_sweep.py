"""Sweep parent-crop MARGIN × downward THICKNESS and re-run AnyPlace at each combo.

parent_pc_k out of crop_region is a thin surface, so AnyPlace slides that sheet into the gap
between the grasped object and the hand instead of resting the object on it. This experiment
extrudes each margin's crop DOWNWARD (gravity, world -z) by 1..5 cm (geometry.thicken) so it
becomes a solid slab, re-runs AnyPlace on the thickened crop + the saved child, and measures
how much the placed child still penetrates below the parent surface (wedged) vs sits on top.

Requires AnyPlace up:  python vision_pipeline/services/anyplace_service.py --port 8801

  python vision_pipeline/tools/crop_thickness_sweep.py --compute   # run the 10x5 sweep -> npz
  python vision_pipeline/tools/crop_thickness_sweep.py             # interactive view (cached npz)
  python vision_pipeline/tools/crop_thickness_sweep.py --save DIR  # render a PNG per combo
  python vision_pipeline/tools/crop_thickness_sweep.py --table     # numbers only

Interactive keys:  N/P margin   ·   [ / ] thickness
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

MARGINS = list(np.round(np.arange(0.005, 0.0500001, 0.005), 3))    # 0.5 .. 5 cm  (10)
THICKS = list(np.round(np.arange(0.010, 0.0500001, 0.010), 3))     # 1 .. 5 cm    (5)
GRAY, GREEN, BLUE = [0.72, 0.72, 0.72], [0.10, 0.85, 0.25], [0.20, 0.45, 0.95]


def cache_path(run_dir):
    return os.path.join(run_dir, "_thickness_sweep.npz")


def latest_run():
    runs = sorted(glob.glob(os.path.join(REPO, "vision_pipeline/test_logs/run_*")))
    if not runs:
        sys.exit("no test_logs/run_* found")
    return runs[-1]


def load(run_dir):
    d = np.load(os.path.join(run_dir, "debug.npz"), allow_pickle=True)
    return dict(parent=np.asarray(d["parent_pc_full"], float), local_crop=float(d["local_crop"]),
                center=np.asarray(d["centers"], float).reshape(-1, 3)[int(d["k_top"]) if "k_top" in d.files else 0],
                child=np.asarray(d["child_pc_zalign"], float), run=run_dir)


def thick_crop(S, m, t):
    reg = G.crop_region(S["parent"], S["center"], S["local_crop"], axis=G.WORLD_DOWN, margin=m)
    return reg, G.thicken(reg, t)


def penetration(reg, placed):
    """(parent_top z, frac of placed child below the surface, object-bottom gap to surface)."""
    top = float(np.percentile(reg[:, 2], 90))
    below = float((placed[:, 2] < top - 0.002).mean())
    gap = float(placed[:, 2].min() - top)          # <0 penetrates, ~0 rests, >0 floats
    return top, below, gap


def compute(S):
    from vision_pipeline.models_client import ModelsHTTP
    M = ModelsHTTP()
    nm, nt = len(MARGINS), len(THICKS)
    T = np.zeros((nm, nt, 4, 4)); cos = np.zeros((nm, nt)); below = np.zeros((nm, nt))
    gap = np.zeros((nm, nt)); npts = np.zeros((nm, nt), int)
    for i, m in enumerate(MARGINS):
        for j, t in enumerate(THICKS):
            reg, thick = thick_crop(S, m, t)
            npts[i, j] = len(thick)
            cand = np.asarray(M.place(thick, S["child"]))              # RE-INFER on the thick slab
            k, sc = G.rank_upright(cand)
            T[i, j] = cand[k]; cos[i, j] = sc[k]
            _, below[i, j], gap[i, j] = penetration(reg, G.apply(cand[k], S["child"]))
            print(f"  m={m*100:4.1f}cm t={t*100:.0f}cm  {len(thick):6d} pts  cos={sc[k]:+.3f}  "
                  f"below={below[i,j]*100:3.0f}%  gap={gap[i,j]*100:+.1f}cm")
    np.savez(cache_path(S["run"]), margins=MARGINS, thicks=THICKS, T=T, cos=cos, below=below,
             gap=gap, npts=npts, center=S["center"], local_crop=S["local_crop"], run=S["run"])
    print(f"\nsaved sweep -> {cache_path(S['run'])}")
    return dict(margins=MARGINS, thicks=THICKS, T=T, cos=cos, below=below, gap=gap, npts=npts)


def load_cache(run_dir):
    if not os.path.exists(cache_path(run_dir)):
        sys.exit(f"no cached sweep in {run_dir} — run with --compute first (needs AnyPlace :8801)")
    d = np.load(cache_path(run_dir), allow_pickle=True)
    return {k: d[k] for k in d.files}


def print_table(R):
    mg, th = R["margins"], R["thicks"]
    print("\n  child-below-surface %% (lower = rests on top, not wedged) — rows=margin, cols=thickness")
    print("        " + "".join(f"{t*100:5.0f}cm" for t in th))
    for i, m in enumerate(mg):
        print(f"  m{m*100:4.1f} " + "".join(f"{R['below'][i,j]*100:6.0f}" for j in range(len(th))))
    print("\n  object-bottom gap to surface (cm; ~0 rests, <0 penetrates) — rows=margin, cols=thickness")
    print("        " + "".join(f"{t*100:5.0f}cm" for t in th))
    for i, m in enumerate(mg):
        print(f"  m{m*100:4.1f} " + "".join(f"{R['gap'][i,j]*100:+6.1f}" for j in range(len(th))))
    b0 = R["below"][:, 0].mean()
    print(f"\n  mean child-below-surface: thinnest(1cm)={R['below'][:,0].mean()*100:.0f}%  "
          f"thickest(5cm)={R['below'][:,-1].mean()*100:.0f}%  "
          f"(baseline thin crop ~63%). Thickening reduces wedging.")


def _pcd(pts, color):
    import open3d as o3d
    p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(np.asarray(pts, float).reshape(-1, 3))
    if len(p.points):
        p.paint_uniform_color(color)
    return p


def _view(vis):
    vc = vis.get_view_control(); vc.set_front([0.25, -1.0, 0.4]); vc.set_up([0, 0, 1]); vc.set_zoom(0.5)


def interactive(S, R):
    import open3d as o3d
    st = {"i": 0, "j": 0}
    reg, thick = thick_crop(S, R["margins"][0], R["thicks"][0])
    parent_pcd, region_pcd = _pcd(S["parent"], GRAY), _pcd(thick, GREEN)
    placed_pcd = _pcd(G.apply(R["T"][0, 0], S["child"]), BLUE)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("crop margin x thickness -> AnyPlace", 1300, 940)
    for g in (parent_pcd, region_pcd, placed_pcd):
        vis.add_geometry(g)
    vis.get_render_option().point_size = 3.0
    _view(vis)

    def redraw():
        m, t = R["margins"][st["i"]], R["thicks"][st["j"]]
        reg, thick = thick_crop(S, m, t)
        region_pcd.points = o3d.utility.Vector3dVector(thick); region_pcd.paint_uniform_color(GREEN)
        placed = G.apply(R["T"][st["i"], st["j"]], S["child"])
        placed_pcd.points = o3d.utility.Vector3dVector(placed); placed_pcd.paint_uniform_color(BLUE)
        for g in (region_pcd, placed_pcd):
            vis.update_geometry(g)
        print(f"[margin {m*100:.1f}cm  thick {t*100:.0f}cm]  slab {len(thick)} pts  "
              f"cos={R['cos'][st['i'],st['j']]:+.3f}  below-surface={R['below'][st['i'],st['j']]*100:.0f}%")

    def mv(di, dj):
        def cb(_):
            st["i"] = (st["i"] + di) % len(R["margins"]); st["j"] = (st["j"] + dj) % len(R["thicks"])
            redraw(); return False
        return cb
    vis.register_key_callback(ord("N"), mv(1, 0)); vis.register_key_callback(ord("P"), mv(-1, 0))
    vis.register_key_callback(ord("]"), mv(0, 1)); vis.register_key_callback(ord("["), mv(0, -1))
    print("keys: N/P margin  ·  [ / ] thickness  ·  green=thick slab  blue=re-inferred placement")
    redraw(); vis.run(); vis.destroy_window()


def save_pngs(S, R, out):
    import open3d as o3d
    os.makedirs(out, exist_ok=True)
    parent_pcd = _pcd(S["parent"], GRAY)
    region_pcd = _pcd(thick_crop(S, R["margins"][0], R["thicks"][0])[1], GREEN)
    placed_pcd = _pcd(G.apply(R["T"][0, 0], S["child"]), BLUE)
    vis = o3d.visualization.Visualizer(); vis.create_window(visible=False, width=640, height=480)
    for g in (parent_pcd, region_pcd, placed_pcd):
        vis.add_geometry(g)
    vis.get_render_option().point_size = 2.5
    _view(vis)
    for i, m in enumerate(R["margins"]):
        for j, t in enumerate(R["thicks"]):
            thick = thick_crop(S, m, t)[1]
            region_pcd.points = o3d.utility.Vector3dVector(thick); region_pcd.paint_uniform_color(GREEN)
            placed_pcd.points = o3d.utility.Vector3dVector(G.apply(R["T"][i, j], S["child"])); placed_pcd.paint_uniform_color(BLUE)
            for g in (region_pcd, placed_pcd):
                vis.update_geometry(g)
            vis.poll_events(); vis.update_renderer()
            vis.capture_screen_image(os.path.join(out, f"m{m*100:04.1f}_t{t*100:02.0f}.png"), do_render=True)
    vis.destroy_window()
    print(f"saved {len(R['margins'])*len(R['thicks'])} PNGs to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--compute", action="store_true", help="run the AnyPlace sweep -> cache npz")
    ap.add_argument("--save", metavar="DIR", help="render a PNG per combo")
    ap.add_argument("--table", action="store_true", help="print the tables, no GUI")
    a = ap.parse_args()
    S = load(a.run or latest_run())
    R = compute(S) if a.compute else load_cache(S["run"])
    print_table(R)
    if a.save:
        save_pngs(S, R, a.save)
    elif not a.table and not a.compute:
        interactive(S, R)


if __name__ == "__main__":
    main()
