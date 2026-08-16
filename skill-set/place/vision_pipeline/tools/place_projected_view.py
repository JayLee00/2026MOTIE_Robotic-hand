"""Visualize the FINAL placement recipe on a saved run: PaXini-surface (cloud) completion +
optimal parent crop (CROP_MARGIN + CROP_THICKNESS extrude) + AnyPlace + CONTACT PROJECTION
(lift the object to first contact so it rests on the real tray, no penetration).

AnyPlace is stochastic, so it samples N placements; each is projected and its penetration
measured (against the real parent surface, per-column) before vs after.

Requires AnyPlace :8801.
  python vision_pipeline/tools/place_projected_view.py [run_dir] [--n 6] [--save DIR]
Interactive: N/P cycle samples.
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from vision_pipeline.core import geometry as G                         # noqa: E402
from vision_pipeline.orchestrator import CROP_MARGIN, CROP_THICKNESS   # noqa: E402
from vision_pipeline.models_client import ModelsHTTP                   # noqa: E402

GRAY, GREEN, RED, BLUE, HAND = [0.72]*3, [0.15, 0.8, 0.3], [0.9, 0.2, 0.2], [0.15, 0.5, 0.95], [0.55, 0.55, 0.62]


def load(run):
    d = np.load(os.path.join(run, "debug.npz"), allow_pickle=True)
    return dict(parent=np.asarray(d["parent_pc_full"], float), lc=float(d["local_crop"]),
                center=np.asarray(d["centers"], float).reshape(-1, 3)[int(d["k_top"]) if "k_top" in d.files else 0],
                childz=np.asarray(d["child_pc_zalign"], float),
                ncom=int(len(np.asarray(d["child_pc_com"], float))), run=run)


def sample(S, M):
    from scipy.spatial import cKDTree
    reg = G.crop_region(S["parent"], S["center"], S["lc"], axis=G.WORLD_DOWN, margin=CROP_MARGIN)
    thick = G.thicken(reg, CROP_THICKNESS)
    tree = cKDTree(S["parent"][:, :2])

    def pen(obj):                                     # frac object pts below local parent surface
        below = tot = 0
        for i, nb in enumerate(tree.query_ball_point(obj[:, :2], r=0.006)):
            if nb:
                tot += 1; below += obj[i, 2] < S["parent"][nb, 2].max() - 0.002
        return below / max(tot, 1)

    cand = np.asarray(M.place(thick, S["childz"]))
    coss = cand[:, 2, 2]
    k = int(np.argmax(coss))
    placed = G.apply(cand[k], S["childz"])
    obj = placed[:S["ncom"]]
    pb = pen(obj)
    dz, _ = G.contact_project(obj, S["parent"])
    pa = pen(obj + [0, 0, dz])
    return dict(cos=float(coss[k]), cos_min=float(coss.min()), cos_mean=float(coss.mean()),
                n_ge08=int((coss >= 0.8).sum()), dz=dz, pen_before=pb, pen_after=pa,
                placed=placed, placed_proj=placed + [0, 0, dz], thick=thick)


def _pcd(p, c, ds=1):
    import open3d as o3d
    p = np.asarray(p, float).reshape(-1, 3)[::ds]
    q = o3d.geometry.PointCloud(); q.points = o3d.utility.Vector3dVector(p)
    if len(p):
        q.paint_uniform_color(c)
    return q


def _cam(vis):
    vc = vis.get_view_control(); vc.set_front([0.3, -1.0, 0.42]); vc.set_up([0, 0, 1]); vc.set_zoom(0.45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--save", metavar="DIR")
    a = ap.parse_args()
    run = a.run or (sorted(glob.glob(os.path.join(REPO, "vision_pipeline/test_logs/run_*_cloudcontact")))
                    or sorted(glob.glob(os.path.join(REPO, "vision_pipeline/test_logs/run_*"))))[-1]
    S = load(run)
    M = ModelsHTTP()
    print(f"run {run}\nrecipe: cloud completion + crop(margin {CROP_MARGIN*100:.0f}cm, "
          f"thick {CROP_THICKNESS*100:.0f}cm) + contact projection\n")
    res = []
    for s in range(a.n):
        r = sample(S, M); res.append(r)
        print(f"  sample {s}: cos[min={r['cos_min']:+.2f} pick={r['cos']:+.2f}] {r['n_ge08']}/20>=0.8  "
              f"pen before={r['pen_before']*100:3.0f}%  lift={r['dz']*100:+.1f}cm  pen after={r['pen_after']*100:3.0f}%")
    print(f"\n  mean penetration: before={np.mean([r['pen_before'] for r in res])*100:.0f}%  "
          f"after={np.mean([r['pen_after'] for r in res])*100:.0f}%  (contact projection guarantees ~0)")

    import open3d as o3d
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for i, r in enumerate(res):
            vis = o3d.visualization.Visualizer(); vis.create_window(visible=False, width=900, height=720)
            for g in (_pcd(S["parent"], GRAY, 2), _pcd(r["thick"], GREEN),
                      _pcd(r["placed_proj"][:S["ncom"]], BLUE), _pcd(r["placed_proj"][S["ncom"]:], HAND)):
                vis.add_geometry(g)
            vis.get_render_option().point_size = 2.5; _cam(vis)
            vis.poll_events(); vis.update_renderer()
            vis.capture_screen_image(os.path.join(a.save, f"place_{i}.png"), do_render=True); vis.destroy_window()
        print(f"saved {len(res)} PNGs to {a.save}")
        return

    st = {"i": 0}
    par = _pcd(S["parent"], GRAY, 2); thick = _pcd(res[0]["thick"], GREEN)
    obj = _pcd(res[0]["placed_proj"][:S["ncom"]], BLUE); hand = _pcd(res[0]["placed_proj"][S["ncom"]:], HAND)
    before = _pcd(res[0]["placed"][:S["ncom"]], RED)
    vis = o3d.visualization.VisualizerWithKeyCallback(); vis.create_window("placement + contact projection", 1300, 940)
    for g in (par, thick, before, obj, hand):
        vis.add_geometry(g)
    vis.get_render_option().point_size = 3.0; _cam(vis)

    def redraw():
        r = res[st["i"]]
        thick.points = o3d.utility.Vector3dVector(r["thick"]); thick.paint_uniform_color(GREEN)
        before.points = o3d.utility.Vector3dVector(r["placed"][:S["ncom"]]); before.paint_uniform_color(RED)
        obj.points = o3d.utility.Vector3dVector(r["placed_proj"][:S["ncom"]]); obj.paint_uniform_color(BLUE)
        hand.points = o3d.utility.Vector3dVector(r["placed_proj"][S["ncom"]:]); hand.paint_uniform_color(HAND)
        for g in (thick, before, obj, hand):
            vis.update_geometry(g)
        print(f"[sample {st['i']}] cos={r['cos']:+.2f} lift={r['dz']*100:+.1f}cm "
              f"pen {r['pen_before']*100:.0f}%->{r['pen_after']*100:.0f}%  "
              f"(red=before proj, blue=object after, gray=hand, green=slab)")

    vis.register_key_callback(ord("N"), lambda _: (st.__setitem__("i", (st["i"]+1) % len(res)), redraw(), False)[-1])
    vis.register_key_callback(ord("P"), lambda _: (st.__setitem__("i", (st["i"]-1) % len(res)), redraw(), False)[-1])
    print("keys: N/P cycle samples")
    redraw(); vis.run(); vis.destroy_window()


if __name__ == "__main__":
    main()
