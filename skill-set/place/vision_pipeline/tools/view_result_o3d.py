"""Interactive Open3D viewer for a place-pipeline run's IMPROVED placement result.

Loads test_logs/<run>/improved_placement.npz (written by the offline recompute: dense hand_pc
+ AnyPlace re-inference) and shows, in one Open3D window with a per-layer visibility panel:

  object (child_pc_com) · dense hand_pc · fingertip contacts · parent tray ·
  child_pc_zalign (the AnyPlace input) · each candidate's placed child cloud +
  a palm-direction arrow coloured by upright-cos (green = palm-DOWN/gravity, red = palm-UP) ·
  the SELECTED placement (brightest) · a gravity arrow · world axes.

Run (env with open3d, on a machine with a display / X11):
    conda activate anyplace_cu128
    python vision_pipeline/tools/view_result_o3d.py [run_dir]     # default: latest run with the npz

Toggle layers with the checkboxes in the left "Geometries" panel (o3d.visualization.draw).
"""
import glob
import os
import sys

import numpy as np
import open3d as o3d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pcd(pts, color):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, float).reshape(-1, 3))
    p.paint_uniform_color(color)
    return p


def _apply(T, pc):
    T = np.asarray(T, float)
    return np.asarray(pc, float) @ T[:3, :3].T + T[:3, 3]


def _arrow(origin, direction, color, length=0.13, radius=0.0045):
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    d = d / n
    a = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, cone_radius=radius * 2.3,
        cylinder_height=length * 0.72, cone_height=length * 0.28)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-9:                         # parallel/anti-parallel to +z
        R = np.eye(3) if c > 0 else o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
    else:
        vn = v / np.linalg.norm(v)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(vn * np.arccos(np.clip(c, -1, 1)))
    a.rotate(R, center=(0, 0, 0))
    a.translate(np.asarray(origin, float))
    a.paint_uniform_color(color)
    a.compute_vertex_normals()
    return a


def _spheres(centers, color, r=0.006):
    mesh = o3d.geometry.TriangleMesh()
    for c in np.asarray(centers, float).reshape(-1, 3):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.translate(c)
        mesh += s
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def build(npz):
    z = np.load(npz, allow_pickle=True)
    obj, hand, parent = z["object"], z["hand"], z["parent"]
    zc, cands, cos = z["child_zalign"], z["candidates"], z["cos"]
    contacts = z["contacts"]
    sel = int(z["selected"])
    geoms = []                                           # (name, geometry, visible_by_default)
    geoms.append(("parent tray", _pcd(parent, [0.80, 0.78, 0.72]), True))
    geoms.append(("1 object (child_pc_com)", _pcd(obj, [0.55, 0.55, 0.58]), True))
    geoms.append((f"2 hand_pc dense ({len(hand)})", _pcd(hand, [0.15, 0.5, 1.0]), True))
    geoms.append(("3 contacts", _spheres(contacts, [1.0, 0.0, 1.0]), True))
    geoms.append(("4 child_pc_zalign (AnyPlace input)", _pcd(zc, [0.1, 0.8, 0.8]), False))
    # per-candidate placed clouds + palm arrows, coloured by cos (green down .. red up)
    for i, (T, cs) in enumerate(zip(cands, cos)):
        placed = _apply(T, zc)
        down = cs > 0
        col = [0.1, 0.75, 0.1] if down else [0.85, 0.15, 0.15]
        vis = (i == sel)                                 # only the selected placed cloud on by default
        tag = "SELECTED " if i == sel else ""
        geoms.append((f"{tag}placed cand{i} cos={cs:+.2f}", _pcd(placed, col), vis))
        palm = np.asarray(T)[:3, :3] @ np.array([0, 0, -1.0])   # palm axis after placement (zalign palm=-z)
        ar = _arrow(placed.mean(0), palm, col)
        if ar is not None:
            geoms.append((f"palm arrow cand{i} ({'DOWN' if down else 'UP'})", ar, True))
    # gravity reference at the selected placement + world axes
    g0 = _apply(cands[sel], zc).mean(0) if 0 <= sel < len(cands) else parent.mean(0)
    grav = _arrow(g0 + [0, 0, 0.18], [0, 0, -1.0], [0.05, 0.05, 0.05], length=0.15)
    if grav is not None:
        geoms.append(("gravity (world -z)", grav, True))
    geoms.append(("world axes", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1), False))
    return geoms, sel, cos


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    run = args[0] if args else None
    if run is None:
        hits = sorted(glob.glob(os.path.join(REPO, "vision_pipeline/test_logs/run_*/improved_placement.npz")))
        if not hits:
            sys.exit("no improved_placement.npz found — run the recompute first.")
        npz = hits[-1]
    else:
        npz = run if run.endswith(".npz") else os.path.join(run, "improved_placement.npz")
    print(f"[o3d] loading {npz}")
    geoms, sel, cos = build(npz)
    print(f"[o3d] candidates cos={np.round(cos, 3).tolist()}  selected={sel} "
          f"(cos={cos[sel]:+.2f}, {'palm-DOWN' if cos[sel] > 0 else 'palm-UP'})")
    print("[o3d] toggle layers with the left 'Geometries' panel checkboxes; drag=rotate, scroll=zoom.")
    named = [{"name": n, "geometry": g, "is_visible": v} for (n, g, v) in geoms]
    o3d.visualization.draw(named, title="place result — run improved placement",
                           show_skybox=False, point_size=3, bg_color=(1, 1, 1, 1))


if __name__ == "__main__":
    main()
