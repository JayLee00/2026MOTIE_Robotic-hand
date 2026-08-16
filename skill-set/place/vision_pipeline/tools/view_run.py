"""Interactive open3d viewer for a place-run's coloured point clouds (host env
`anyplace_cu128`). Colours are baked into the .ply (parent=gray, region=green,
placed child / candidates=red-orange, child_pc_zalign=red).

    conda activate anyplace_cu128
    python vision_pipeline/tools/view_run.py                  # latest run, scene + top-1
    python vision_pipeline/tools/view_run.py --list           # list every .ply in the run
    python vision_pipeline/tools/view_run.py all              # overlay the main clouds (30-34)
    python vision_pipeline/tools/view_run.py 34_child_pc_zalign.ply
    python vision_pipeline/tools/view_run.py 40_candidates/cand_rank00*      # one candidate
    python vision_pipeline/tools/view_run.py <run_dir> <selector>            # explicit run

Drag = rotate, scroll = zoom. The triad at the origin is the world frame (z up)."""
import glob
import os
import sys

import open3d as o3d

BASE = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place/vision_pipeline/test_logs"


def latest_run():
    runs = sorted(glob.glob(os.path.join(BASE, "run_*")))
    return runs[-1] if runs else None


def resolve_run(tok):
    if tok and (os.path.isdir(tok) or os.path.isdir(os.path.join(BASE, tok))):
        return tok if os.path.isdir(tok) else os.path.join(BASE, tok)
    return None


def main():
    args = sys.argv[1:]
    run = resolve_run(args[0]) if args else None
    if run:
        args = args[1:]
    run = run or latest_run()
    if not run:
        print("no run_* dirs under", BASE)
        return
    sel = args[0] if args else "scene"

    if sel == "--list":
        for p in sorted(glob.glob(os.path.join(run, "**", "*.ply"), recursive=True)):
            print(os.path.relpath(p, run))
        return

    if sel in ("scene", "top1"):
        pattern = "41_scene_plus_top1.ply"
    elif sel == "all":
        pattern = "3[0-4]_*.ply"                      # parent_pc_full + child clouds
    else:
        pattern = sel

    paths = (sorted(glob.glob(os.path.join(run, pattern)))
             or sorted(glob.glob(os.path.join(run, "**", pattern), recursive=True)))
    if not paths:
        print(f"no .ply match '{pattern}' in {run}\n(try --list)")
        return

    geos = []
    for p in paths:
        pcd = o3d.io.read_point_cloud(p)
        print(f"loaded {os.path.relpath(p, run)}  ({len(pcd.points)} pts)")
        geos.append(pcd)
    geos.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(geos, window_name=f"{os.path.basename(run)} : {sel}")


if __name__ == "__main__":
    main()
