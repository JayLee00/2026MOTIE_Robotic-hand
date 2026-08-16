"""open3d viewer: per-ATTEMPT (pipeline run) parent_pc_full + cropped candidate clouds.

Scans test_logs/run_*/debug.npz — each run that has a `parent_pc_full` is one ATTEMPT.
N / P step through attempts. The DEFAULT view shows ONLY parent_pc_full (gray); pressing
M toggles the cropped candidate point clouds (each `region_*` in a distinct colour) on/off,
shown together with parent_pc_full. All clouds are in the world frame, so the camera is kept
as you step through attempts (R re-fits to the current one).

    conda activate anyplace_cu128
    python vision_pipeline/tools/view_attempts.py

Keys:  N / P  next / prev attempt     M  toggle candidate crops     R  reset/fit view    Q/Esc  quit
"""
import glob
import os
import re

import numpy as np
import open3d as o3d

LOGS = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place/vision_pipeline/test_logs"
PARENT_COLOR = (150, 150, 150)
# distinct bright colours for the candidate crops (tab10-ish), cycled if there are more.
CROP_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
               (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (255, 152, 150),
               (197, 176, 213), (196, 156, 148)]


def _pcd(xyz, color):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(xyz, float).reshape(-1, 3))
    p.paint_uniform_color(np.array(color) / 255.0)
    return p


def _load_attempts():
    """Each run_*/debug.npz with a parent_pc_full -> {name, parent(pcd), crops:[pcd,...]}."""
    attempts = []
    for npz in sorted(glob.glob(f"{LOGS}/run_*/debug.npz")):
        name = os.path.basename(os.path.dirname(npz))
        try:
            z = np.load(npz, allow_pickle=True)
        except Exception:                                          # noqa: BLE001
            continue
        if "parent_pc_full" not in z.files or not len(z["parent_pc_full"]):
            continue
        regs = sorted((k for k in z.files if re.fullmatch(r"region_\d+", k)),
                      key=lambda k: int(k.split("_")[1]))
        crops = [_pcd(z[k], CROP_COLORS[i % len(CROP_COLORS)])
                 for i, k in enumerate(regs) if len(z[k])]
        attempts.append({"name": name, "parent": _pcd(z["parent_pc_full"], PARENT_COLOR),
                         "crops": crops})
    return attempts


class Viewer:
    def __init__(self):
        self.attempts = _load_attempts()
        if not self.attempts:
            raise SystemExit(f"no attempts with parent_pc_full under {LOGS}")
        self.i = 0
        self.show_crops = False                                    # default: parent_pc_full only

    # ---- state (also used by the headless self-test) ----
    def step(self, d):
        self.i = (self.i + d) % len(self.attempts)

    def toggle(self):
        self.show_crops = not self.show_crops

    def _geoms(self):
        a = self.attempts[self.i]
        return [a["parent"]] + (a["crops"] if self.show_crops else [])

    def _title(self):
        a = self.attempts[self.i]
        return (f"attempt {self.i + 1}/{len(self.attempts)}  [{a['name']}]  "
                f"parent={len(a['parent'].points)} pts  crops={len(a['crops'])}  "
                f"M:crops {'ON' if self.show_crops else 'off'}")

    # ---- rendering ----
    def rebuild(self, vis, reset=False):
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        first = reset
        for g in self._geoms():
            vis.add_geometry(g, reset_bounding_box=first)         # fit to parent (1st) on reset
            first = False
        if not reset:
            ctr.convert_from_pinhole_camera_parameters(cam)       # keep the viewpoint across N/P/M
        print("\r" + self._title() + " " * 6, end="", flush=True)
        return False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="place: attempts (parent_pc_full + crops)",
                          width=1280, height=900)
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.11])

        vis.register_key_callback(ord("N"), lambda v: (self.step(+1), self.rebuild(v))[1])
        vis.register_key_callback(ord("P"), lambda v: (self.step(-1), self.rebuild(v))[1])
        vis.register_key_callback(ord("M"), lambda v: (self.toggle(), self.rebuild(v))[1])
        vis.register_key_callback(ord("R"), lambda v: self.rebuild(v, reset=True))

        first = True
        for g in self._geoms():
            vis.add_geometry(g, reset_bounding_box=first)
            first = False
        print("keys: N/P next/prev attempt | M toggle crops | R reset view | Q/Esc quit")
        print(self._title())
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    Viewer().run()
