"""open3d viewer for the child-object pipeline flow (run_pipeline_viz.py output). Each
stage is a separate colour, toggleable on/off:

  1 child_pc_i        gray     (logged raw partial)
  2 child_pc_refined  orange   (after DBSCAN outlier removal)
  3 contacts          magenta  (4 PaXini 지두 지문 centres, spheres)
  4 child_pc_com      blue     (Act-VH IGR completion)
  5 hand_pc           silver   (PaXini-URDF-FK hand cloud)
  6 child_pc          green    (child_pc_com + hand_pc fused)

    conda activate anyplace_cu128
    python vision_pipeline/tools/actvh_test/view_pipeline.py

Keys: 1..6 toggle layers   R reset view   Q / Esc quit
"""
import os

import numpy as np
import open3d as o3d

VIZ = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place/vision_pipeline/fixtures/actvh_test/pipeline_viz"

# (key, name, color 0-255)
LAYERS = [
    ("1", "child_pc_i", (150, 150, 150)),
    ("2", "child_pc_refined", (255, 140, 40)),
    ("3", "contacts", (255, 0, 255)),
    ("4", "child_pc_com", (60, 120, 255)),
    ("5", "hand_pc", (180, 180, 195)),
    ("6", "child_pc", (60, 220, 60)),
]


def _pcd(xyz, color):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(xyz, float).reshape(-1, 3))
    p.paint_uniform_color(np.array(color) / 255.0)
    return p


def _contact_spheres(xyz, color, r=0.006):
    out = []
    for c in np.asarray(xyz, float).reshape(-1, 3):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12)
        s.translate(c); s.compute_vertex_normals(); s.paint_uniform_color(np.array(color) / 255.0)
        out.append(s)
    return out


class Viewer:
    def __init__(self):
        self.geoms, self.show = {}, {}
        for _, name, color in LAYERS:
            if name == "contacts":
                self.geoms[name] = _contact_spheres(np.load(f"{VIZ}/contacts.npy"), color)
            else:
                self.geoms[name] = [_pcd(np.asarray(o3d.io.read_point_cloud(f"{VIZ}/{name}.ply").points), color)]
            self.show[name] = os.path.exists(f"{VIZ}/{name}.ply") or name == "contacts"
        # start with the raw + refined + contacts + completion visible; hand/fused off
        self.show.update({"hand_pc": False, "child_pc": False})

    def _active(self):
        g = []
        for _, name, _c in LAYERS:
            if self.show.get(name):
                g += self.geoms[name]
        return g

    def _title(self):
        return "  ".join(f"[{k}]{n} {'ON' if self.show.get(n) else 'off'}" for k, n, _ in LAYERS)

    def rebuild(self, vis):
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        for g in self._active():
            vis.add_geometry(g, reset_bounding_box=False)
        ctr.convert_from_pinhole_camera_parameters(cam)
        print("\r" + self._title() + "   ", end="", flush=True)
        return False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="pipeline flow (child object)", width=1280, height=900)
        opt = vis.get_render_option(); opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.11])

        def toggle(name):
            def cb(v):
                self.show[name] = not self.show.get(name); return self.rebuild(v)
            return cb
        for key, name, _c in LAYERS:
            vis.register_key_callback(ord(key), toggle(name))

        first = True
        for g in self._active():
            vis.add_geometry(g, reset_bounding_box=first); first = False
        print("keys: " + " ".join(f"{k}:{n}" for k, n, _ in LAYERS) + "  |  R reset  Q quit")
        print(self._title())
        vis.run(); vis.destroy_window()


if __name__ == "__main__":
    Viewer().run()
