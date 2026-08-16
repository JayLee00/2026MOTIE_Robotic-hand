"""open3d viewer to COMPARE point-cloud outlier-removal methods for the IGR completion.
For each method it shows the cleaned (kept) orange partial, the REMOVED outliers (red —
so surface-shaving is visible), the 10 IGR completions, the PaXini fingerprint contacts,
and the kistar hand. Cycle methods with M to see which removed only the far cluster vs
which shaved the surface.

    conda activate anyplace_cu128
    python vision_pipeline/tools/actvh_test/view.py            # start on dbscan
    python vision_pipeline/tools/actvh_test/view.py sor        # start on a given method

Keys:  1 partial(kept)   2 contacts (PaXini)   3 result   4 kistar hand   5 removed(red)
       N / P  next / prev completion     M  switch method
       R  reset view    Q / Esc  quit
"""
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d

TEST = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place/vision_pipeline/fixtures/actvh_test"
OUT = f"{TEST}/outlier_compare"
METHODS = ["dbscan", "connected", "ror", "sor"]


def _pcd(xyz, color):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(xyz, float).reshape(-1, 3))
    p.paint_uniform_color(np.array(color) / 255.0)
    return p


def _contact_spheres(xyz, color=(255, 0, 255), r=0.006):
    out = []
    for c in np.asarray(xyz, float).reshape(-1, 3):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12)
        s.translate(c); s.compute_vertex_normals(); s.paint_uniform_color(np.array(color) / 255.0)
        out.append(s)
    return out


def _load_hand():
    meshes = []
    spec = json.load(open(f"{TEST}/hand_meshes.json")) if os.path.exists(f"{TEST}/hand_meshes.json") else {}
    for info in spec.values():
        try:
            m = o3d.io.read_triangle_mesh(info["mesh"])
            if not m.has_vertices():
                continue
            m.scale(float(info.get("scale", [1])[0]), center=(0, 0, 0))
            m.transform(np.array(info["pose"], float))
            m.compute_vertex_normals(); m.paint_uniform_color([0.55, 0.55, 0.58])
            meshes.append(m)
        except Exception:                                          # noqa: BLE001
            pass
    return meshes


class Viewer:
    def __init__(self, method="dbscan"):
        inp = np.load(f"{TEST}/inputs.npz")
        self.contacts = _contact_spheres(inp["contacts"])
        self.hand = _load_hand()
        self.data = {}
        for m in METHODS:
            cl = np.load(f"{OUT}/{m}/cleaned.npz")
            plys = sorted(glob.glob(f"{OUT}/{m}/result_*.ply"))
            self.data[m] = {
                "kept": _pcd(cl["kept"], (150, 150, 150)),
                "removed": _pcd(cl["removed"], (255, 40, 40)) if len(cl["removed"]) else None,
                "n_removed": len(cl["removed"]),
                "results": [o3d.io.read_point_cloud(p) for p in plys],
            }
        self.method = method if method in self.data and self.data[method]["results"] else METHODS[0]
        self.ri = 0
        self.show = {"partial": True, "contacts": True, "result": True, "hand": True, "removed": True}

    def _result(self):
        res = self.data[self.method]["results"]
        if not res:
            return None
        self.ri %= len(res)
        r = res[self.ri]
        r.paint_uniform_color([0.2, 0.7, 1.0])
        return r

    def _title(self):
        dm = self.data[self.method]
        n = len(dm["results"])
        return (f"[{self.method}]  removed {dm['n_removed']}  completion {self.ri + 1 if n else 0}/{n}  "
                f"1:partial {self.show['partial']} 2:contacts {self.show['contacts']} "
                f"3:result {self.show['result']} 4:hand {self.show['hand']} 5:removed(red) {self.show['removed']}")

    def _geoms(self):
        dm = self.data[self.method]
        g = []
        if self.show["partial"]:
            g.append(dm["kept"])
        if self.show["removed"] and dm["removed"] is not None:
            g.append(dm["removed"])
        if self.show["contacts"]:
            g += self.contacts
        if self.show["result"]:
            r = self._result()
            if r is not None:
                g.append(r)
        if self.show["hand"]:
            g += self.hand
        return g

    def rebuild(self, vis):
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        for g in self._geoms():
            vis.add_geometry(g, reset_bounding_box=False)
        ctr.convert_from_pinhole_camera_parameters(cam)
        print("\r" + self._title() + " " * 4, end="", flush=True)
        return False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="outlier-removal comparison", width=1280, height=900)
        opt = vis.get_render_option(); opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.11])

        def toggle(key):
            def cb(v):
                self.show[key] = not self.show[key]; return self.rebuild(v)
            return cb
        for k, name in zip("12345", ("partial", "contacts", "result", "hand", "removed")):
            vis.register_key_callback(ord(k), toggle(name))

        def nxt(v):
            self.ri += 1; return self.rebuild(v)

        def prv(v):
            self.ri -= 1; return self.rebuild(v)

        def swap(v):
            i = METHODS.index(self.method)
            self.method = METHODS[(i + 1) % len(METHODS)]; self.ri = 0
            return self.rebuild(v)
        vis.register_key_callback(ord("N"), nxt)
        vis.register_key_callback(ord("P"), prv)
        vis.register_key_callback(ord("M"), swap)

        first = True
        for g in self._geoms():
            vis.add_geometry(g, reset_bounding_box=first); first = False
        print("keys: 1 partial | 2 contacts | 3 result | 4 hand | 5 removed(red) | N/P cycle | M method | Q quit")
        print(self._title())
        vis.run(); vis.destroy_window()


if __name__ == "__main__":
    Viewer(sys.argv[1] if len(sys.argv) > 1 else "dbscan").run()
