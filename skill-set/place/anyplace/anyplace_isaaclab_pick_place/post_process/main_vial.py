import os
import copy
import json
import numpy as np
import open3d as o3d

from utils import read_exp_cfg, load_mesh, read_obj_poses, read_rel_trans, get_trans_mat_from_pose


class CalcMetrics:
    def __init__(self, obj_model_info_path, collision_eps=0.005):
        self.obj_model_info_path = obj_model_info_path
        self.collision_eps = collision_eps

    def __call__(self, method, exp_path):
        collision_dist, hole_ids, cnt_execute_ok, cnt_inserted = self.calc_pose_metrics(method, exp_path)
        return {
            "precision": np.mean(collision_dist),
            "percentage_without_collision": np.mean(hole_ids >= 0),
            "coverage": np.mean([(hole_ids == i).any() for i in range(self.hole_pos.shape[0])]),
            "ok": cnt_execute_ok,
            "inserted": cnt_inserted,
        }

    def calc_pose_metrics(self, method, exp_path):
        self.prepare_exp(method, exp_path)
        list_hole_ids = []
        list_collision_dist = []
        for i in range(self.rel_trans_mat.shape[0]):
            obj_pose_mesh = self.get_obj_pose_mesh(i)
            intersect_pts = self.get_intersection_pts(obj_pose_mesh)
            if intersect_pts.shape[0]:
                hole_id = self.get_hole_id(intersect_pts)
                rad = self.calc_radius(hole_id)
                dis = self.calc_to_hole_center_dis(intersect_pts, hole_id)
                if dis > rad + self.collision_eps:
                    hole_id = -1
                    list_collision_dist.append(dis - rad)
                else:
                    list_collision_dist.append(max(dis - rad, 0))
            else:
                obj_pose_mat = self.obj_pose_mat[i]
                obj_pose_pts = self.obj_vertices @ obj_pose_mat[:3, :3].T + obj_pose_mat[np.newaxis, :3, 3]
                dis_to_hole = np.linalg.norm(obj_pose_pts[np.newaxis, ...] - self.hole_pose_pos[:, np.newaxis, :], axis=2)
                list_collision_dist.append(np.min(dis_to_hole))
                hole_id = -1
            list_hole_ids.append(hole_id)

        cnt_execute_ok = 0
        cnt_inserted = 0
        for i in range(self.results["grasp_id"].shape[0]):
            exec_state = self.results["execution_result"][i]
            if exec_state <= 1 or self.grasps[self.results["grasp_id"][i]]["pos"][2] < 0:
                continue
            cnt_execute_ok += 1
            cnt_inserted += self.is_inserted(i)

        return np.array(list_collision_dist), np.array(list_hole_ids), cnt_execute_ok, cnt_inserted

    def prepare_exp(self, method, exp_path):
        self.exp_path = exp_path
        self.exp_cfg = read_exp_cfg(method, self.exp_path)

        self.base_mesh = load_mesh(self.exp_cfg, "base")
        self.obj_mesh = load_mesh(self.exp_cfg, "obj")
        self.read_poses()
        self.read_model_info()
        self.process_base()
        self.process_obj()

    def read_poses(self):
        self.init_pose = read_obj_poses(self.exp_cfg["camera_info_path"], self.exp_cfg["obj"]["name"] + "_init")
        self.init_pose_mat = get_trans_mat_from_pose(self.init_pose)

        self.rel_trans_mat = read_rel_trans(self.exp_cfg["rel_transform_path"])
        self.init_pose_mat = self.init_pose_mat.repeat(self.rel_trans_mat.shape[0], axis=0)
        self.obj_pose_mat = np.matmul(self.rel_trans_mat, self.init_pose_mat)

        self.base_pose = read_obj_poses(self.exp_cfg["stable_poses_path"], self.exp_cfg["base"]["name"])[:1].repeat(self.obj_pose_mat.shape[0], axis=0)
        self.base_pose_mat = get_trans_mat_from_pose(self.base_pose)

        with open(self.exp_cfg["grasps"]["path"], "r") as f:
            self.grasps = json.load(f)

        self.results = np.load(self.exp_cfg["save_exp_result_path"], allow_pickle=True).item()

    def read_model_info(self):
        base_name = self.exp_cfg["base"]["name"]
        base_typ = base_name.split("_")[0]
        self.hole_pos = np.load(os.path.join(self.obj_model_info_path, base_typ, base_name + "_data.npy"))

    def process_base(self):
        self.base_pose_mesh = copy.deepcopy(self.base_mesh).transform(self.base_pose_mat[0])
        bounding_box = self.base_pose_mesh.get_axis_aligned_bounding_box()
        self.min_bound, self.max_bound = bounding_box.get_min_bound().copy(), bounding_box.get_max_bound().copy()
        self.min_bound[2] = self.max_bound[2] - 0.001
        self.bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.min_bound, self.max_bound)
        self.base_pose_mesh = self.base_pose_mesh.crop(self.bounding_box)
        self.base_pose_vertices = np.asarray(self.base_pose_mesh.vertices)

        self.hole_pose_pos = self.hole_pos @ self.base_pose_mat[0, :3, :3].T + self.base_pose_mat[0, np.newaxis, :3, 3]
        self.hole_pose_pos[:, 2] = self.max_bound[2]

        self.base_final_mat = get_trans_mat_from_pose(np.array(self.results["exp_cfg"]["base"]["pose"]))
        self.base_final_mesh = copy.deepcopy(self.base_mesh).transform(self.base_final_mat)

        bounding_box = self.base_final_mesh.get_axis_aligned_bounding_box()
        self.min_final_bound, self.max_final_bound = bounding_box.get_min_bound().copy(), bounding_box.get_max_bound().copy()

        self.min_final_bound[2] += self.collision_eps
        self.max_final_bound[2] -= self.collision_eps

    def process_obj(self):
        self.obj_vertices = np.asarray(self.obj_mesh.vertices)
        z_lb, z_ub = np.min(self.obj_vertices[:, 2]), np.max(self.obj_vertices[:, 2])
        self.lower_obj_pts = self.obj_vertices[self.obj_vertices[:, 2] < z_lb + 0.001]
        self.upper_obj_pts = self.obj_vertices[self.obj_vertices[:, 2] > z_ub - 0.001]

    def get_obj_pose_mesh(self, relative_pose_id: int = 0):
        return copy.deepcopy(self.obj_mesh).transform(self.init_pose_mat[0]).transform(self.rel_trans_mat[relative_pose_id])

    def get_intersection_pts(self, obj_pose_mesh):
        triangles = np.asarray(obj_pose_mesh.triangles)
        obj_line_ids = np.concatenate([triangles[:, [i, j]] for i in range(3) for j in range(i)])
        obj_line_coor = np.asarray(obj_pose_mesh.vertices)[obj_line_ids]
        cross_ids = (np.min(obj_line_coor[..., 2], axis=1) < self.min_bound[2]) * (np.max(obj_line_coor[..., 2], axis=1) > self.max_bound[2])

        cross_vertices = np.asarray(obj_pose_mesh.vertices)[obj_line_ids[cross_ids]]
        k = (self.max_bound[2] - cross_vertices[..., 0, 2]) / (cross_vertices[..., 1, 2] - cross_vertices[..., 0, 2])
        xy = cross_vertices[..., 0, :2] * (1 - k)[:, np.newaxis] + cross_vertices[..., 1, :2] * k[:, np.newaxis]
        return xy

    def get_hole_id(self, intersect_pts):
        center_pt = np.mean(intersect_pts, axis=0)
        center_to_hole_dis = np.linalg.norm(center_pt[np.newaxis, :] - self.hole_pos[:, :2], axis=1)
        return center_to_hole_dis.argmin()

    def calc_to_hole_center_dis(self, intersect_pts, hole_id):
        pts_to_hole_dis = np.linalg.norm(intersect_pts - self.hole_pos[[hole_id], :2], axis=1)
        return np.max(pts_to_hole_dis) if pts_to_hole_dis.shape else np.inf

    def calc_radius(self, hole_id):
        return np.min(np.linalg.norm(self.base_pose_vertices[:, :2] - self.hole_pos[[hole_id], :2], axis=1))

    def get_obj_final_mat(self, result_id: int = 0):
        return get_trans_mat_from_pose(self.results["final_object_pose"][result_id])

    def get_obj_final_surface_pts(self, result_id: int = 0):
        mat = self.get_obj_final_mat(result_id)
        return [pts @ mat[:3, :3].T + mat[:3, 3] for pts in [self.lower_obj_pts, self.upper_obj_pts]]

    def is_inserted(self, result_id: int = 0):
        def test_inside(vertices):
            ok = None
            for i in [2]: # range(3):
                ok_i = (vertices[:, i] > self.min_final_bound[i]) * (vertices[:, i] < self.max_final_bound[i])
                if ok is None:
                    ok = ok_i
                else:
                    ok *= ok_i
            return ok
        for surface_pts in self.get_obj_final_surface_pts(result_id):
            if np.mean(test_inside(surface_pts)) >= 1:
                return True
        return False


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "isaac_lab_pick_place", "data", "inserting")
    obj_model_info_path = os.path.join(os.path.dirname(__file__), "obj_model_with_info")
    list_methods = ["anyplace_diffusion_molmocrop"]

    for method in list_methods:
        print("Method:", method)

        list_precision = []
        list_percentage = []
        list_coverage = []
        list_success_rate = []

        calc_metrics = CalcMetrics(obj_model_info_path=obj_model_info_path)
        for exp_name in os.listdir(data_path):
            if exp_name.split("_")[0] not in ["vialplateobj"]:
                continue
            exp_path = os.path.join(data_path, exp_name)
            res = calc_metrics(method, exp_path)
            # print(exp_name, res)
            list_precision.append(res["precision"])
            list_percentage.append(res["percentage_without_collision"])
            list_coverage.append(res["coverage"])
            if res["ok"] > 0:
                list_success_rate.append(res["inserted"] / res["ok"])

        print("mean precision:", np.mean(list_precision))
        print("mean percentage:", np.mean(list_percentage))
        print("mean coverage:", np.mean(list_coverage))
        print("mean success rate:", np.mean(list_success_rate))
