import os
import json
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d


def read_exp_cfg(method, exp_path):
    with open(os.path.join(exp_path, "exp_config.json"), "r") as f:
        exp_cfg = json.load(f)

    def traverse(cfg):
        for key, value in cfg.items():
            if "path" in key:
                cfg[key] = value.format(method=method)
                if not os.path.isabs(cfg[key]):
                    cfg[key] = os.path.join(exp_path, cfg[key])
            elif isinstance(value, dict):
                cfg[key] = traverse(cfg[key])
        return cfg

    return traverse(exp_cfg)


def load_mesh(exp_cfg, name, color=(0.5, 0.5, 0.5)):
    mesh_path = os.path.splitext(exp_cfg[name]["usd_path"])[0] + ".stl"
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    colors = np.array(color, dtype=np.float32)[np.newaxis, :]
    colors = colors.repeat(np.asarray(mesh.vertices).shape[0], axis=0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


def matrix_from_quat(quat_wxyz):
    quat_xyzw = np.concatenate([quat_wxyz[..., [i]] for i in [1, 2, 3, 0]], axis=-1)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def read_obj_poses(stable_poses_npy_path, key):
    poses_arr = np.load(stable_poses_npy_path, allow_pickle=True)
    poses_arr = poses_arr.item()[key]
    return np.array(poses_arr, dtype=np.float32)


def read_rel_trans(rel_transform_npy_path):
    mat_arr = np.load(rel_transform_npy_path, allow_pickle=True)
    mat_arr = mat_arr.item()["relative"]
    return np.array(mat_arr, dtype=np.float32)


def get_trans_mat_from_pose(pose):
    pos, quat = pose[..., :3], pose[..., 3:]

    mat = matrix_from_quat(quat)
    mat = np.concatenate((mat, pos[..., np.newaxis]), axis=-1)

    ext_row = np.zeros(mat.shape[:-2] + (1, 4), dtype=mat.dtype)
    ext_row[..., 3] = 1

    mat = np.concatenate((mat, ext_row), axis=-2)
    return mat


# Computational geometry
def calc_intersect(p1, p2, z):
    k = (z - p1[..., 2]) / (p2[..., 2] - p1[..., 2])
    return p1[..., :2] * (1 - k)[:, np.newaxis] + p2[..., :2] * k[:, np.newaxis]


def create_face_side_pts(face_pts):
    rot_face_side_pts = np.zeros(face_pts.shape, dtype=face_pts.dtype)
    rot_face_side_pts[..., :-1, :] = face_pts[..., 1:, :].copy()
    rot_face_side_pts[..., -1, :] = face_pts[..., 0, :].copy()
    res = np.concatenate((
        face_pts[..., np.newaxis, :],
        rot_face_side_pts[..., np.newaxis, :],
    ), axis=-2)
    return res


def test_pts_in_face(pts, face_side_pts):
    cross_prod = np.cross(face_side_pts[np.newaxis, ..., 0, :] - pts[:, np.newaxis, np.newaxis, ...],
                          face_side_pts[np.newaxis, ..., 1, :] - pts[:, np.newaxis, np.newaxis, ...])
    return np.all(cross_prod > 0, axis=-1)
