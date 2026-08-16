import json
import torch
import numpy as np

from omni.isaac.lab.utils.math import matrix_from_quat, quat_from_matrix


def read_grasp_poses(grasps_json_path, device="cuda:0"):
    with open(grasps_json_path, "r") as f:
        grasps_dict = json.load(f)

    grasp_poses = []
    for grasp in grasps_dict:
        pos = grasp["pos"]
        quat = grasp["orn"]
        quat = [quat[-1]] + quat[:3]  # (x, y, z, w) -> (w, x, y, z)
        grasp_poses.append(pos + quat)

    return torch.tensor(grasp_poses, dtype=torch.float32, device=device)


def read_obj_poses(stable_poses_npy_path, key, device="cuda:0"):
    poses_arr = np.load(stable_poses_npy_path, allow_pickle=True)
    poses_arr = poses_arr.item()[key]
    return torch.tensor(poses_arr, dtype=torch.float32, device=device)


def read_rel_trans(rel_transform_npy_path, device="cuda:0"):
    mat_arr = np.load(rel_transform_npy_path, allow_pickle=True)
    mat_arr = mat_arr.item()["relative"]
    return torch.tensor(mat_arr, dtype=torch.float32, device=device)


def get_trans_mat_from_pose(pose, device="cuda:0"):
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)
    pose = pose.to(device)
    pos, quat = pose[..., :3], pose[..., 3:]

    mat = matrix_from_quat(quat)
    mat = torch.cat((mat, pos.unsqueeze(-1)), axis=-1)
    
    ext_row = torch.zeros(mat.shape[:-2] + (1, 4), dtype=mat.dtype, device=device)
    ext_row[..., 3] = 1

    mat = torch.cat((mat, ext_row), axis=-2)
    return mat


def get_pose_from_trans_mat(trans_mat):
    pos = trans_mat[..., :3, 3]
    quat = quat_from_matrix(trans_mat[..., :3, :3])
    return torch.cat((pos, quat), axis=-1)


def calc_obj_target_pose(orig_obj_pose, orig_base_pose, new_base_pose, device="cuda:0"):
    orig_obj_mat = get_trans_mat_from_pose(orig_obj_pose, device=device)
    orig_base_mat = get_trans_mat_from_pose(orig_base_pose, device=device)
    new_base_mat = get_trans_mat_from_pose(new_base_pose, device=device)

    obj_to_base = torch.matmul(torch.linalg.inv(orig_base_mat), orig_obj_mat)
    obj_target_mat = torch.matmul(new_base_mat, obj_to_base)

    pos = obj_target_mat[..., :3, 3]
    quat = quat_from_matrix(obj_target_mat[..., :3, :3])
    return torch.cat((pos, quat), axis=-1)


def calc_ee_pose(obj_pose, ee_to_obj_pose, device="cuda:0"):
    ee_to_obj_mat = get_trans_mat_from_pose(ee_to_obj_pose, device=device)
    obj_mat = get_trans_mat_from_pose(obj_pose, device=device)

    ee_mat = torch.matmul(obj_mat, ee_to_obj_mat)

    ee_pos = ee_mat[..., :3, 3]
    ee_quat = quat_from_matrix(ee_mat[..., :3, :3])
    return torch.cat((ee_pos, ee_quat), axis=-1)
