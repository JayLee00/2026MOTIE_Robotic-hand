import os
import json
import torch
import numpy as np
from typing import Sequence

from workspace_config import WorkspaceConfiguration
from pick_place_sim import PickPlaceSimulation
from retrieve_grasps import (
    read_grasp_poses,
    read_obj_poses,
    read_rel_trans,
    get_trans_mat_from_pose,
    get_pose_from_trans_mat
)


class PickPlaceExperiment:
    def __init__(
        self,
        method,
        exp_cfg_path,
        num_envs,
        device,
        max_tries=100,
        try_stable=False,
        pose_ids=None,
        grasp_ids=None,
    ):
        self.method = method
        self.num_envs = num_envs
        self.device = device
        self.max_tries = max_tries
        self.try_stable = try_stable

        self.exp_cfg = self.read_exp_cfg(exp_cfg_path)
        self.read_stable_poses(pose_ids)
        self.read_grasps(grasp_ids)
        self.initialize_workspace_cfg()

        self.pick_place_sim = PickPlaceSimulation(
            self.workspace_cfg,
            self.num_envs,
            self.device,
            **self.sim_kwargs,
        )

    def execute(self, save_result=True):
        stable_pose_id = torch.arange(self.stable_obj_poses.shape[0], device=self.device)
        grasp_id = torch.arange(self.grasp_poses.shape[0], device=self.device)
        print('stable_pose_id.shape:', stable_pose_id.shape)
        print('grasp_id.shape:', grasp_id.shape)

        list_exec_res, list_obj_pose = [], []

        grid = torch.meshgrid(stable_pose_id, grasp_id)
        grid = tuple(map(lambda t: t.unsqueeze(-1), grid))
        pair_id = torch.concatenate(grid, axis=-1).reshape(-1, 2)

        if pair_id.shape[0] > self.max_tries:
            pair_id = pair_id[torch.randperm(pair_id.shape[0])[:self.max_tries], :]

        batch_exps = (pair_id.shape[0] + self.num_envs - 1) // self.num_envs
        for i in range(batch_exps):
            print('Starting batch experiment %d / %d' % (i+1, batch_exps))
            if i > 0:
                self.pick_place_sim.reset()

            st_i = i * self.num_envs
            ed_i = st_i + self.num_envs
            batch_pair_id = pair_id[st_i:ed_i]
            while batch_pair_id.shape[0] < self.num_envs:
                ext_pair_id = pair_id[:self.num_envs - batch_pair_id.shape[0]]
                batch_pair_id = torch.cat((batch_pair_id, ext_pair_id))

            # print('Pair id:', batch_pair_id)
            batch_stable_pose_id, batch_grasp_id = batch_pair_id[:, 0], batch_pair_id[:, 1]
            exec_res, obj_pose = self.pick_place_sim.run_pick_place(
                self.grasp_poses[batch_grasp_id],
                self.stable_obj_poses[batch_stable_pose_id],
                self.stable_base_poses[batch_stable_pose_id],
            )

            list_exec_res.append(exec_res)
            list_obj_pose.append(obj_pose)

            print()

        exec_res = torch.concatenate(list_exec_res).flatten()
        obj_pose = torch.concatenate(list_obj_pose).reshape(-1, 7)
        exec_res = exec_res[:pair_id.shape[0]]
        obj_pose = obj_pose[:pair_id.shape[0]]

        exp_result = {
            "exp_cfg": self.exp_cfg,
            "grasp_id": pair_id[:, 1].cpu().numpy(),
            "stable_pose_id": pair_id[:, 0].cpu().numpy(),
            "execution_result": exec_res.cpu().numpy(),
            "final_object_pose": obj_pose.cpu().numpy(),
        }

        if save_result:
            with open(self.exp_cfg["save_exp_result_path"], "wb") as f:
                np.save(f, exp_result)
        
        return exp_result

    def read_exp_cfg(self, exp_cfg_path):
        with open(exp_cfg_path, "r") as f:
            exp_cfg = json.load(f)

        root_path = os.path.abspath(os.path.dirname(exp_cfg_path))
        def traverse(exp_cfg):
            nonlocal root_path
            for key, value in exp_cfg.items():
                if "path" in key: 
                    exp_cfg[key] = value = value.format(method=self.method)
                    if not os.path.isabs(value):
                        exp_cfg[key] = os.path.join(root_path, value)
                elif isinstance(value, dict):
                    exp_cfg[key] = traverse(value)
            return exp_cfg
        
        return traverse(exp_cfg)

    def read_stable_poses(self, pose_ids: Sequence[int] = None):
        if pose_ids is None:
            pose_ids = slice(pose_ids)
        if "camera_info_path" in self.exp_cfg and "rel_transform_path" in self.exp_cfg and not self.try_stable:
            init_pose = read_obj_poses(
                self.exp_cfg["camera_info_path"],
                self.exp_cfg["obj"]["name"] + "_init",
                device=self.device,
            )
            init_pose_mat = get_trans_mat_from_pose(init_pose)

            rel_trans_mat = read_rel_trans(self.exp_cfg["rel_transform_path"])

            init_pose_mat = init_pose_mat.repeat(rel_trans_mat.shape[0], 1, 1)

            # obj_poses_mat = torch.matmul(init_pose_mat, rel_trans_mat)
            obj_poses_mat = torch.matmul(rel_trans_mat, init_pose_mat)
            self.stable_obj_poses = get_pose_from_trans_mat(obj_poses_mat)
            self.stable_base_poses = read_obj_poses(
                self.exp_cfg["stable_poses_path"],
                self.exp_cfg["base"]["name"],
                device=self.device,
            )[:1].repeat(rel_trans_mat.shape[0], 1)
        else:
            path = self.exp_cfg["stable_poses_path"]
            self.stable_obj_poses = read_obj_poses(path, self.exp_cfg["obj"]["name"], device=self.device)
            self.stable_base_poses = read_obj_poses(path, self.exp_cfg["base"]["name"], device=self.device)
        self.stable_obj_poses = self.stable_obj_poses[pose_ids]
        self.stable_base_poses = self.stable_base_poses[pose_ids]
        self.sim_kwargs = {}
        for key in "place_axis", "approach_dis":
            if key in self.exp_cfg["obj"]:
                self.sim_kwargs[key] = self.exp_cfg["obj"][key]

    def read_grasps(self, grasp_ids: Sequence[int] = None):
        path = self.exp_cfg["grasps"]["path"]
        self.grasp_poses = read_grasp_poses(path, device=self.device)

        if grasp_ids is None:
            max_n_grasps = self.exp_cfg["grasps"]["max_num"]
            if self.grasp_poses.shape[0] > max_n_grasps:
                self.grasp_poses = self.grasp_poses[:max_n_grasps, ...].contiguous()
        else:
            self.grasp_poses = self.grasp_poses[grasp_ids]

    def initialize_workspace_cfg(self):
        self.workspace_cfg = WorkspaceConfiguration(
            self.exp_cfg["base"]["pose"],
            self.exp_cfg["base"]["usd_path"],
            self.exp_cfg["obj"]["pose"],
            self.exp_cfg["obj"]["usd_path"],
        )

    def close(self):
        self.pick_place_sim.close()
