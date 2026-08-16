import os
import json
import argparse
import torch
from typing import Sequence

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="NVIDIA Isaac Lab Grasp Simulation")
parser.add_argument("--num-envs", type=int, default=1, help="the number of environments")
parser.add_argument("--exp", type=str, required=True, help="the path to the experiment folder")
parser.add_argument("--try-stable", action="store_true", help="try grasps from stable poses")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils.math import matrix_from_quat

from workspace_config import WorkspaceConfiguration
from scene import GraspingSceneCfg
from retrieve_grasps import (
    read_obj_poses,
    read_rel_trans,
    get_trans_mat_from_pose,
    get_pose_from_trans_mat,
    calc_obj_target_pose,
)


class PickPlaceVis:
    def __init__(
        self,
        exp_cfg_path,
        num_envs,
        device,
        max_tries=100,
        try_stable=False,
        pose_ids=None,
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_tries = max_tries
        self.try_stable = try_stable

        self.exp_cfg = self.read_exp_cfg(exp_cfg_path)
        self.read_stable_poses(pose_ids)
        self.initialize_workspace_cfg()
        self.initialize_sim()
        self.initialize_scene()
        self.set_objects()

    def vis(self, stable_obj_pose, stable_base_pose):
        base = self.scene["base"]
        base_pose_b = base.data.root_state_w[:, :7]
        base_pose_b[:, :3] -= self.env_origins

        obj_target_pose = calc_obj_target_pose(stable_obj_pose, stable_base_pose, base_pose_b)
        obj_target_pose[:, :3] += self.env_origins
        print(obj_target_pose)
        ang = torch.arccos(matrix_from_quat(obj_target_pose[:, 3:])[:, 2, 2])
        ang *= 180 / torch.pi
        print(torch.sort(ang))

        obj = self.scene["obj"]
        for _ in range(1000):
            obj.write_root_pose_to_sim(obj_target_pose)
            obj.write_root_velocity_to_sim(torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device))

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())


    def execute(self):
        pair_id = torch.arange(self.stable_obj_poses.shape[0], device=self.device)

        if pair_id.shape[0] > self.max_tries:
            pair_id = pair_id[torch.randperm(pair_id.shape[0])[:self.max_tries]]

        batch_exps = (pair_id.shape[0] + self.num_envs - 1) // self.num_envs
        for i in range(batch_exps):
            print('Starting batch experiment %d / %d' % (i+1, batch_exps))
            st_i = i * self.num_envs
            ed_i = st_i + self.num_envs
            batch_pair_id = pair_id[st_i:ed_i]
            while batch_pair_id.shape[0] < self.num_envs:
                ext_pair_id = pair_id[:self.num_envs - batch_pair_id.shape[0]]
                batch_pair_id = torch.cat((batch_pair_id, ext_pair_id))

            print('Pair id:', batch_pair_id)
            batch_stable_pose_id = batch_pair_id
            self.vis(
                self.stable_obj_poses[batch_stable_pose_id],
                self.stable_base_poses[batch_stable_pose_id],
            )

    @staticmethod
    def read_exp_cfg(exp_cfg_path):
        with open(exp_cfg_path, "r") as f:
            exp_cfg = json.load(f)

        root_path = os.path.abspath(os.path.dirname(exp_cfg_path))
        def traverse(exp_cfg):
            nonlocal root_path
            for key, value in exp_cfg.items():
                if "path" in key and not os.path.isabs(value):
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

            print('rel_trans_mat:', rel_trans_mat)
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
        self.place_axis = self.exp_cfg["obj"]["place_axis"]

    def initialize_workspace_cfg(self):
        self.workspace_cfg = WorkspaceConfiguration(
            self.exp_cfg["base"]["pose"],
            self.exp_cfg["base"]["usd_path"],
            self.exp_cfg["obj"]["pose"],
            self.exp_cfg["obj"]["usd_path"],
        )
        # self.workspace_cfg.obj.spawn.rigid_props.rigid_body_enabled = False
        # self.workspace_cfg.obj.spawn.rigid_props.kinematic_enabled = True

    def initialize_sim(self):
        self.sim_cfg = sim_utils.SimulationCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=10,
                dynamic_friction=10,
                friction_combine_mode="min",
            ),
        )
        self.sim = sim_utils.SimulationContext(self.sim_cfg)
        self.sim.set_camera_view([0, 2.5, 2.5], [0.0, 0.0, 0.2])

    def initialize_scene(self):
        self.scene_cfg = GraspingSceneCfg(
            num_envs=self.num_envs,
            env_spacing=2.0,
        )
        self.scene_cfg.load_objs_from_workspace_cfg(self.workspace_cfg)
        self.scene = InteractiveScene(self.scene_cfg)
        self.env_origins = self.scene.env_origins
        self.sim.reset()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

    def set_objects(self):
        # Set object default pose
        self.obj = self.scene["obj"]
        self.obj.default_root_state = torch.zeros((self.num_envs, 13), device=self.device)
        self.obj.default_root_state[:, 3] = 1
        for i in range(3):
            self.obj.default_root_state[:, i] = self.scene.env_origins[:, i] + self.workspace_cfg.obj_init_pos[i]
        # Write the pose into workspace_cfg for CuRobo world collision
        self.workspace_cfg.set_object_root_states(self.scene)


def main():
    exp_path = args_cli.exp
    exp_cfg_path = os.path.join(exp_path, "exp_config.json")

    vis = PickPlaceVis(
        exp_cfg_path,
        args_cli.num_envs,
        device=args_cli.device,
    )
    vis.execute()


if __name__ == "__main__":
    main()
    simulation_app.close()
