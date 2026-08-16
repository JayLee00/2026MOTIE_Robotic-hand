import torch
from typing import Sequence
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from .motion_planner import MotionPlanner

class DiffIKMotionPlanner(MotionPlanner):
    def __init__(
        self,
        robot,
        workspace_config,
        num_envs=1,
        dt=1/60,
        ik_method="dls",
        device="cuda:0",
    ):
        super().__init__(
            robot,
            workspace_config,
            num_envs=num_envs,
            dt=dt,
            device=device,
        )

        self._diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=ik_method,
        )
        self._diff_ik_controller = DifferentialIKController(
            self._diff_ik_cfg,
            num_envs=self._num_envs,
            device=self._device,
        )

        self._ik_command = torch.zeros(
            (self._num_envs, 7),
            dtype=torch.float32,
            device=self._device
        )
        self._ik_command[:, 4] = 1

    def prepare(self, target_pose: torch.Tensor, env_ids: Sequence[int] = None):
        pass

    def set_target(self, target_pose: torch.Tensor, env_ids: Sequence[int] = None, flag = None):
        if env_ids is None:
            env_ids = slice(env_ids)

        target_pos, target_quat = target_pose[:, :3], target_pose[:, 3:]
        target_pos = self._robot.actual_to_model_ee_pos(target_pos, target_quat)
        self._ik_command[env_ids, :3] = target_pos
        self._ik_command[env_ids, 3:] = target_quat
        self._diff_ik_controller.set_command(self._ik_command)

    def get_next_articulation_action(self):
        joint_pos_des = self._diff_ik_controller.compute(
            *self._robot.model_ee_pose_b,
            self._robot.get_jacobian(),
            self._robot.arm_joint_pos,
        )
        return {
            "position": joint_pos_des,
        }