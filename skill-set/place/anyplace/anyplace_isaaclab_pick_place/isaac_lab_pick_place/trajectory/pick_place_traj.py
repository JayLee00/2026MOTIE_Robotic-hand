from typing import Sequence
import torch
from omni.isaac.lab.utils.math import matrix_from_quat
import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom

from .state_machine import PickPlaceState, PickPlaceStateMachine

class PickPlaceTraj:
    def __init__(
        self,
        robot,
        motion_planner,
        num_envs=1,
        pick_approach_dis=0.1,
        place_approach_dis=0.2,
        tsl_dist_thresh=0.005,
        orn_dist_thresh=0.1,
        vel_thresh=0.01,
        device="cuda:0",
    ):
        self._robot = robot
        self._motion_planner = motion_planner

        self._num_envs = num_envs
        self._device = device

        self._pick_approach_dis = pick_approach_dis
        self._place_approach_dis = place_approach_dis
        self._tsl_dist_thresh = tsl_dist_thresh
        self._orn_dist_thresh = orn_dist_thresh
        self._vel_thresh = vel_thresh

        self._target_pose = torch.zeros((4, self._num_envs, 7), dtype=torch.float32, device=self._device)
        self._target_pose[:, :, 0] = 0.4
        self._target_pose[:, :, 2] = 0.4
        self._target_pose[:, :, 4] = 1.

        self._cur_target_pose = self._target_pose[0].clone()

        self._sm = PickPlaceStateMachine(
            default_joint_pos=self._robot.arm_default_joint_pos[0],
            num_envs=self._num_envs,
            device=self._device,
        )

    def reset(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)
        self._sm.reset(env_ids)
        self.plan_success = self._motion_planner.prepare(self._target_pose[:, env_ids, :], env_ids)
        self._cur_target_pose[env_ids] = self._target_pose[0, env_ids]

    def set_target_b(self, pose1, pose2, place_approach_dir, env_ids: Sequence[int] = None):
        orig_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)

        orn1_mat = matrix_from_quat(pose1[..., 3:])

        pose1 = pose1.contiguous()
        pose2 = pose2.contiguous()
        self._target_pose[0, env_ids] = pose1.clone()
        self._target_pose[1, env_ids] = pose1.clone()
        self._target_pose[2, env_ids] = pose2.clone()
        self._target_pose[3, env_ids] = pose2.clone()

        self._target_pose[0, env_ids, :3] -= orn1_mat[:, :, 2] * self._pick_approach_dis
        self._target_pose[2, env_ids, :3] -= place_approach_dir * self._place_approach_dis

        self.reset(orig_env_ids)

    def step(self, dt: float):
        self._catch_state_change()

        actions = self._motion_planner.get_next_articulation_action()

        assert "position" in actions.keys()
        plan_pos = actions["position"]

        if "velocity" in actions.keys():
            plan_vel = actions["velocity"]
        else:
            plan_vel = torch.zeros((self._num_envs, 7), dtype=torch.float32, device=self._device)

        gripper_width = self._get_gripper_width()
        ee_reached = self._get_ee_reached()

        self._sm.compute_command(dt, plan_pos, plan_vel, gripper_width, ee_reached)

        self._robot.set_arm_joint_pos_target(self._sm.command_pos)
        self._robot.set_arm_joint_vel_target(self._sm.command_vel)

    @property
    def done(self):
        return self.done_with_complete_place | self.done_with_incomplete_place

    @property
    def done_with_complete_place(self):
        return self._sm.sm_state == PickPlaceState.DONE

    @property
    def done_with_incomplete_place(self):
        return self._sm.sm_state == PickPlaceState.DONE_INCOMPLETE_PLACE

    @property
    def failed(self):
        return self._sm.sm_state == PickPlaceState.FAILED

    @property
    def finished(self):
        return self.done | self.failed

    def _get_gripper_width(self):
        return torch.sum(self._robot.gripper_joint_pos, axis=1)

    def _get_ee_reached(self):
        tsl_err, orn_err = self._robot.get_pose_error_to_target_b(
            self._cur_target_pose[:, :3],
            self._cur_target_pose[:, 3:],
        )
        vel = torch.linalg.norm(self._robot.ee_vel_w[..., :3], dim=1)
        return (tsl_err < self._tsl_dist_thresh) & (orn_err < self._orn_dist_thresh) & (vel < self._vel_thresh)

    @staticmethod
    def _get_env_ids(flag):
        return flag.nonzero(as_tuple=True)[0]

    def get_fix_obj_env_ids(self):
        return self._get_env_ids(self._sm.sm_state <= PickPlaceState.CLOSE)

    def _catch_state_change(self):
        state_changed = self._sm.sm_state_changed
        if not state_changed.any():
            return

        state = self._sm.sm_state

        self._robot.open_gripper(self._get_env_ids(state_changed & (state == PickPlaceState.OPEN)))
        self._robot.open_gripper(self._get_env_ids(state_changed & (state == PickPlaceState.OPEN_PLACE)))
        self._robot.open_gripper(self._get_env_ids(state_changed & (state == PickPlaceState.OPEN_INCOMPLETE_PLACE)))
        self._robot.open_gripper(self._get_env_ids(state_changed & (state == PickPlaceState.FAILED)))
        self._robot.close_gripper(self._get_env_ids(state_changed & (state == PickPlaceState.CLOSE)))

        for i, state_id in enumerate([PickPlaceState.PREPICK, PickPlaceState.PICK, \
                                      PickPlaceState.PREPLACE, PickPlaceState.PLACE]):
            env_ids = self._get_env_ids(state_changed & (state == state_id))
            if not env_ids.shape[0]:
                continue
            self._cur_target_pose[env_ids] = self._target_pose[i, env_ids]
            self._motion_planner.set_target(self._cur_target_pose[env_ids], env_ids, flag=i)
