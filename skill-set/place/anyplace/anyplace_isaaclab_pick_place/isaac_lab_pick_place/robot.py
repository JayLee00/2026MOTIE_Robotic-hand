import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils.math import matrix_from_quat, subtract_frame_transforms, compute_pose_error


class Robot:
    def __init__(
        self,
        articulation: Articulation,
        scene: InteractiveScene,
        arm_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
        delta_z=0.104,
    ):
        self._robot = articulation
        self._scene = scene
        self._arm_cfg = arm_cfg
        self._gripper_cfg = gripper_cfg

        self._delta_z = delta_z

        self._prepare_robot_cfg()

    def _prepare_robot_cfg(self):
        self._arm_cfg.resolve(self._scene)
        self._gripper_cfg.resolve(self._scene)

        self.ee_id = self._arm_cfg.body_ids[0]
        self.ee_jacobi_id = self.ee_id - int(self._robot.is_fixed_base)
        self.arm_joint_ids = self._arm_cfg.joint_ids
        self.gripper_joint_ids = self._gripper_cfg.joint_ids

    def reset(self, **kwargs):
        self._robot.reset(**kwargs)

    @property
    def data(self):
        return self._robot.data

    def get_jacobian(self):
        return self._robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_id, :, self.arm_joint_ids
        ]

    @property
    def root_pose_w(self):
        pose = self.data.root_state_w[:, :7]
        return pose[:, :3], pose[:, 3:7]

    def pose_w_to_b(self, pos_w, quat_w):
        return subtract_frame_transforms(
            *self.root_pose_w, pos_w, quat_w
        )

    @property
    def model_ee_pose_w(self):
        pose = self.data.body_state_w[:, self.ee_id, :7]
        return pose[:, :3], pose[:, 3:7]

    @property
    def model_ee_pose_b(self):
        return self.pose_w_to_b(*self.model_ee_pose_w)

    @property
    def ee_vel_w(self):
        return self.data.body_vel_w[:, self.ee_id]

    def model_to_actual_ee_pos(self, model_ee_pos, model_ee_quat):
        orn_mat = matrix_from_quat(model_ee_quat)
        return model_ee_pos + orn_mat[..., 2] * self._delta_z

    def model_to_actual_ee_pose(self, model_ee_pos, model_ee_quat):
        return self.model_to_actual_ee_pos(model_ee_pos, model_ee_quat), model_ee_quat

    def actual_to_model_ee_pos(self, actual_ee_pos, actual_ee_quat):
        orn_mat = matrix_from_quat(actual_ee_quat)
        return actual_ee_pos - orn_mat[..., 2] * self._delta_z

    def actual_to_model_ee_pose(self, actual_ee_pos, actual_ee_quat):
        return self.actual_to_model_ee_pos(actual_ee_pos, actual_ee_quat), actual_ee_quat

    @property
    def actual_ee_pose_w(self):
        return self.model_to_actual_ee_pose(*self.model_ee_pose_w)

    @property
    def actual_ee_pose_b(self):
        return self.model_to_actual_ee_pose(*self.model_ee_pose_b)

    @property
    def arm_joint_names(self):
        return [self.data.joint_names[i] for i in self.arm_joint_ids]

    @property
    def arm_default_joint_pos(self):
        return self.data.default_joint_pos[:, self.arm_joint_ids]

    @property
    def arm_joint_pos(self):
        return self.data.joint_pos[:, self.arm_joint_ids]

    def set_joint_pos_target(self, target, joint_ids=None, env_ids=None):
        self._robot.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)

    def set_joint_vel_target(self, target, joint_ids=None, env_ids=None):
        self._robot.set_joint_velocity_target(target, joint_ids=joint_ids, env_ids=env_ids)

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        return self._robot.set_joint_effort_target(target, joint_ids=joint_ids, env_ids=env_ids)

    def set_arm_joint_pos_target(self, target, env_ids=None):
        return self.set_joint_pos_target(target, joint_ids=self.arm_joint_ids, env_ids=env_ids)

    def set_arm_joint_vel_target(self, target, env_ids=None):
        return self.set_joint_vel_target(target, joint_ids=self.arm_joint_ids, env_ids=env_ids)

    def set_arm_joint_effort_target(self, target, env_ids=None):
        return self.set_joint_effort_target(target, joint_ids=self.arm_joint_ids, env_ids=env_ids)

    @property
    def gripper_joint_pos(self):
        return self.data.joint_pos[:, self.gripper_joint_ids]

    def set_gripper_joint_pos_target(self, target, env_ids=None):
        return self.set_joint_pos_target(target, joint_ids=self.gripper_joint_ids, env_ids=env_ids)

    def set_gripper_joint_vel_target(self, target, env_ids=None):
        return self.set_joint_vel_target(target, joint_ids=self.gripper_joint_ids, env_ids=env_ids)

    def set_gripper_joint_effort_target(self, target, env_ids=None):
        return self.set_joint_effort_target(target, joint_ids=self.gripper_joint_ids, env_ids=env_ids)

    def open_gripper(self, env_ids=None):
        joint_limits = self.data.default_joint_limits
        target = joint_limits[:, self.gripper_joint_ids, 1]
        if env_ids is not None:
            target = target[env_ids]
        return self.set_gripper_joint_pos_target(target, env_ids=env_ids)

    def close_gripper(self, env_ids=None):
        joint_limits = self.data.default_joint_limits
        target = joint_limits[:, self.gripper_joint_ids, 0]
        if env_ids is not None:
            target = target[env_ids]
        return self.set_gripper_joint_pos_target(target, env_ids=env_ids)

    def get_pose_error_to_target_w(self, target_pos_w, target_quat_w):
        tsl_error, orn_error = compute_pose_error(
            *self.actual_ee_pose_w,
            target_pos_w, target_quat_w,
            rot_error_type="axis_angle",
        )
        tsl_error = torch.linalg.norm(tsl_error, dim=1)
        orn_error = torch.linalg.norm(orn_error, dim=1)
        return tsl_error, orn_error

    def get_pose_error_to_target_b(self, target_pos_b, target_quat_b):
        tsl_error, orn_error = compute_pose_error(
            *self.actual_ee_pose_b,
            target_pos_b, target_quat_b,
            rot_error_type="axis_angle"
        )
        # print('Target:', target_pos_b, target_quat_b)
        # print('Actual:', *self.actual_ee_pose_b)
        # print(tsl_error, orn_error)
        tsl_error = torch.linalg.norm(tsl_error, dim=1)
        orn_error = torch.linalg.norm(orn_error, dim=1)
        # print(tsl_error, orn_error)
        return tsl_error, orn_error