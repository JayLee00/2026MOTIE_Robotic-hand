import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric

from .motion_planner import MotionPlanner, Sequence


class CuRoboMotionPlanner(MotionPlanner):
    def __init__(
        self,
        robot,
        workspace_config,
        num_envs=1,
        dt=1/60,
        interpolation_steps=2000,
        max_attempts=10,
        preplace_xyz_fixed=[0, 0, 0],
        device="cuda:0",
    ):
        super().__init__(
            robot,
            workspace_config,
            num_envs=num_envs,
            dt=dt,
            device=device,
        )

        self._interpolation_steps = interpolation_steps
        self._preplace_xyz_fixed = preplace_xyz_fixed

        self._plan_lengths = torch.zeros((4, self._num_envs), dtype=torch.int32, device=self._device)
        self._plan_success = torch.zeros((4, self._num_envs), dtype=torch.bool, device=self._device)
        self._plan_target = torch.zeros((4, self._num_envs, 7), dtype=torch.float32, device=self._device)
        plan_dof_shape = (self._num_envs, self._interpolation_steps, 7)
        self._cur_plan_pos = torch.zeros(plan_dof_shape, dtype=torch.float32, device=self._device)
        self._cur_plan_vel = torch.zeros(plan_dof_shape, dtype=torch.float32, device=self._device)
        self._cur_step = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        all_dof_shape = (4, self._num_envs, self._interpolation_steps, 7)
        self._plan_pos = torch.zeros(all_dof_shape, dtype=torch.float32, device=self._device)
        self._plan_vel = torch.zeros(all_dof_shape, dtype=torch.float32, device=self._device)

        collision_cfg = {
            "cuboid": {
                "ground_plane": {
                    "dims": [2, 2, 0.02],
                    "pose": [0, 0, -0.01, 1, 0, 0, 0],
                }
            }
        }
        motion_gen_args = {
            "robot_cfg": "franka.yml",
            "tensor_args": TensorDeviceType(device=self._device),
            "collision_checker_type": CollisionCheckerType.MESH,
            "interpolation_dt": self._dt,
            "use_cuda_graph": False,
            "num_trajopt_seeds": 12,
            "num_graph_seeds": 1,
            "num_ik_seeds": 30,
            "interpolation_steps": self._interpolation_steps,
        }

        gen_cfg_no_base = MotionGenConfig.load_from_robot_config(
            world_model=collision_cfg,
            **motion_gen_args
        )

        collision_with_base_cfg = self._workspace_config.curobo_config()
        collision_with_obj_cfg = self._workspace_config.curobo_config(with_obj=True)

        collision_with_base_cfg["cuboid"] = {
            "ground_plane": collision_cfg["cuboid"]["ground_plane"]
        }
        collision_with_obj_cfg["cuboid"] = {
            "ground_plane": collision_cfg["cuboid"]["ground_plane"]
        }

        gen_with_base_cfg = MotionGenConfig.load_from_robot_config(
            world_model=collision_with_base_cfg, **motion_gen_args
        )
        gen_with_obj_cfg = MotionGenConfig.load_from_robot_config(
            world_model=collision_with_obj_cfg, **motion_gen_args
        )

        self._list_motion_gen = [MotionGen(gen_with_obj_cfg)]
        self._list_motion_gen.append(MotionGen(gen_with_base_cfg))
        self._list_motion_gen.append(MotionGen(gen_with_obj_cfg))
        self._list_motion_gen.append(MotionGen(gen_cfg_no_base))

        motion_gen_plan_config_kwargs = {
            "enable_graph": False,
            "enable_graph_attempt": 1,
            "enable_opt": True,
            "parallel_finetune": True,
        }

        self._plan_cfg_move = MotionGenPlanConfig(**motion_gen_plan_config_kwargs)

        self._plan_cfg_prepick = MotionGenPlanConfig(
            pose_cost_metric=PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=torch.tensor([1., 1, 1, 1, 1, 0], device=self._device),
            ),
            **motion_gen_plan_config_kwargs
        )
        self._plan_cfg_preplace = MotionGenPlanConfig(
            pose_cost_metric=PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=torch.tensor([1., 1, 1] + self._preplace_xyz_fixed, device=self._device),
            ),
            **motion_gen_plan_config_kwargs
        )
        self._list_plan_cfg = [
            self._plan_cfg_move,
            self._plan_cfg_prepick,
            self._plan_cfg_move,
            self._plan_cfg_preplace,
        ]

    def prepare(self, targets: torch.Tensor, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)

        targets = targets.clone()
        targets[..., :3] = self._robot.actual_to_model_ee_pos(
            targets[..., :3], targets[..., 3:]
        )
        self._plan_target[:, env_ids, :] = targets.clone()

        self._plan_success[:, env_ids].zero_()
        self._plan_lengths[:, env_ids].zero_()
        self._plan_pos[:, env_ids].zero_()
        self._plan_vel[:, env_ids].zero_()

        ok_env_ids = torch.arange(0, self._plan_success.shape[1], device=self._device)[env_ids]

        def save_result(i: int, result):
            nonlocal ok_env_ids
            if result is not None:
                self._plan_success[i, ok_env_ids] = result.success

                lengths = result.path_buffer_last_tstep
                if lengths is not None:
                    self._plan_lengths[i, ok_env_ids] = torch.tensor(lengths, device=self._device)
                    self._plan_pos[i, ok_env_ids] = result.interpolated_plan.position
                    self._plan_vel[i, ok_env_ids] = result.interpolated_plan.velocity
                
                ok_env_ids = ok_env_ids[result.success]

            print('CuRobo plan success on trajectory phase %d:' % i, ok_env_ids.cpu())

            return JointState.from_position(
                self._plan_pos[i, ok_env_ids, -1].contiguous(),
                joint_names=self._robot.arm_joint_names
            )

        state = JointState.from_position(
            self._robot.arm_default_joint_pos.contiguous(),
            joint_names=self._robot.arm_joint_names
        )

        for i, (motion_gen, plan_cfg) in enumerate(zip(self._list_motion_gen, self._list_plan_cfg)):
            target = self._plan_target[i, ok_env_ids]
            goal_pose = Pose(position=target[:, :3], quaternion=target[:, 3:])

            # Attach object for approaching pre_place
            detach = False
            if i == 2 and state.shape[0]:
                detach = True
                motion_gen.attach_objects_to_robot(state, ["obj"])

            if state.shape[0]:
                result = motion_gen.plan_batch(state, goal_pose, plan_cfg)
            else:
                result = None
            state = save_result(i, result)

            if detach:
                motion_gen.detach_object_from_robot()

        return self._plan_success[-1, env_ids]

    def set_target(self, target_pose: torch.Tensor, env_ids: Sequence[int] = None, flag: int = None):
        assert flag is not None
        if env_ids is None:
            env_ids = slice(None)

        self._cur_plan_pos[env_ids] = self._plan_pos[flag, env_ids]
        self._cur_plan_vel[env_ids] = self._plan_vel[flag, env_ids]
        self._cur_step[env_ids] = 0

    def get_next_articulation_action(self):
        action = {
            "position": self._cur_plan_pos[torch.arange(self._num_envs), self._cur_step].contiguous(),
            # "velocity": self._cur_plan_vel[torch.arange(self._num_envs), self._cur_step].contiguous(),
        }

        self._cur_step += self._cur_step < self._interpolation_steps - 1

        return action
