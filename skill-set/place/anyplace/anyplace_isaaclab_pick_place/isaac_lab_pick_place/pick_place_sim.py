import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import matrix_from_quat

from scene import GraspingSceneCfg
from retrieve_grasps import calc_ee_pose, calc_obj_target_pose
from robot import Robot
from motion_planning import CuRoboMotionPlanner
from trajectory import PickPlaceTraj
from vis_pose import PoseVisualization


class PickPlaceSimulation:
    SIM_DT = 1 / 240
    CUROBO_DT = 1 / 360
    SIM_FINISH_WAIT_STEPS = 150

    def __init__(
        self,
        workspace_cfg,
        num_envs,
        device,
        place_axis="-z",
        approach_dis=0.15,
    ):
        self.workspace_cfg = workspace_cfg
        self.num_envs = num_envs
        self.device = device
        self.place_axis = place_axis
        self.approach_dis = approach_dis

        self.initialize_sim()
        self.initialize_scene()
        self.set_objects()
        self.initialize_robot()
        self.initialize_grasping()

    def initialize_sim(self):
        self.sim_cfg = sim_utils.SimulationCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=10,
                dynamic_friction=10,
                friction_combine_mode="min",
            ),
            dt=self.SIM_DT,
        )
        self.sim = sim_utils.SimulationContext(self.sim_cfg)
        self.sim.set_camera_view([0, 2.5, 2.5], [0.0, 0.0, 0.2])

    def close(self):
        self.sim._has_gui = False  # For directly closing the entire app
        self.sim.stop()

    def initialize_scene(self):
        self.scene_cfg = GraspingSceneCfg(
            num_envs=self.num_envs,
            env_spacing=2.0,
        )
        self.scene_cfg.load_objs_from_workspace_cfg(self.workspace_cfg)
        self.scene = InteractiveScene(self.scene_cfg)
        self.env_origins = self.scene.env_origins
        self.sim.reset()
        self.scene.update(self.SIM_DT)

    def set_objects(self):
        # Set object default pose
        self.obj = self.scene["obj"]
        self.obj.default_root_state = torch.zeros((self.num_envs, 13), device=self.device)
        self.obj.default_root_state[:, 3] = 1
        for i in range(3):
            self.obj.default_root_state[:, i] = self.scene.env_origins[:, i] + self.workspace_cfg.obj_init_pos[i]
        # Write the pose into workspace_cfg for CuRobo world collision
        self.workspace_cfg.set_object_root_states(self.scene)

    def initialize_robot(self):
        self.robot = Robot(
            self.scene["robot"],
            self.scene,
            SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]),
            SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
        )

    def initialize_grasping(self):
        curobo_kwargs = {}
        # if self.place_axis[1] in "xyz":
        #     i = "xyz".find(self.place_axis[1])
        #     curobo_kwargs["preplace_xyz_fixed"] = list(i != j for j in range(3))

        self.motion_planner = CuRoboMotionPlanner(
            self.robot,
            self.workspace_cfg,
            num_envs=self.num_envs,
            device=self.device,
            dt=self.CUROBO_DT,
            **curobo_kwargs
        )
        self.traj = PickPlaceTraj(
            self.robot,
            self.motion_planner,
            num_envs=self.num_envs,
            place_approach_dis=self.approach_dis,
            device=self.device,
        )
        self.pose_vis = PoseVisualization(self.scene)

    def reset(self):
        self.scene.reset()
        self.step_sim()

    def run_pick_place(
        self,
        grasp_pose,
        stable_obj_pose,
        stable_base_pose,
    ):
        assert self.num_envs == grasp_pose.shape[0]
        assert self.num_envs == stable_obj_pose.shape[0]
        assert self.num_envs == stable_base_pose.shape[0]

        robot = self.scene["robot"]
        robot_init_pos = robot.data.default_joint_pos.clone()
        robot.set_joint_position_target(robot_init_pos)
        robot.write_joint_state_to_sim(
            robot_init_pos,
            robot.data.default_joint_vel,
        )

        for obj_name in self.workspace_cfg.obj_names:
            self.scene[obj_name].write_root_state_to_sim(self.workspace_cfg.obj_state[obj_name])
        self.step_sim()

        base = self.scene["base"]
        base_pose_b = base.data.root_state_w[:, :7]
        base_pose_b[:, :3] -= self.env_origins
        obj_pose_b = self.obj.data.root_state_w[..., :7]
        obj_pose_b[:, :3] -= self.env_origins

        obj_target_pose = calc_obj_target_pose(stable_obj_pose, stable_base_pose, base_pose_b)
        pose1 = calc_ee_pose(obj_pose_b, grasp_pose, device=self.device)
        pose2 = calc_ee_pose(obj_target_pose, grasp_pose, device=self.device)

        obj_target_mat = matrix_from_quat(obj_target_pose[:, 3:])
        obj_place_dir = self.get_place_direction(obj_target_mat, self.place_axis)
        self.traj.set_target_b(pose1.clone(), pose2.clone(), obj_place_dir)
        self.pose_vis.vis(self.traj._target_pose[1], self.traj._target_pose[2], self.traj._target_pose[3])

        while not self.traj.finished.all().item():
            self.traj.step(self.SIM_DT)
            self.step_sim(self.traj.get_fix_obj_env_ids())

        exec_res = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        exec_res[self.traj.done_with_incomplete_place] = 2
        exec_res[self.traj.done_with_complete_place] = 3
        print('Plan success:', self.traj.plan_success)
        exec_res[self.traj.plan_success == False] = 0
        print('exec_res:', exec_res)

        obj_pose = self.obj.data.root_state_w.clone()[:, :7]
        obj_pose[:, :3] -= self.env_origins

        for _ in range(self.SIM_FINISH_WAIT_STEPS):
            self.sim.step()
        self.sim.step()
        self.pose_vis.del_vis()

        return exec_res, obj_pose

    def step_sim(self, set_obj_env_ids=None):
        tensor_env_ids = set_obj_env_ids
        if tensor_env_ids is None:
            tensor_env_ids = slice(None)

        gravity = self.obj.data.projected_gravity_b[tensor_env_ids].unsqueeze(1)
        torque = torch.zeros(gravity.shape, device=self.device)
        self.obj.set_external_force_and_torque(-gravity, torque, env_ids=set_obj_env_ids)
        self.obj.write_root_velocity_to_sim(torch.zeros((gravity.shape[0], 6), device=self.device), set_obj_env_ids)

        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.SIM_DT)

    @staticmethod
    def get_place_direction(obj_pose_mat, place_axis):
        d = place_axis[1]
        if d in "xyz":  # Extrinsic
            # vec = obj_pose_mat[:, :, "xyz".find(d)]
            vec = torch.eye(3, device=obj_pose_mat.device)[:, "xyz".find(d)]
            vec = vec.unsqueeze(0).repeat(obj_pose_mat.shape[0], 1)
        else:
            vec = obj_pose_mat[:, :, "XYZ".find(d)]
        return vec * (1 if place_axis[0] == "+" else -1)
