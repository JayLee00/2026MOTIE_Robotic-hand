from typing import Sequence
import torch
import warp as wp

wp.init()

vec7 = wp.types.vector(length=7, dtype=float)


class PickPlaceState:
    num_states = 11
    OPEN = wp.constant(0)
    PREPICK = wp.constant(1)
    PICK = wp.constant(2)
    CLOSE = wp.constant(3)
    PREPLACE = wp.constant(4)
    PLACE = wp.constant(5)
    OPEN_PLACE = wp.constant(6)
    OPEN_INCOMPLETE_PLACE = wp.constant(7)
    DONE = wp.constant(8)
    DONE_INCOMPLETE_PLACE = wp.constant(9)
    FAILED = wp.constant(10)


class PickPlaceConstants:
    GRIPPER = wp.constant(2.)
    MOVE = wp.constant(12.)
    APPROACH = wp.constant(6.)
    ZERO_VEL_TARGET = vec7(0., 0., 0., 0., 0., 0., 0.)
    GRIPPER_GRASP_WIDTH_THRESH = wp.constant(0.005)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    plan_pos: wp.array(dtype=vec7),
    plan_vel: wp.array(dtype=vec7),
    command_pos: wp.array(dtype=vec7),
    command_vel: wp.array(dtype=vec7),
    gripper_width: wp.array(dtype=wp.float32),
    sm_state: wp.array(dtype=wp.int32),
    sm_wait_time: wp.array(dtype=wp.float32),
    sm_state_changed: wp.array(dtype=wp.bool),
    sm_ee_reached: wp.array(dtype=wp.bool),
):
    tid = wp.tid()

    state = sm_state[tid]
    if state == PickPlaceState.PREPLACE and gripper_width[tid] < PickPlaceConstants.GRIPPER_GRASP_WIDTH_THRESH:
        command_vel[tid] = PickPlaceConstants.ZERO_VEL_TARGET
        sm_state[tid] = PickPlaceState.FAILED
        sm_state_changed[tid] = True
        sm_wait_time[tid] = 0.
    elif state == PickPlaceState.OPEN or state == PickPlaceState.CLOSE or \
         state == PickPlaceState.OPEN_PLACE or state == PickPlaceState.OPEN_INCOMPLETE_PLACE:
        if sm_wait_time[tid] > PickPlaceConstants.GRIPPER:
            if state == PickPlaceState.OPEN:
                sm_state[tid] = PickPlaceState.PREPICK
            elif state == PickPlaceState.CLOSE:
                sm_state[tid] = PickPlaceState.PREPLACE
            elif state == PickPlaceState.OPEN_PLACE:
                sm_state[tid] = PickPlaceState.DONE
            else:
                sm_state[tid] = PickPlaceState.DONE_INCOMPLETE_PLACE
            sm_state_changed[tid] = True
            sm_wait_time[tid] = 0.
        else:
            sm_state_changed[tid] = False
            sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
    elif state == PickPlaceState.PREPICK or \
            state == PickPlaceState.PICK or \
            state == PickPlaceState.PREPLACE or \
            state == PickPlaceState.PLACE:
        if sm_ee_reached[tid]:
            command_vel[tid] = PickPlaceConstants.ZERO_VEL_TARGET

            if state == PickPlaceState.PREPICK:
                sm_state[tid] = PickPlaceState.PICK
            elif state == PickPlaceState.PICK:
                sm_state[tid] = PickPlaceState.CLOSE
            elif state == PickPlaceState.PREPLACE:
                sm_state[tid] = PickPlaceState.PLACE
            else:
                sm_state[tid] = PickPlaceState.OPEN_PLACE

            sm_state_changed[tid] = True
            sm_wait_time[tid] = 0.
        elif sm_wait_time[tid] > PickPlaceConstants.MOVE or \
            (sm_wait_time[tid] > PickPlaceConstants.APPROACH and \
                (state == PickPlaceState.PICK or state == PickPlaceState.PLACE)):
            if state == PickPlaceState.PLACE:
                sm_state[tid] = PickPlaceState.OPEN_INCOMPLETE_PLACE
            else:
                sm_state[tid] = PickPlaceState.FAILED
            sm_state_changed[tid] = True
            sm_wait_time[tid] = 0.
        else:
            command_pos[tid] = plan_pos[tid]
            command_vel[tid] = plan_vel[tid]

            sm_state_changed[tid] = False
            sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
    else:
        sm_state_changed[tid] = False


class PickPlaceStateMachine:
    def __init__(
        self,
        default_joint_pos: torch.Tensor,
        num_envs=1,
        device="cuda:0"
    ):
        self.num_envs = num_envs
        self.device = device

        default_joint_pos = default_joint_pos.flatten()
        if default_joint_pos.shape[0] > 7:
            default_joint_pos = default_joint_pos[:7]
        self.default_joint_pos = default_joint_pos.type(torch.float32)

        self.command_pos = self.default_joint_pos.unsqueeze(0).repeat(self.num_envs, 1).contiguous()
        self.command_vel = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.sm_state = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.sm_state_changed = torch.ones((num_envs,), dtype=bool, device=self.device)

        self.command_pos_wp = wp.from_torch(self.command_pos, vec7)
        self.command_vel_wp = wp.from_torch(self.command_vel, vec7)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.sm_state_changed_wp = wp.from_torch(self.sm_state_changed, wp.bool)

    def reset(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)
        self.command_pos[env_ids] = self.default_joint_pos
        self.command_vel[env_ids, :] = 0
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.
        self.sm_state_changed[env_ids] = True

    def compute_command(
        self,
        dt: float,
        plan_pos: torch.Tensor,
        plan_vel: torch.Tensor,
        gripper_width: torch.Tensor,
        ee_reached: torch.Tensor,
    ):
        dt_wp = wp.from_torch(torch.full((self.num_envs,), dt, device=self.device), wp.float32)
        plan_pos_wp = wp.from_torch(plan_pos.contiguous(), vec7)
        plan_vel_wp = wp.from_torch(plan_vel.contiguous(), vec7)
        gripper_width_wp = wp.from_torch(gripper_width.contiguous(), wp.float32)
        ee_reached_wp = wp.from_torch(ee_reached.contiguous(), wp.bool)

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                dt_wp,
                plan_pos_wp,
                plan_vel_wp,
                self.command_pos_wp,
                self.command_vel_wp,
                gripper_width_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.sm_state_changed_wp,
                ee_reached_wp,
            ],
            device=self.device,
        )
