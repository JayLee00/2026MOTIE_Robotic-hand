from typing import Sequence
import torch

class MotionPlanner:
    def __init__(
        self,
        robot,
        workspace_config,
        num_envs=1,
        dt=1/60,
        device="cuda:0",
    ):
        self._robot = robot
        self._workspace_config = workspace_config
        self._num_envs = num_envs
        self._dt = dt
        self._device = device

    def prepare(self, target_pose: torch.Tensor, env_ids: Sequence[int] = None):
        raise NotImplementedError

    def set_target(self, target_pose: torch.Tensor, env_ids: Sequence[int] = None, flag = None):
        raise NotImplementedError

    def get_next_articulation_action(self):
        raise NotImplementedError
