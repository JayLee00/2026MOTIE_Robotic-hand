from dataclasses import InitVar

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG

from workspace_config import WorkspaceConfiguration


@configclass
class GraspingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def load_objs_from_workspace_cfg(self, workspace_cfg: WorkspaceConfiguration):
        for cfg_name in workspace_cfg.cfg_names:
            setattr(self, cfg_name, getattr(workspace_cfg, cfg_name))
