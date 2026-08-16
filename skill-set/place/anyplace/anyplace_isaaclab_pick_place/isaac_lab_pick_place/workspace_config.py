import os

from omni.isaac.lab.assets import RigidObjectCfg
import omni.isaac.lab.sim as sim_utils


class WorkspaceConfiguration:
    data_path = os.path.join(os.path.dirname(__file__), "data")

    obj_names = ["obj", "base"]
    cam_names = []

    cfg_names = obj_names + cam_names

    def __init__(self, base_pose, base_usd, obj_pose, obj_usd):
        self.base_pos, self.base_orn = base_pose[:3], base_pose[3:]
        self.base_usd = base_usd
        self.base = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/base",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.base_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self.base_pos,
                rot=self.base_orn
            ),
        )

        self.obj_init_pos, self.obj_init_orn = obj_pose[:3], obj_pose[3:]
        self.obj_usd = obj_usd
        self.obj = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/obj",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.obj_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self.obj_init_pos,
                rot=self.obj_init_orn
            ),
        )

    def set_object_root_states(self, scene):
        self.obj_state = {}
        self.env_origins = scene.env_origins
        for obj_name in self.obj_names:
            self.obj_state[obj_name] = scene[obj_name].data.root_state_w.clone()

    def curobo_config(self, with_obj=False):
        def get_pose(obj_name):
            if hasattr(self, "obj_state") and obj_name in self.obj_state:
                state = self.obj_state[obj_name].clone()
                state[:, :3] -= self.env_origins
                pose = state[0, :7].cpu().numpy().tolist()
                return tuple(pose)
            else:
                cfg = getattr(self, obj_name)
                return cfg.init_state.pos + cfg.init_state.rot

        def get_stl_path(usd_path):
            stl_path = os.path.splitext(usd_path)[0] + ".stl"
            assert os.path.exists(stl_path), "STL file %s doesn't exist" % stl_path
            return stl_path

        cfg = {
            "mesh": {
                "base": {
                    "file_path": get_stl_path(self.base_usd),
                    "pose": get_pose("base"),
                }
            }
        }
        if with_obj:
            cfg["mesh"]["obj"] = {
                "file_path": get_stl_path(self.obj_usd),
                "pose": get_pose("obj"),
            }
        return cfg
