import os
import json
import argparse
import trimesh
import numpy as np

from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher({ "headless": True })
simulation_app = app_launcher.app


from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema


parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, required=True,
                    help="path to the experiment folder that contains exp_config.json")


def get_mesh_from_usd(stage):
    list_vertices, list_faces = [], []

    cnt_vertices = 0
    for prim in stage.Traverse():
        if not UsdGeom.Mesh(prim):
            continue

        xform_prim_path = os.path.dirname(str(prim.GetPath()))
        print('Found prim:', xform_prim_path)
        xform_prim = UsdGeom.Xform.Get(stage, xform_prim_path)
        trans_to_world_mat = xform_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        trans_to_world_mat = np.array(trans_to_world_mat).T

        usd_mesh = UsdGeom.Mesh(prim)
        pts = np.array(usd_mesh.GetPointsAttr().Get())
        faces = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())
        faces = faces.reshape(-1, 3) + cnt_vertices
        print(cnt_vertices)

        pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
        # pts = (trans_to_world_mat @ pts.T).T
        pts = pts @ trans_to_world_mat.T
        pts = pts[:, :3]

        list_vertices.append(pts)
        list_faces.append(faces)
        cnt_vertices += pts.shape[0]

    mesh = trimesh.Trimesh(
        vertices=np.concatenate(list_vertices),
        faces=np.concatenate(list_faces),
    )
    print('Extent:', mesh.bounds[1] - mesh.bounds[0])

    return mesh


def create_rigid_body(stage):
    for prim in stage.Traverse():
        if not UsdGeom.Mesh(prim):
            continue

        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr().Set(True)
        collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collision.CreateApproximationAttr().Set("sdf")
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)

    root = stage.GetDefaultPrim()
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

        
def create_mass(stage, mass=0.1):
    root = stage.GetDefaultPrim()

    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr().Set(mass)


def create_physics_material(stage, friction=0.5):
    material_path = "/root/PhysicsMaterial"
    material_prim = stage.DefinePrim(material_path, "Material")

    material = UsdPhysics.MaterialAPI.Apply(material_prim)
    material.CreateStaticFrictionAttr().Set(friction)
    material.CreateDynamicFrictionAttr().Set(friction)

    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    physx_material.CreateFrictionCombineModeAttr().Set("min")

    for prim in stage.Traverse():
        if not UsdGeom.Mesh(prim):
            continue

        material_binding = UsdShade.MaterialBindingAPI.Apply(prim)
        material_binding.Bind(UsdShade.Material(material_prim), materialPurpose="physics")


def read_exp_cfg(exp_cfg_path):
    dir_path = os.path.abspath(os.path.dirname(exp_cfg_path))

    def traverse(cfg):
        for key, value in cfg.items():
            if "path" in key and not os.path.isabs(value):
                cfg[key] = os.path.join(dir_path, cfg[key])
            elif isinstance(value, dict):
                cfg[key] = traverse(value)
        return cfg

    with open(exp_cfg_path, "r") as f:
        exp_cfg = json.load(f)
    return traverse(exp_cfg)

        
def main(exp_cfg_path):
    exp_cfg = read_exp_cfg(exp_cfg_path)

    base = exp_cfg["base"]
    print("Process", base["name"])
    stage_base = Usd.Stage.Open(base["usd_path"])

    stl_path = os.path.splitext(base["usd_path"])[0] + ".stl"
    print("Read mesh and export to", stl_path)
    mesh = get_mesh_from_usd(stage_base)
    mesh.export(stl_path)

    print("Apply Rigid Body API with Mesh Simplification Colliders")
    create_rigid_body(stage_base)

    print("Create Physics Material with Friction 0.5 and bind them")
    create_physics_material(stage_base, friction=0.5)

    print("Save to", base["usd_path"])
    stage_base.Export(base["usd_path"])

    print()

    obj = exp_cfg["obj"]
    print("Process", obj["name"])
    stage_obj = Usd.Stage.Open(obj["usd_path"])

    stl_path = os.path.splitext(obj["usd_path"])[0] + ".stl"
    print("Read mesh and export to", stl_path)
    mesh = get_mesh_from_usd(stage_obj)
    mesh.export(stl_path)

    print("Apply Rigid Body API with Mesh Simplification Colliders")
    create_rigid_body(stage_obj)

    print("Create Mass API with mass 0.1")
    create_mass(stage_obj, mass=0.1)

    print("Save to", obj["usd_path"])
    stage_obj.Export(obj["usd_path"])

    print()


if __name__ == '__main__':
    args = parser.parse_args()
    main(os.path.join(args.exp, "exp_config.json"))
    simulation_app.close()
