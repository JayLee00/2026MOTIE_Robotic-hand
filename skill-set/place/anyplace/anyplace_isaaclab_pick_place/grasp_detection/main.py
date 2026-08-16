import os
import argparse
import json
import numpy as np
from pxr import Usd, UsdGeom

from get_grasp import GetGraspByAnyGrasp


def parse_exp_cfg(exp_cfg_path):
    exp_root_path = os.path.dirname(exp_cfg_path)
    with open(exp_cfg_path, "r") as f:
        exp_cfg = json.load(f)

    def traverse(cfg):
        for key, value in cfg.items():
            if "path" in key and not os.path.isabs(value):
                cfg[key] = os.path.join(exp_root_path, value)
            elif isinstance(value, dict):
                cfg[key] = traverse(value)
        return cfg

    return traverse(exp_cfg)


def main(args, exp_cfg):
    usd_path = exp_cfg['obj']['usd_path']
    stage = Usd.Stage.Open(usd_path)
    geom_prims = [prim for prim in stage.Traverse() if UsdGeom.Mesh(prim)]

    print(geom_prims)
    assert len(geom_prims) == 1

    mesh_prim = geom_prims[0]

    trans_to_world_mat = UsdGeom.Mesh(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    trans_to_world_mat = np.array(trans_to_world_mat).T

    scale = exp_cfg['grasps']['scale'] if 'scale' in exp_cfg['grasps'] else 2

    get_anygrasp = GetGraspByAnyGrasp(args)
    grasps = get_anygrasp.get_grasp(UsdGeom.Mesh(mesh_prim), trans_to_world_mat, scale)

    grasps_dict = [dict(grasp) for grasp in grasps]
    grasps_json = json.dumps(grasps_dict)
    print(grasps_json)

    with open(exp_cfg['grasps']['path'], 'w') as f:
        f.write(grasps_json)

    # get_anygrasp.visualize()  # Comment this line if you don't want to have visualization


if __name__ == '__main__':
    dir_path = os.path.dirname(__file__)
    default_checkpoint_path = os.path.join(dir_path, 'checkpoint-rs.tar')

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True,
                        help='path to the folder containing exp_config.json')

    # AnyGrasp arguments
    parser.add_argument('--checkpoint_path', type=str, default=default_checkpoint_path,
                        help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=100000,
                        help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300,
                        help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01,
                        help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                        help='Voxel Size to process point clouds before collision detection [default: 0.01]')

    args = parser.parse_args()
    exp_cfg_path = os.path.join(args.exp, 'exp_config.json')

    exp_cfg = parse_exp_cfg(exp_cfg_path)
    args.max_grasps = exp_cfg['grasps']['max_num']

    main(args, exp_cfg)
