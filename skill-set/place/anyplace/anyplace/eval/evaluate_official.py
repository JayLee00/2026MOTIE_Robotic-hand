import os, os.path as osp
import random
import numpy as np
import time
import signal
import torch
import argparse
import threading
from scipy.spatial.transform import Rotation as R
import open3d as o3d

import pybullet as p
import meshcat

from airobot import set_log_level
from airobot.utils.pb_util import create_pybullet_client
from anyplace.utils import util, config_util
from anyplace.utils import path_util
from anyplace.utils.pb2mc.pybullet_meshcat import PyBulletMeshcat
from anyplace.utils.mesh_util import three_util
from anyplace.utils.anyplace.multistep_pose_regression_anyplace import policy_inference_methods_dict
from anyplace.utils.relational_policy.multistep_pose_regression_nsm import nsm_policy_inference_methods_dict
from anyplace.model.transformer.policy import (
    NSMTransformerSingleTransformationRegression, 
    NSMTransformerImplicit
)



def pb2mc_update(
        recorder: PyBulletMeshcat, 
        mc_vis: meshcat.Visualizer, 
        stop_event: threading.Event, 
        run_event: threading.Event) -> None:
    iters = 0
    # while True:
    while not stop_event.is_set():
        run_event.wait()
        iters += 1
        try:
            recorder.add_keyframe()
            recorder.update_meshcat_current_state(mc_vis)
        except KeyError as e:
            print(f'PyBullet to Meshcat thread Exception: {e}')
            time.sleep(0.1)
        time.sleep(1/230.0)


def main(args: config_util.AttrDict) -> None:
    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    signal.signal(signal.SIGINT, util.signal_handler)
    
    if args.new_meshcat:
        mc_vis = meshcat.Visualizer()
    else:
        zmq_url = f'tcp://127.0.0.1:{args.port_vis}'
        mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis['scene'].delete()

    pb_client = create_pybullet_client(
        gui=args.experiment.pybullet_viz, 
        opengl_render=True, 
        realtime=True, 
        server=args.experiment.pybullet_server)
    recorder = PyBulletMeshcat(pb_client=pb_client)
    recorder.clear()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    infer_kwargs = {}
    #####################################################################################
    # Set up the models (feat encoder, voxel affordance, pose refinement, success)

    model_ckpt_logdir = osp.join(path_util.get_anyplace_model_weights(), args.experiment.logdir) 
    reso_grid = args.data.voxel_grid.reso_grid  # args.reso_grid
    padding_grid = args.data.voxel_grid.padding
    raster_pts = three_util.get_raster_points(reso_grid, padding=padding_grid)
    raster_pts = raster_pts.reshape(reso_grid, reso_grid, reso_grid, 3)
    raster_pts = raster_pts.transpose(2, 1, 0, 3)
    raster_pts = raster_pts.reshape(-1, 3)
    rot_grid_samples = args.data.rot_grid_samples
    rot_grid = util.generate_healpix_grid(size=rot_grid_samples) 
    args.data.rot_grid_bins = rot_grid.shape[0]
    exp_args = args.experiment

    # Load pose refinement model, loss, and optimizer
    pose_refine_model_path = None
    pr_model = None
    if exp_args.load_pose_regression:
        # assumes model path exists
        pose_refine_model_path = args.experiment.eval.ckpt_path
        print(f'!!!!!!!!!!!!! Model path: {pose_refine_model_path}')
        pose_refine_ckpt = torch.load(pose_refine_model_path, map_location=torch.device('cpu'))

        # update config with config used during training
        config_util.update_recursive(args.model.refine_pose, config_util.recursive_attr_dict(pose_refine_ckpt['args']['model']['refine_pose']))

        # model (pr = pose refine)
        pr_type = args.model.refine_pose.type
        pr_args = config_util.copy_attr_dict(args.model[pr_type])
        if args.model.refine_pose.get('model_kwargs') is not None:
            custom_pr_args = args.model.refine_pose.model_kwargs[pr_type]
            config_util.update_recursive(pr_args, custom_pr_args)

        if pr_type == 'nsm_transformer':
            print("!!!!!!!!!!!!! AnyPlace Diffusion !!!!!!!!!!!!!")
            pr_model_cls = NSMTransformerSingleTransformationRegression
            pr_model = pr_model_cls(
                mc_vis=mc_vis, 
                feat_dim=args.model.refine_pose.feat_dim, 
            **pr_args).cuda()
        elif pr_type == 'nsm_implicit':
            print("!!!!!!!!!!!!! Using NSMTransformerImplicit !!!!!!!!!!!!!")
            pr_model_cls = NSMTransformerImplicit
            pr_model = pr_model_cls(
                mc_vis=mc_vis, 
                feat_dim=args.model.refine_pose.feat_dim, 
                is_train=False,
                **pr_args).cuda()

        pr_model.load_state_dict(pose_refine_ckpt['refine_pose_model_state_dict'])

    # Load success classifier and optimizer
    success_model = None
    coarse_aff_model = None

    #####################################################################################
    # prepare function used for inference using set of trained models
    if args.experiment.eval.model_type == 'anyplace_diffusion_molmocrop':
        infer_relation_policy = policy_inference_methods_dict[args.experiment.eval.inference_method]
    elif args.experiment.eval.model_type == 'nsm_implicit':
        infer_relation_policy = nsm_policy_inference_methods_dict[args.experiment.eval.inference_method]

    #####################################################################################
    # prepare the simuation environment
    rec_stop_event = threading.Event()
    rec_run_event = threading.Event()
    rec_th = threading.Thread(target=pb2mc_update, args=(recorder, mc_vis, rec_stop_event, rec_run_event))
    rec_th.daemon = True
    rec_th.start()

    pause_mc_thread = lambda pause_bool : rec_run_event.clear() if pause_bool else rec_run_event.set()
    pause_mc_thread(False)
    
    # Save full name of model paths used
    if exp_args.load_pose_regression:
        args.experiment.eval.pose_refine_model_name_full = pose_refine_model_path

    if util.exists_and_true(exp_args.eval, 'multi_aff_rot'):
        infer_kwargs['multi_aff_rot'] = True

    eval_dataset_path = args.experiment.eval.eval_dataset_path
    eval_dataset_list = [f.path for f in os.scandir(eval_dataset_path) if f.is_dir()]
    num_clons = exp_args.eval.init_k_val 
    for data_path in eval_dataset_list:
        base_obj_name = data_path.split('/')[-1].split('_on_')[-1]
        grasp_obj_name = data_path.split('/')[-1].split('_on_')[0]
        output_path = f'{data_path}/{args.experiment.eval.model_name}_relative_pose_prediction.npy' 
        print(f'!!!!!!!!! Running inference for {base_obj_name} and {grasp_obj_name} !!!!!!!!')
        exp_args.num_iterations = 1

        base_obj_path= f'{data_path}/base_obj_pointcloud.ply'
        target_obj_path = f'{data_path}/target_obj_pointcloud.ply' 
        base_obj = o3d.io.read_point_cloud(base_obj_path)
        target_obj = o3d.io.read_point_cloud(target_obj_path)
        base_obj_pc = np.asarray(base_obj.points)
        grasp_obj_pc = np.asarray(target_obj.points)

        # copy usd files from obj folder to current folder
        obj_directory = args.experiment.eval.obj_path
        base_obj_class = base_obj_name.split('_')[0]
        base_obj_usd_path = f'{obj_directory}/{base_obj_class}/{base_obj_name}.usd'
        grasp_obj_class = grasp_obj_name.split('_')[0]
        grasp_obj_usd_path = f'{obj_directory}/{grasp_obj_class}/{grasp_obj_name}.usd'
        os.system(f'cp {base_obj_usd_path} {data_path}')
        os.system(f'cp {grasp_obj_usd_path} {data_path}')
        if ('rack' in base_obj_name or 'vialplate' in base_obj_name):
            base_obj_info_path = f'{obj_directory}/{base_obj_class}/{base_obj_name}_data.npy'
            os.system(f'cp {base_obj_info_path} {data_path}')

        use_molmo_crop = False   # this is to prevent cropping the base object like pegs in insertion eval
        # check if the word "rack, vialplate, tuberack" is in the base object name
        if args.experiment.eval.use_molmo and ('rack' in base_obj_name or 'vialplate' in base_obj_name or 'tuberack' in base_obj_name) :
            # read predicted hole location of base objecfrom Molmo
            use_molmo_crop = True
            base_obj_info_path = f'{obj_directory}/{base_obj_class}/{base_obj_name}_place_info/{base_obj_name}_points_3d.npy' 
            hole_locations = np.load(base_obj_info_path)
            print("!!!!!!!!!!!!!!!!!!!!! LOAD 3D POINT Using Molmo crop !!!!!!!!!!!!!!!!!!!!!")
            num_holes = hole_locations.shape[0]
            exp_args.num_iterations = num_holes
        
        orginal_base_obj_pc = base_obj_pc.copy()
        for iteration in range(exp_args.start_iteration, exp_args.num_iterations):
            if args.experiment.eval.use_molmo and use_molmo_crop:    
                base_obj_name = base_obj_name.split('_')[0]
                if base_obj_name == "rack":
                    hole_location = hole_locations[iteration][1:]   # eliminate the first element which is the rack pole orentation (ground truth)
                else:
                    print("cropping base object")
                    hole_location = hole_locations[iteration]

                # find the width, height, and length of grasp_obj_pc
                grasp_obj_pc_min = np.min(grasp_obj_pc, axis=0)
                grasp_obj_pc_max = np.max(grasp_obj_pc, axis=0)
                grasp_obj_pc_width = grasp_obj_pc_max[0] - grasp_obj_pc_min[0]
                grasp_obj_pc_height = grasp_obj_pc_max[1] - grasp_obj_pc_min[1]
                grasp_obj_pc_length = grasp_obj_pc_max[2] - grasp_obj_pc_min[2]
                
                # find the minumum of width, height, and length of the grasp_obj_pc
                min_dim = min(grasp_obj_pc_width, grasp_obj_pc_height, grasp_obj_pc_length)
                max_length = min_dim + 0.01
                xmin, xmax = hole_location[0] - max_length, hole_location[0] + max_length
                ymin, ymax = hole_location[1] - max_length, hole_location[1] + max_length
                zmin, zmax = hole_location[2] - max_length-0.1, hole_location[2] + max_length+0.1 # shift up only for simulated data
                crop_base_obj_pc = util.crop_pcd(orginal_base_obj_pc,x=[xmin, xmax],y=[ymin, ymax],z=[zmin, zmax])
            else:
                crop_base_obj_pc = base_obj_pc
                num_clons = 100  # for object that does not need cropping, do 100 iterations

            if crop_base_obj_pc.shape[0] < 10:
                continue
            #####################################################################################
            # set up the trial
            pause_mc_thread(True)
            multi_mesh_dict = dict(
                parent_file=None,
                parent_scale=None,
                parent_pose=None,
                child_file=None,
                child_scale=None,
                child_pose=None,
                multi=True)
            
            scene_extents = args.data.coarse_aff.scene_extents 
            scene_scale = 1 / np.max(scene_extents)

            args.data.coarse_aff.scene_scale = scene_scale 
            infer_kwargs['gt_child_cent'] = None
            infer_kwargs['export_viz'] = args.export_viz
            infer_kwargs['export_viz_dirname'] = args.export_viz_dirname
            infer_kwargs['export_viz_relative_trans_guess'] = None
            infer_kwargs['iteration'] = iteration
            
            base_obj_pc  = crop_base_obj_pc
            parent_pcd = base_obj_pc
            child_pcd_guess = grasp_obj_pc
            relative_trans_preds = infer_relation_policy(
                mc_vis, 
                parent_pcd, child_pcd_guess, 
                coarse_aff_model,
                pr_model, 
                success_model,
                scene_mean=args.data.coarse_aff.scene_mean, scene_scale=args.data.coarse_aff.scene_scale, 
                grid_pts=raster_pts, rot_grid=rot_grid, 
                viz=False, n_iters=exp_args.eval.n_refine_iters, 
                no_parent_crop=(not exp_args.parent_crop),
                return_top=(not exp_args.eval.return_rand), with_coll=exp_args.eval.with_coll, 
                run_affordance=exp_args.eval.run_affordance, init_k_val=num_clons,
                no_sc_score=exp_args.eval.no_success_classifier, 
                init_parent_mean=exp_args.eval.init_parent_mean_pos, init_orig_ori=exp_args.eval.init_orig_ori,
                refine_anneal=exp_args.eval.refine_anneal,
                mesh_dict=multi_mesh_dict,
                add_per_iter_noise=exp_args.eval.add_per_iter_noise,
                per_iter_noise_kwargs=exp_args.eval.per_iter_noise_kwargs,
                variable_size_crop=exp_args.eval.variable_size_crop,
                timestep_emb_decay_factor=exp_args.eval.timestep_emb_decay_factor,
                remove_redundant_pose = args.experiment.eval.remove_redundant_pose,
                **infer_kwargs)

            # save data
            data = {"relative": relative_trans_preds}
            if args.experiment.eval.use_molmo and os.path.exists(output_path):
                existing_data = np.load(output_path, allow_pickle=True).item()
                existing_data["relative"] = np.vstack([existing_data["relative"], data["relative"]])
                np.save(output_path, existing_data)
            else:
                np.save(output_path, np.array([data]))


if __name__ == "__main__":
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_fname', type=str, required=True, help='Name of config file')
    parser.add_argument('-d', '--debug', action='store_true', help='If True, run in debug mode')
    parser.add_argument('-dd', '--debug_data', action='store_true', help='If True, run data loader in debug mode')
    parser.add_argument('-p', '--port_vis', type=int, default=6000, help='Port for ZMQ url (meshcat visualization)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-l', '--local_dataset_dir', type=str, default=None, help='If the data is saved on a local drive, pass the root of that directory here')
    parser.add_argument('-ex', '--export_viz', action='store_true', help='If True, save data for post-processed visualization')
    parser.add_argument('--export_viz_dirname', type=str, default='rpdiff_export_viz')
    parser.add_argument('-nm', '--new_meshcat', action='store_true')

    args = parser.parse_args()
    eval_args = config_util.load_config(osp.join(path_util.get_eval_config_dir(), args.config_fname), demo_train_eval='eval')
    eval_args['debug'] = args.debug
    eval_args['debug_data'] = args.debug_data
    eval_args['port_vis'] = args.port_vis
    eval_args['seed'] = args.seed
    eval_args['local_dataset_dir'] = args.local_dataset_dir
    
    # other runtime options
    eval_args['export_viz'] = args.export_viz
    eval_args['export_viz_dirname'] = args.export_viz_dirname
    
    # if we want to override the port setting for meshcat and directly start our own
    eval_args['new_meshcat'] = args.new_meshcat
    eval_args = config_util.recursive_attr_dict(eval_args)

    main(eval_args)
