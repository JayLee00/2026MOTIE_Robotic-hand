import os.path as osp
import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import copy
import datetime
import shutil
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import meshcat
from anyplace.utils import util, config_util, path_util
from anyplace.utils.torch_util import dict_to_gpu
from anyplace.training import dataio_full_chunked as dataio, losses
from anyplace.model.transformer.policy import (
    NSMTransformerSingleTransformationRegression, 
    NSMTransformerImplicit
    )
from anyplace.training.train_loops import train_iter_refine_pose
from typing import Callable
from torch.optim.optimizer import Optimizer
import wandb

wandb.login()

MC_SIZE = 0.005


def train(
        mc_vis: meshcat.Visualizer, 
        refine_pose_model: nn.Module, 
        pr_optimizer: Optimizer, 
        train_dataloader: DataLoader, test_dataloader: DataLoader, 
        pr_loss_fn: Callable,
        dev: torch.device, 
        logger: SummaryWriter, 
        logdir: str, 
        args: config_util.AttrDict,
        start_iter: int=0,
        **kwargs):

    refine_pose_model.train()

    bs = args.experiment.batch_size
    it = start_iter
    voxel_grid_pts = torch.from_numpy(train_dataloader.dataset.raster_pts).float().cuda()
    rot_mat_grid = torch.from_numpy(train_dataloader.dataset.rot_grid).float().cuda()
    args.experiment.dataset_length = len(train_dataloader.dataset)

    while True:
        if it > args.experiment.num_iterations:
            break
        for sample in train_dataloader:
            it += 1
            current_epoch = it * bs / len(train_dataloader.dataset)
            start_time = time.time()
            coarse_aff_sample, refine_pose_sample = sample
            coarse_aff_mi, coarse_aff_gt = coarse_aff_sample
            refine_pose_mi, refine_pose_gt = refine_pose_sample
            refine_pose_out = None
            loss_dict = {}

            if args.experiment.train.train_refine_pose and (len(refine_pose_mi) > 0):
                # prepare input and gt
                refine_pose_mi = dict_to_gpu(refine_pose_mi)
                refine_pose_gt = dict_to_gpu(refine_pose_gt)
                refine_pose_out = train_iter_refine_pose(
                    refine_pose_mi,
                    refine_pose_gt,
                    refine_pose_model,
                    pr_optimizer,
                    pr_loss_fn,
                    args,
                    it, current_epoch,
                    logger,
                    mc_vis=mc_vis)
                
                # process output for logging
                for k, v in refine_pose_out['loss'].items():
                    loss_dict[k] = v
                    wandb.log({k: v})

            #######################################################################
            # Logging and checkpoints

            if it % args.experiment.log_interval == 0 and args.experiment.train.out_log_full:
                string = f'Iteration {it} -- '

                for loss_name, loss_val in loss_dict.items():
                    if isinstance(loss_val, dict):
                        # don't let these loss dicts get more than two levels deep
                        for k, v in loss_val.items():
                            string += f'{k}: {v.mean().item():.6f} '
                            logger.add_scalar(k, v.mean().item(), it)
                    else:
                        string += f'{loss_name}: {loss_val.mean().item():.6f} '
                        logger.add_scalar(loss_name, loss_val.mean().item(), it)

                end_time = time.time()
                total_duration = end_time - start_time
                string += f'duration: {total_duration:.4f}'
                print(string)

            if it % args.experiment.save_interval == 0 and it > 0:
                model_path = osp.join(logdir, f'model_{it}.pth')
                model_path_latest = osp.join(logdir, 'model_latest.pth')

                ckpt = {'args': config_util.recursive_dict(args)}
                ckpt['refine_pose_model_state_dict'] = refine_pose_model.state_dict()
                ckpt['pr_optimizer_state_dict'] = pr_optimizer.state_dict()
                
                torch.save(ckpt, model_path)
                torch.save(ckpt, model_path_latest)


def main(args: config_util.AttrDict):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ##############################################
    # Setup basic experiment params

    logdir = osp.join(
        path_util.get_anyplace_model_weights(), 
        args.experiment.logdir, 
        args.experiment.experiment_name)
    util.safe_makedirs(logdir)

    # Set up experiment run/config logging
    nowstr = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_logs = osp.join(logdir, 'run_logs')
    util.safe_makedirs(run_logs)
    run_log_folder = osp.join(run_logs, nowstr)
    util.safe_makedirs(run_log_folder)

    # copy everything we would like to know about this run in the run log folder
    for fn in os.listdir(os.getcwd()):
        if not (fn.endswith('.py') or fn.endswith('.sh') or fn.endswith('.bash')):
            continue
        log_fn = osp.join(run_log_folder, fn)
        shutil.copy(fn, log_fn) 

    full_cfg_dict = copy.deepcopy(config_util.recursive_dict(args))
    full_cfg_fname = osp.join(run_log_folder, 'full_exp_cfg.txt')
    json.dump(full_cfg_dict, open(full_cfg_fname, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    if args.experiment.meshcat_on and args.meshcat_ap:
        zmq_url=f'tcp://127.0.0.1:{args.port_vis}'
        mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
        mc_vis['scene'].delete()
    else:
        mc_vis = None

    # prepare dictionary for extra kwargs in train function
    train_kwargs = {}

    ##############################################
    # Prepare dataset and dataloader

    data_args = args.data
    if osp.exists(str(args.local_dataset_dir)):
        dataset_path = osp.join(
            args.local_dataset_dir, 
            data_args.data_root,
            data_args.dataset_path)
    else:
        dataset_path = osp.join(
            path_util.get_anyplace_data(), 
            data_args.data_root,
            data_args.dataset_path)

    assert osp.exists(dataset_path), f'Dataset path: {dataset_path} does not exist'
    
    train_dataset = dataio.FullRelationPointcloudPolicyDataset(
        dataset_path, 
        data_args,
        phase='train', 
        train_coarse_aff=args.experiment.train.train_coarse_aff,
        train_refine_pose=args.experiment.train.train_refine_pose,
        train_success=args.experiment.train.train_success,
        mc_vis=mc_vis, 
        debug_viz=args.debug_data)
    val_dataset = dataio.FullRelationPointcloudPolicyDataset(
        dataset_path, 
        data_args,
        phase='val', 
        train_coarse_aff=args.experiment.train.train_coarse_aff,
        train_refine_pose=args.experiment.train.train_refine_pose,
        train_success=args.experiment.train.train_success,
        mc_vis=mc_vis,
        debug_viz=args.debug_data)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.experiment.batch_size, 
        shuffle=True, 
        num_workers=args.experiment.num_train_workers, 
        drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=2, 
        num_workers=1,
        shuffle=False, 
        drop_last=True)

    # grab some things we need for training
    args.experiment.epochs = args.experiment.num_iterations / len(train_dataloader) 
    args.data.rot_grid_bins = train_dataset.rot_grid.shape[0]

    ##############################################
    # Prepare networks

    ###
    # Load pose refinement model, loss, and optimizer
    ###
    pr_type = args.model.refine_pose.type
    pr_args = config_util.copy_attr_dict(args.model[pr_type])
    if args.model.refine_pose.get('model_kwargs') is not None:
        custom_pr_args = args.model.refine_pose.model_kwargs[pr_type]
        config_util.update_recursive(pr_args, custom_pr_args)

    if pr_type == 'nsm_transformer':                    # AnyPlace diffusion
        pr_model_cls = NSMTransformerSingleTransformationRegression
    elif pr_type == 'nsm_implicit':                     # AnyPlace energy baseline
        pr_model_cls = NSMTransformerImplicit
    else:
        raise ValueError(f'Unrecognized: {pr_type}')

    pr_model = pr_model_cls(
        mc_vis=mc_vis, 
        feat_dim=args.model.refine_pose.feat_dim, 
        **pr_args).cuda()
    pr_model_params = pr_model.parameters()

    # loss
    pr_loss_type = args.loss.refine_pose.type
    assert pr_loss_type in args.loss.refine_pose.valid_losses, f'Loss type: {pr_loss_type} not in {args.loss.refine_pose.valid_losses}'

    if pr_loss_type == 'tf_chamfer':
        tfc_mqa_wrapper = losses.TransformChamferWrapper(
            l1=args.loss.tf_chamfer.l1,
            trans_offset=args.loss.tf_chamfer.trans_offset)
        pr_loss_fn = tfc_mqa_wrapper.tf_chamfer
    else:
        raise ValueError(f'Unrecognized: {pr_loss_fn}')

    # optimizer
    pr_opt_type = args.optimizer.refine_pose.type
    assert pr_opt_type in args.optimizer.refine_pose.valid_opts, f'Opt type: {pr_opt_type} not in {args.optimizer.refine_pose.valid_opt}'

    if pr_opt_type == 'Adam':
        pr_opt_cls = torch.optim.Adam 
    elif pr_opt_type == 'AdamW':
        pr_opt_cls = torch.optim.AdamW 
    else:
        raise ValueError(f'Unrecognized: {pr_opt_type}')

    pr_opt_kwargs = config_util.copy_attr_dict(args.optimizer[pr_opt_type])
    if args.optimizer.refine_pose.get('opt_kwargs') is not None:
        custom_pr_opt_kwargs = args.optimizer.refine_pose.opt_kwargs[pr_opt_type]
        config_util.update_recursive(pr_opt_kwargs, custom_pr_opt_kwargs)

    print(f'@@@@@@@@@@@@@@@@@@@ Pose refine optimizer kwargs: {pr_opt_kwargs}')
    print(f'@@@@@@@@@@@@@@@@@@@ pr_model_params optimizer kwargs: {pr_model_params}')
    print(f'@@@@@@@@@@@@@@@@@@@ pr_opt_type: {pr_opt_type}')
        
    pr_optimizer = pr_opt_cls(pr_model_params, **pr_opt_kwargs)

    ##############################################
    # Load checkpoints if resuming
    if args.experiment.resume and args.resume_ap:
        # find the latest iteration
        ckpts = [int(val.split('model_')[1].replace('.pth', '')) for val in os.listdir(logdir) if (val.endswith('.pth') and 'latest' not in val)]
        args.experiment.resume_iter = max(ckpts)
        print(ckpts)

    if args.experiment.resume_iter != 0:
        print(f'Resuming at iteration: {args.experiment.resume_iter}')
        model_path = osp.join(logdir, 'model_latest.pth')
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if args.experiment.train.train_refine_pose:
            pr_model.load_state_dict(checkpoint['refine_pose_model_state_dict'])
            pr_optimizer.load_state_dict(checkpoint['pr_optimizer_state_dict'])

    logger = SummaryWriter(logdir)
    it = args.experiment.resume_iter
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise ValueError('Cuda not available')
    
    train(
        mc_vis, 
        pr_model, 
        pr_optimizer,
        train_dataloader, val_dataloader, 
        pr_loss_fn,
        device, 
        logger, 
        logdir, 
        args,
        it, 
        **train_kwargs)


if __name__ == "__main__":
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_fname', type=str, required=True, help='Name of config file')
    parser.add_argument('-d', '--debug', action='store_true', help='If True, run in debug mode')
    parser.add_argument('-dd', '--debug_data', action='store_true', help='If True, run data loader in debug mode')
    parser.add_argument('-p', '--port_vis', type=int, default=6000, help='Port for ZMQ url (meshcat visualization)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-l', '--local_dataset_dir', type=str, default=None, help='If the data is saved on a local drive, pass the root of that directory here')
    parser.add_argument('-r', '--resume', action='store_true', help='If set, resume experiment (required to be set in config as well)')
    parser.add_argument('-m', '--meshcat', action='store_true', help='If set, run with meshcat visualization (required to be set in config as well)')

    args = parser.parse_args()

    train_args = config_util.load_config(osp.join(path_util.get_train_config_dir(), args.config_fname))
    
    train_args['debug'] = args.debug
    train_args['debug_data'] = args.debug_data
    train_args['port_vis'] = args.port_vis
    train_args['seed'] = args.seed
    train_args['local_dataset_dir'] = args.local_dataset_dir
    train_args['meshcat_ap'] = args.meshcat
    train_args['resume_ap'] = args.resume
    train_args = config_util.recursive_attr_dict(train_args)

    wandb.init(
            project="anyplace",
            name=train_args.experiment.run_name
    )
    
    print(train_args.experiment.run_name)
    main(train_args)
