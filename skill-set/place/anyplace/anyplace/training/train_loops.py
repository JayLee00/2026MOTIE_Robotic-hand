import torch
import torch.nn as nn
import time
import wandb
import numpy as np
from anyplace.utils.torch3d_util import matrix_to_quaternion
from anyplace.training.train_util import adjust_learning_rate, get_grad_norm
from anyplace.utils.config_util import AttrDict
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from meshcat import Visualizer
from typing import Callable

MC_SIZE=0.007

def train_iter_refine_pose(
            pose_refine_mi: dict,
            pose_refine_gt: dict,
            pose_refine_model: nn.Module,
            pr_optimizer: Optimizer,
            pr_loss_fn: Callable,
            args: AttrDict,
            it: int, current_epoch: float,
            logger: SummaryWriter,
            mc_vis: Visualizer=None):

    mc_iter_name = 'scene/train_iter_pose'
    if mc_vis is not None:
        mc_vis[mc_iter_name].delete()
    start_time = time.time()

    bs = pose_refine_mi['parent_start_pcd'].shape[0]
    db_idx = 0

    #######################################################################   

    if args.data.refine_pose.diffusion_steps:
        # position embedding for the timestep
        timestep_emb = pose_refine_model.pos_emb(pose_refine_mi['diffusion_timestep'])
        pose_refine_mi['timestep_emb'] = timestep_emb

    rot_model_output_raw = pose_refine_model(pose_refine_mi, rot=True)

    # apply output rotation to object point cloud for translation prediction
    child_start_pcd_original = pose_refine_mi['child_start_pcd'].clone().detach()
    rot_idx = torch.zeros(rot_model_output_raw['rot_mat'].shape[0]).long().cuda()
    rot_model_output = {}
    rot_model_output['rot_mat'] = torch.gather(rot_model_output_raw['rot_mat'], dim=1, index=rot_idx[:, None, None, None].repeat(1, 1, 3, 3)).reshape(bs, 3, 3)
    rot_model_output['quat'] = matrix_to_quaternion(rot_model_output['rot_mat'])

    # apply output rotation to object point cloud for translation prediction
    child_pcd_final_pred = torch.bmm(
        rot_model_output['rot_mat'], 
        pose_refine_mi['child_start_pcd'].transpose(1, 2)).transpose(2, 1).contiguous()  # flip to B x 3 x N, back to B x N x 3
    child_pcd_rot = child_pcd_final_pred.clone().detach()
    pose_refine_mi['child_start_pcd'] = child_pcd_rot

    #######################################################################
    # After re-encoding the rotated shape, make translation refinement prediction
    # make translation prediction
    trans_model_output_raw = pose_refine_model(pose_refine_mi, stop=False)
    trans_idx = torch.zeros(trans_model_output_raw['trans'].shape[0]).long().cuda()
    trans_model_output = {}
    trans_model_output['trans'] = torch.gather(trans_model_output_raw['trans'], dim=1, index=trans_idx[:, None, None].repeat(1, 1, 3)).reshape(bs, 3)

    pose_refine_model_output = {}
    pose_refine_model_output['trans_raw'] = trans_model_output_raw['trans']
    pose_refine_model_output['trans'] = trans_model_output['trans']
    pose_refine_model_output['quat_raw'] = rot_model_output_raw['quat']
    pose_refine_model_output['rot_mat_raw'] = rot_model_output_raw['rot_mat']
    pose_refine_model_output['quat'] = rot_model_output['quat']
    pose_refine_model_output['rot_mat'] = rot_model_output['rot_mat']
    pose_refine_model_output['unnorm_quat'] = pose_refine_model_output['quat']

    # apply output transformation to object point cloud (for chamfer loss + visualization)
    child_pcd_pred_rot = torch.bmm(pose_refine_model_output['rot_mat'], child_start_pcd_original.transpose(1, 2)).transpose(2, 1)
    child_pcd_pred_rot = child_pcd_pred_rot + pose_refine_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat(1, child_pcd_pred_rot.shape[1], 1) 
    child_pcd_final_pred = child_pcd_pred_rot + pose_refine_model_output['trans'].reshape(-1, 1, 3).repeat(1, child_pcd_pred_rot.shape[1], 1) 
    pose_refine_model_output['child_pcd_final_pred'] = child_pcd_final_pred

    if args.experiment.train.predict_offset:
        # update mean for pose refine model input offset prediction
        pose_refine_mi['child_start_pcd_mean'] = torch.mean(pose_refine_model_output['child_pcd_final_pred'], dim=1)
        if pose_refine_mi.get('parent_start_pcd_offset') is not None:
            pose_refine_mi['parent_start_pcd'] = pose_refine_mi['parent_start_pcd_offset']
        trans_offset_model_output_raw = pose_refine_model(pose_refine_mi, stop=False)
        offset_ind = 0  # force the first index for unimodal offset prediction
        offset_idx = (torch.ones((bs)) * offset_ind).long().cuda()

        trans_offset_out = torch.gather(trans_offset_model_output_raw['trans_offset'], dim=1, index=offset_idx[:, None, None].repeat(1, 1, 3)).reshape(bs, 3)
        pose_refine_model_output['trans_offset_raw'] = trans_offset_model_output_raw['trans_offset']
        pose_refine_model_output['trans_offset'] = trans_offset_out

    #######################################################################
    # Combine model outputs and compute loss
    loss_dict = pr_loss_fn(pose_refine_model_output, pose_refine_gt) 
    loss_trans = loss_dict['trans']
    loss_rot = loss_dict['rot']
    loss_chamf = loss_dict['chamf']

    if args.loss.refine_pose.chamf_only:
        loss = loss_chamf
    elif args.loss.refine_pose.trans_rot_only:
        loss = loss_trans + loss_rot
    elif "implicit" in args.model.refine_pose.type:
        loss = rot_model_output_raw["energy_loss"] + loss_trans 
        loss_dict["energy_loss"] = rot_model_output_raw["energy_loss"]
    else:
        loss = loss_trans + loss_rot + loss_chamf
    loss_dict['total_loss'] = loss

    #######################################################################
    # Gradient step, log, and return
    if args.optimizer.refine_pose.use_schedule:
        sched_args = args.optimizer.schedule
        if args.optimizer.refine_pose.get('schedule') is not None:
            if args.optimizer.refine_pose.schedule is not None:
                sched_args = args.optimizer.refine_pose.schedule

        opt_type = args.optimizer.refine_pose.type
        sched_args.lr = args.optimizer[opt_type].lr
        sched_args.epochs = args.experiment.num_iterations * bs / args.experiment.dataset_length

        adj_lr = adjust_learning_rate(pr_optimizer, current_epoch, sched_args) 
        lr = pr_optimizer.param_groups[0]['lr'] 

    pr_optimizer.zero_grad()
    loss.backward()
    grad_norm = get_grad_norm(pose_refine_model) 
    pr_optimizer.step()

    if it % args.experiment.log_interval == 0 and args.experiment.train.out_log_refine_pose:
        string = f'[Pose Refinement] Iteration: {it} (Epoch: {int(current_epoch)}) '
        for loss_name, loss_val in loss_dict.items():
            string += f'{loss_name}: {loss_val.mean().item():.6f} '
            logger.add_scalar(loss_name, loss_val.mean().item(), it)
            end_time = time.time()
            total_duration = end_time - start_time

        string += f' Grad norm : {grad_norm:.4f}'
        if args.optimizer.refine_pose.use_schedule:
            string += f' LR : {lr:.7f}'

        string += f' Duration: {total_duration:.4f}'
        print(string)
        parent_final_pcd_visual = pose_refine_gt['parent_final_pcd'][db_idx].detach().cpu().numpy()
        child_pcd_final_pred_visual = pose_refine_model_output['child_pcd_final_pred'][db_idx].detach().cpu().numpy()
        wandb.log(
            {
                "point_scene": wandb.Object3D(np.concatenate([parent_final_pcd_visual, child_pcd_final_pred_visual], axis=0))
            }
        )

    out_dict = {}
    out_dict['loss'] = loss_dict
    out_dict['model_output'] = pose_refine_model_output

    return out_dict
