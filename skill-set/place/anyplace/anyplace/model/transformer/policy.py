import torch
import torch.nn as nn

from anyplace.model.transformer import nsm_transformer
from anyplace.model.policy_feat_encoder import LocalAbstractPolicy
from anyplace.utils.config_util import AttrDict
from anyplace.utils.torch_util import maxpool, meanpool, SinusoidalPosEmb
from anyplace.model.transformer import implicit_rot   

import numpy as np
from meshcat import Visualizer


class NSMTransformerImplicit(LocalAbstractPolicy):
    def __init__(self, 
                 feat_dim: int, 
                 in_dim: int=3, 
                 hidden_dim: int=256, 
                 n_heads: int=4, 
                 drop_p: float=0.0, 
                 n_blocks: int=2, 
                 n_pts: int=64, pn_pts: int=None, cn_pts: int=None, 
                 n_queries: int=2, 
                 pooling: str='mean', 
                 predict_offset: bool=False, 
                 bidir: bool=False, 
                 use_timestep_emb: bool=False, 
                 max_timestep: int=None, 
                 is_train: bool=True,
                 timestep_pool_method: str='meanpool',
                 mc_vis: Visualizer=None):
        """
        Args:
            feat_dim (int): Dimensionality of the input features (typically 3D for point clouds, or 5D for point clouds A and B - with one-hot)
            in_dim (int): Dimensionality of the first layer of our policy, after encoding
                the point cloud and the point cloud features (from the potentially pre-trained encoder).
                The very first operation is to project 3D -> in_dim/2 and feat_dim -> in_dim/2
                and concatenate these to provide to our main module
            hidden_dim (int): Internal dimensionality of our main modules
            n_heads (int): Number of heads for multi-head self-attention
            drop_p (float): Between 0.0 and 1.0, probability of dropout
            n_blocks (int): Number of multi-head self-attention blocks to use in the transformer
            n_pts (int): Number of points to downsample to, for each shape (so total number of
                points will be 2*n_pts)
            pn_pts (int): Number of points to downsample parent/scene to
            cn_pts (int): Number of points to downsample child/object to
            n_queries (int): Number of query tokens to use for output (unused currently, deafult 1 output)
            pooling (str): 'mean' or 'max', for how we pool the output features from the transformer
            bidir (bool): If True, compute object-scene cross attention in both directions
            use_timestep_emb (bool): If True, also condition on timestep/iteration embedding
            max_timestep (int): Value to clip the maximum timestep to
            timestep_pool_method (str): 'meanpool' or 'concat'
            mc_vis (Visualizer): Meshcat interface for debugging visualization
        """
        super().__init__(n_pts=n_pts, pn_pts=pn_pts, cn_pts=cn_pts)
        self.hidden_dim = hidden_dim
        self.in_dim = hidden_dim
        self.n_pts = n_pts
        self.pn_pts = pn_pts
        self.cn_pts = cn_pts
        self.mc_vis = mc_vis
        self.is_train = is_train
        
        # input projections
        self.pcd_emb = nn.Linear(3, int(self.in_dim/2))
        self.des_emb = nn.Linear(feat_dim, int(self.in_dim/2))
        print("!!!!!!!!!!!!!!!!!!! IN Implicit TRANSFOERM !!!!!!!!!!!!!!!!!!!!!!!!!")

        # one query per output transformation
        self.n_queries = n_queries
        self.bidir = bidir

        cfg = AttrDict(dict())
        cfg.model = AttrDict(dict())
        cfg.model.num_heads = n_heads
        cfg.model.num_blocks = n_blocks
        cfg.model.pc_feat_dim = self.in_dim + 2
        cfg.model.transformer_feat_dim = hidden_dim 
        self.transformer = nsm_transformer.Transformer(cfg)
        
        self.implicit_rot = implicit_rot.ImplicitSO3(
            len_visual_description = 258, 
            number_fourier_components = 4, 
            mlp_layer_sizes = [258]*4,
            so3_sampling_mode = 'random', 
            number_train_queries = 4096, 
            number_eval_queries = 2000000)
        self.set_predict_offset(predict_offset)
        
        self._build_output_heads(pooled_dim=hidden_dim + 2, hidden_dim=hidden_dim)
        self.pool = meanpool if pooling == 'mean' else maxpool
        self.perm = None

        self.per_point_h = None
        
        self.pos_emb_max_timestep = max_timestep
        self.pos_emb = SinusoidalPosEmb(dim=self.hidden_dim + 2, max_pos=self.pos_emb_max_timestep)
        self.timestep_emb_proj = nn.Linear(hidden_dim+2, hidden_dim+2)
        

        self.use_timestep_emb = use_timestep_emb
        if timestep_pool_method == 'meanpool':
            self.pool_with_var = self.pool_with_var_meanpool
        elif timestep_pool_method == 'concat':
            self.pool_with_var = self.pool_with_var_concat
            if self.use_timestep_emb:
                self._build_output_heads(pooled_dim=2*(hidden_dim + 2), hidden_dim=hidden_dim)
        else:
            pass

    def set_pos_emb_max_timestep(self, max_timestep: int):
        self.pos_emb_max_timestep = max_timestep
        self.pos_emb = SinusoidalPosEmb(dim=self.hidden_dim + 2, max_pos=self.pos_emb_max_timestep)

    def pool_with_var(self, *args, **kwargs):
        return

    def pool_with_var_meanpool(self, pooled_h: torch.FloatTensor, 
                               new_var: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        out_pooled_h = torch.mean(torch.stack(
            [
                pooled_h, 
                new_var
            ], 1), 1)
        return out_pooled_h

    def pool_with_var_concat(self, pooled_h: torch.FloatTensor, 
                             new_var: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        out_pooled_h = torch.cat((pooled_h, new_var), dim=-1)
        return out_pooled_h

    def forward(self, model_input: dict, *args, **kwargs) -> dict:
        n_pts = self.n_pts

        # centerd 3D point clouds
        ppcd_cent = model_input['parent_start_pcd']
        cpcd_cent = model_input['child_start_pcd']
        B, N = ppcd_cent.shape[0], ppcd_cent.shape[1]
        Np = ppcd_cent.shape[1]
        Nc = cpcd_cent.shape[1]

        # process inputs
        des_dict, pcd_dict = self.process_model_input(model_input)
        ppcd, cpcd = pcd_dict['parent'], pcd_dict['child']
        des_full_parent_coords, des_full_child_coords = des_dict['parent'], des_dict['child']
        
        # downsample with random permutation (voxel or fps downsampling would be better...)
        ppcd, perm1 = self.fps_ds_p.forward(ppcd, return_idx=True)
        cpcd, perm2 = self.fps_ds_c.forward(cpcd, return_idx=True)
        pcd_full = torch.cat([ppcd, cpcd], dim=1)


        des_full_parent_coords = torch.gather(des_full_parent_coords, dim=1, index=perm1[:, :, None].repeat((1, 1, des_full_parent_coords.shape[-1])))
        des_full_child_coords = torch.gather(des_full_child_coords, dim=1, index=perm2[:, :, None].repeat((1, 1, des_full_child_coords.shape[-1])))
        des_full = torch.cat([des_full_parent_coords, des_full_child_coords], dim=1)

        # process per-point features and pool
        per_point_h2 = self.transformer(des_full_child_coords.transpose(2, 1), des_full_parent_coords.transpose(2, 1)).transpose(2, 1)
        per_point_h = per_point_h2
        pooled_h = self.pool(per_point_h, dim=1, keepdim=True)  # B x 1 x D   # 1, 1, 258

        if self.is_train:
            energy_loss = self.implicit_rot(pooled_h, model_input['gt_rotation'], training=True)
        # process model output predictions
        model_output = self.process_model_output(pooled_h)

        number_train_queries = 4096
        _, _, model_output['rot_mat'], _ = self.implicit_rot.predict_probability(pooled_h, number_train_queries, take_softmax=True)
        model_output['rot_mat'] = model_output['rot_mat'].unsqueeze(1)
        if self.is_train:
            model_output["energy_loss"] = energy_loss   

        if 'save_attn' in kwargs:
            self.ppcd_ds = ppcd
            self.cpcd_ds = cpcd
            self.pdes_ds = des_full_parent_coords
            self.cdes_ds = des_full_child_coords

        return model_output
    

class NSMTransformerSingleTransformationRegression(LocalAbstractPolicy):
    def __init__(self, 
                 feat_dim: int, 
                 in_dim: int=3, 
                 hidden_dim: int=256, 
                 n_heads: int=4, 
                 drop_p: float=0.0, 
                 n_blocks: int=2, 
                 n_pts: int=64, pn_pts: int=None, cn_pts: int=None, 
                 n_queries: int=2, 
                 pooling: str='mean', 
                 predict_offset: bool=False, 
                 bidir: bool=False, 
                 use_timestep_emb: bool=False, 
                 max_timestep: int=None, 
                 timestep_pool_method: str='meanpool',
                 mc_vis: Visualizer=None):
        """
        Args:
            feat_dim (int): Dimensionality of the input features (typically 3D for point clouds, or 5D for point clouds A and B - with one-hot)
            in_dim (int): Dimensionality of the first layer of our policy, after encoding
                the point cloud and the point cloud features (from the potentially pre-trained encoder).
                The very first operation is to project 3D -> in_dim/2 and feat_dim -> in_dim/2
                and concatenate these to provide to our main module
            hidden_dim (int): Internal dimensionality of our main modules
            n_heads (int): Number of heads for multi-head self-attention
            drop_p (float): Between 0.0 and 1.0, probability of dropout
            n_blocks (int): Number of multi-head self-attention blocks to use in the transformer
            n_pts (int): Number of points to downsample to, for each shape (so total number of
                points will be 2*n_pts)
            pn_pts (int): Number of points to downsample parent/scene to
            cn_pts (int): Number of points to downsample child/object to
            n_queries (int): Number of query tokens to use for output (unused currently, deafult 1 output)
            pooling (str): 'mean' or 'max', for how we pool the output features from the transformer
            bidir (bool): If True, compute object-scene cross attention in both directions
            use_timestep_emb (bool): If True, also condition on timestep/iteration embedding
            max_timestep (int): Value to clip the maximum timestep to
            timestep_pool_method (str): 'meanpool' or 'concat'
            mc_vis (Visualizer): Meshcat interface for debugging visualization
        """
        super().__init__(n_pts=n_pts, pn_pts=pn_pts, cn_pts=cn_pts)
        self.hidden_dim = hidden_dim
        self.in_dim = hidden_dim
        self.n_pts = n_pts
        self.pn_pts = pn_pts
        self.cn_pts = cn_pts
        self.mc_vis = mc_vis
        
        # input projections
        self.pcd_emb = nn.Linear(3, int(self.in_dim/2))
        self.des_emb = nn.Linear(feat_dim, int(self.in_dim/2))

        # one query per output transformation
        self.n_queries = n_queries
        self.bidir = bidir

        cfg = AttrDict(dict())
        cfg.model = AttrDict(dict())
        cfg.model.num_heads = n_heads
        cfg.model.num_blocks = n_blocks
        cfg.model.pc_feat_dim = self.in_dim + 2
        cfg.model.transformer_feat_dim = hidden_dim 
        self.transformer = nsm_transformer.Transformer(cfg)
        self.set_predict_offset(predict_offset)
        
        # 2*hidden_dim because we combine the pooled output feats with the global 'cls' token feat
        self._build_output_heads(pooled_dim=hidden_dim + 2, hidden_dim=hidden_dim)

        self.pool = meanpool if pooling == 'mean' else maxpool
        self.perm = None

        self.per_point_h = None
        
        self.pos_emb_max_timestep = max_timestep
        self.pos_emb = SinusoidalPosEmb(dim=self.hidden_dim + 2, max_pos=self.pos_emb_max_timestep)
        self.timestep_emb_proj = nn.Linear(hidden_dim+2, hidden_dim+2)        
        self.use_timestep_emb = use_timestep_emb
        if timestep_pool_method == 'meanpool':
            self.pool_with_var = self.pool_with_var_meanpool
        elif timestep_pool_method == 'concat':
            self.pool_with_var = self.pool_with_var_concat
            # we have to re-build the output heads in this case, since we are concatenating (dim will be double)
            if self.use_timestep_emb:
                self._build_output_heads(pooled_dim=2*(hidden_dim + 2), hidden_dim=hidden_dim)
        else:
            pass

    def set_pos_emb_max_timestep(self, max_timestep: int):
        self.pos_emb_max_timestep = max_timestep
        self.pos_emb = SinusoidalPosEmb(dim=self.hidden_dim + 2, max_pos=self.pos_emb_max_timestep)

    def pool_with_var(self, *args, **kwargs):
        return

    def pool_with_var_meanpool(self, pooled_h: torch.FloatTensor, 
                               new_var: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        out_pooled_h = torch.mean(torch.stack(
            [
                pooled_h, 
                new_var
            ], 1), 1)
        return out_pooled_h

    def pool_with_var_concat(self, pooled_h: torch.FloatTensor, 
                             new_var: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        out_pooled_h = torch.cat((pooled_h, new_var), dim=-1)
        return out_pooled_h

    def forward(self, model_input: dict, *args, **kwargs) -> dict:
        n_pts = self.n_pts

        # centerd 3D point clouds
        ppcd_cent = model_input['parent_start_pcd']
        cpcd_cent = model_input['child_start_pcd']
        B, N = ppcd_cent.shape[0], ppcd_cent.shape[1]
        Np = ppcd_cent.shape[1]
        Nc = cpcd_cent.shape[1]

        # process inputs
        des_dict, pcd_dict = self.process_model_input(model_input)
        ppcd, cpcd = pcd_dict['parent'], pcd_dict['child']
        des_full_parent_coords, des_full_child_coords = des_dict['parent'], des_dict['child']
        
        # downsample with random permutation (voxel or fps downsampling would be better...)
        ppcd, perm1 = self.fps_ds_p.forward(ppcd, return_idx=True)
        cpcd, perm2 = self.fps_ds_c.forward(cpcd, return_idx=True)
        pcd_full = torch.cat([ppcd, cpcd], dim=1)

        des_full_parent_coords = torch.gather(des_full_parent_coords, dim=1, index=perm1[:, :, None].repeat((1, 1, des_full_parent_coords.shape[-1])))
        des_full_child_coords = torch.gather(des_full_child_coords, dim=1, index=perm2[:, :, None].repeat((1, 1, des_full_child_coords.shape[-1])))
        des_full = torch.cat([des_full_parent_coords, des_full_child_coords], dim=1)

        # TURN OFF TIMEMD FOR TESTING
        if 'timestep_emb' in model_input and self.use_timestep_emb:
            des_full_child_coords = torch.cat((des_full_child_coords, model_input['timestep_emb'].reshape(B, 1, -1)), dim=1)

        # process per-point features and pool
        per_point_h2 = self.transformer(des_full_child_coords.transpose(2, 1), des_full_parent_coords.transpose(2, 1)).transpose(2, 1)
        per_point_h = per_point_h2
        pooled_h = self.pool(per_point_h, dim=1, keepdim=True)  # B x 1 x D   # 1, 1, 258
        
        if 'timestep_emb' in model_input and self.use_timestep_emb:
            pooled_h = self.pool_with_var(
                pooled_h, 
                self.timestep_emb_proj(model_input['timestep_emb']).reshape(B, 1, -1)
            )

        # process model output predictions
        model_output = self.process_model_output(pooled_h)

        if 'save_attn' in kwargs:
            self.ppcd_ds = ppcd
            self.cpcd_ds = cpcd
            self.pdes_ds = des_full_parent_coords
            self.cdes_ds = des_full_child_coords

        return model_output

