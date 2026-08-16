"""AnyPlace cu128/Blackwell PoC — build refine model, load ckpt, run one
multistep_regression_scene() on dummy parent/child clouds → print T_pred (K,4,4).
Run: conda activate anyplace_cu128 && python poc_infer.py
"""
import sys, os
AP = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place/anyplace"
sys.path.insert(0, AP + "/_stubs")   # airobot/meshcat/mesh_to_sdf stubs FIRST
sys.path.insert(0, AP)               # anyplace package
os.environ.setdefault("ANYPLACE_SOURCE_DIR", AP + "/anyplace_model")
os.environ.setdefault("ANYPLACE_DATA_DIR", AP + "/data")

import numpy as np
import torch
from meshcat import Visualizer  # stub

from anyplace.model.transformer.policy import NSMTransformerSingleTransformationRegression
from anyplace.utils.mesh_util import three_util
from anyplace.utils import util
from anyplace.utils.anyplace.multistep_pose_regression_anyplace import multistep_regression_scene

CKPT = AP + "/weight/anyplace_ckpts/anyplace_multitask/model.pth"

# ---- build model (kwargs from ckpt['args']['model']) --------------------------
pr_args = dict(n_blocks=4, n_heads=1, drop_p=0.0, n_pts=1024, pn_pts=None, cn_pts=None,
               hidden_dim=256, pooling="max", bidir=False,
               n_queries=1, use_timestep_emb=True, max_timestep=5,
               timestep_pool_method="meanpool")
print("[1] building NSMTransformerSingleTransformationRegression ...")
model = NSMTransformerSingleTransformationRegression(feat_dim=3, **pr_args).cuda().eval()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ck["refine_pose_model_state_dict"], strict=False)
print("    loaded. missing:", len(missing), "unexpected:", len(unexpected))
if missing[:3]:    print("    e.g. missing:", missing[:3])
if unexpected[:3]: print("    e.g. unexpected:", unexpected[:3])

# ---- grids --------------------------------------------------------------------
print("[2] building raster + rot grids ...")
reso, pad = 32, 0.1
raster_pts = three_util.get_raster_points(reso, padding=pad).reshape(reso, reso, reso, 3)
raster_pts = raster_pts.transpose(2, 1, 0, 3).reshape(-1, 3)
rot_grid = util.generate_healpix_grid(size=int(1e4))
print("    raster_pts", raster_pts.shape, "rot_grid", rot_grid.shape)

# ---- dummy clouds (meters, near scene_mean) -----------------------------------
rng = np.random.default_rng(0)
scene_mean = np.array([0.35, 0.0, 0.0])
parent_pcd = rng.normal(scene_mean, [0.10, 0.10, 0.01], size=(4000, 3))  # flat tray-ish region
child_pcd = rng.normal(scene_mean + [0.0, 0.0, 0.20], [0.03, 0.03, 0.03], size=(2000, 3))  # small object
scene_scale = 1.0 / 1.2

# ---- inference ----------------------------------------------------------------
print("[3] running multistep_regression_scene ...")
mc = Visualizer()
out_tf = multistep_regression_scene(
    mc, parent_pcd.astype(np.float32), child_pcd.astype(np.float32),
    None, model, None,
    scene_scale=scene_scale, scene_mean=scene_mean,
    grid_pts=raster_pts, rot_grid=rot_grid,
    viz=False, n_iters=50, no_parent_crop=False,
    return_top=True, with_coll=False, run_affordance=False, init_k_val=20,
    no_sc_score=True, init_parent_mean=False, init_orig_ori=False, refine_anneal=False,
    add_per_iter_noise=True,
    per_iter_noise_kwargs={"rot": {"angle_deg": 20, "rate": 6.5},
                           "trans": {"trans_dist": 0.03, "rate": 5.5}},
    variable_size_crop=True, timestep_emb_decay_factor=20, remove_redundant_pose=False,
    gt_child_cent=None, export_viz=False, export_viz_dirname=None,
    export_viz_relative_trans_guess=None, compute_coverage_scores=False,
    out_coverage_dirname1=None, out_coverage_dirname2=None, iteration=0,
    mesh_dict=dict(parent_file=None, parent_scale=None, parent_pose=None,
                   child_file=None, child_scale=None, child_pose=None, multi=True),
)
out_tf = np.asarray(out_tf)
print("[OK] out_tf shape:", out_tf.shape)
print("     T_pred[0]=\n", out_tf[0] if out_tf.ndim == 3 else out_tf)
