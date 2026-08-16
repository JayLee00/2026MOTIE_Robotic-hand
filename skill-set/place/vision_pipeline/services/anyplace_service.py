"""AnyPlace placement service (run in conda env `anyplace_cu128`).

Loads the cu128-ported AnyPlace refine model once (see ANYPLACE_PORT.md) and
serves SE(3) placement prediction:

    POST /predict  npz{ parent_pcd:(Np,3), child_pcd:(Nc,3) }  ->  npz{ out_tf:(K,4,4) }

`child_pcd` must already be gravity-aligned (child_pc_zalign). `out_tf` are
relative transforms (final = out_tf @ init); scores are all 1.0 (success
classifier disabled), so the orchestrator ranks them with geometry.rank_upright.

Run:  conda activate anyplace_cu128 && \
      python vision_pipeline/services/anyplace_service.py [--port 8801] [--task anyplace_multitask]
"""
import argparse
import os
import sys

import numpy as np

AP = "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place/anyplace"
sys.path.insert(0, AP + "/_stubs")
sys.path.insert(0, AP)
sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/place")  # for vision_pipeline.services.rpc
os.environ.setdefault("ANYPLACE_SOURCE_DIR", AP + "/anyplace_model")
os.environ.setdefault("ANYPLACE_DATA_DIR", AP + "/data")

import torch  # noqa: E402
from meshcat import Visualizer  # stub  # noqa: E402

from anyplace.model.transformer.policy import NSMTransformerSingleTransformationRegression  # noqa: E402
from anyplace.utils.mesh_util import three_util  # noqa: E402
from anyplace.utils import util  # noqa: E402
from anyplace.utils.anyplace.multistep_pose_regression_anyplace import multistep_regression_scene  # noqa: E402

from vision_pipeline.services.rpc import serve  # noqa: E402

SCENE_MEAN = np.array([0.35, 0.0, 0.0])
SCENE_SCALE = 1.0 / 1.2  # 1 / max(scene_extents=[0.7,1.2,0.0])


def build_model(task):
    ckpt = f"{AP}/weight/anyplace_ckpts/{task}/model.pth"
    pr_args = dict(n_blocks=4, n_heads=1, drop_p=0.0, n_pts=1024, pn_pts=None, cn_pts=None,
                   hidden_dim=256, pooling="max", bidir=False,
                   n_queries=1, use_timestep_emb=True, max_timestep=5,
                   timestep_pool_method="meanpool")
    model = NSMTransformerSingleTransformationRegression(feat_dim=3, **pr_args).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["refine_pose_model_state_dict"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"state_dict mismatch: {len(miss)} missing, {len(unexp)} unexpected"
    return model


def build_grids():
    reso, pad = 32, 0.1
    rp = three_util.get_raster_points(reso, padding=pad).reshape(reso, reso, reso, 3)
    rp = rp.transpose(2, 1, 0, 3).reshape(-1, 3)
    rot = util.generate_healpix_grid(size=int(1e4))
    return rp, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--task", default="anyplace_multitask")
    ap.add_argument("--n-iters", type=int, default=50)
    a = ap.parse_args()

    print(f"[anyplace] loading model '{a.task}' ...", flush=True)
    model = build_model(a.task)
    raster_pts, rot_grid = build_grids()
    mc = Visualizer()  # stub no-op
    print("[anyplace] model + grids ready", flush=True)

    def _regress(parent, child):
        return multistep_regression_scene(
            mc, parent, child, None, model, None,
            scene_scale=SCENE_SCALE, scene_mean=SCENE_MEAN,
            grid_pts=raster_pts, rot_grid=rot_grid,
            viz=False, n_iters=a.n_iters, no_parent_crop=False,
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

    @torch.no_grad()
    def predict(inp):
        parent = inp["parent_pcd"].astype(np.float32)
        child = inp["child_pcd"].astype(np.float32)
        # AnyPlace runs LAST on the shared 5090; the earlier services (esp. IGR's Act-VH
        # optimisation) leave ~GBs of held CUDA cache, so free space here is razor-thin and
        # this call can hit CUDA OOM. empty_cache() first (reclaim our own leftovers); on OOM
        # empty_cache + retry once; always release at the end so we don't starve the next call.
        for attempt in (1, 2):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out_tf = _regress(parent, child)
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or attempt == 2:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                free = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
                print(f"[anyplace] CUDA OOM -> empty_cache + retry (free {free:.1f} GB)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"out_tf": np.asarray(out_tf, np.float32)}

    serve(predict, a.port, name="anyplace")


if __name__ == "__main__":
    main()
