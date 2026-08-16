# vision_pipeline — place pipeline (Current PC)

Implements [vision_pipeline_design.md](../vision_pipeline_design.md). A single
Current-PC orchestrator drives the arm via MoveIt and calls per-model HTTP
microservices (each in its own conda env). See the design doc for frames/steps.

## Layout
```
core/geometry.py     T_zalign, T_act, rank_upright (§9-A), local_crop_size, crop   [tested]
core/pointcloud.py   RGB-D -> world cloud back-projection                          [tested]
orchestrator.py      PlacePipeline.run(scenario, grasp)  — steps 1..4 / 3-1..3-23
interfaces           Backend + Models (duck-typed; see mock for the contract)
backends/mock.py     offline synthetic backend+models                             [tested]
backends/ros_backend.py  real ROS2 backend (MoveIt joint/pose goal, TF, RGB-D, hand)  [needs robot]
models_client.py     ModelsHTTP — wires orchestrator -> services
services/rpc.py      npz-over-HTTP (stdlib)
services/anyplace_service.py   :8801  parent+child_zalign -> out_tf(K,4,4)         [live ✓]
services/igr_service.py        :8816  refined partial -> dense + PaXini contacts;   [live ✓]
                                      mode=hand_pc -> PaXini-URDF-FK hand cloud
```

## Status
- ✅ **Offline flow** (`python -m vision_pipeline.test_flow`): both scenarios end-to-end with mocks.
- ✅ **Core math/PC** self-tests: `python -m vision_pipeline.core.geometry` / `.core.pointcloud`.
- ✅ **Model services** on the RTX 5090 (cu128): AnyPlace:8801, IGR completion:8816, DRO:8813 (CPU),
  SAM3:8811 (env `sam3`), Molmo:8810 (env `molmo`), grid:8815. Camera = 산업부-PC ROS topics (subscribed).
- Completion = **Act-VH IGR** (services/igr_service.py, with PaXini fingertip contacts). Outlier
  removal = **DBSCAN** keep-largest everywhere (core/outlier_removal.py). AdaPoinTr removed.
- ✅ **🎯 Full chain on REAL data** (`python -m vision_pipeline.validate_real`): real orange+tray →
  Molmo→SAM→world back-projection→IGR completion→AnyPlace → real T_act (no robot;
  offline extrinsic [core/extrinsic.py](core/extrinsic.py); capture [fixtures/scene.npz](fixtures/)).
- ⏳ **ROS2 backend** — needs the live robot/cameras (user tests): real arm motion, real hand q + live
  right_palm TF (placeholders offline), on-robot calibration (§9-C/E).

## Run (once all 5 services are up)
```bash
# all services at once (each in its conda env):  bash vision_pipeline/run_services.sh
# or individually, e.g. in env anyplace_cu128 (AnyPlace/IGR/DRO/capture/grid):
conda activate anyplace_cu128
python vision_pipeline/services/anyplace_service.py --port 8801 &
python vision_pipeline/services/igr_service.py      --port 8816 &
# ... molmo_service :8810, sam_service :8811, grid_service :8815  (camera = 산업부-PC ROS topics)

# orchestrator (in the dex_ros Humble container, ROS_DOMAIN_ID=9, digital-twin up):
#   python -m vision_pipeline.run scenario=fruit hand_pc=true
#   args: scenario=table|fruit  hand_pc=true|false (fuse the PaXini hand cloud)
#         complete=igr|sphere    backend=mock (offline)
```

## Test now (no robot/GPU needed for the first two)
```bash
python -m vision_pipeline.core.geometry
python -m vision_pipeline.core.pointcloud
python -m vision_pipeline.test_flow
# services (GPU): bash vision_pipeline/run_services.sh, then run.py / validate_real.py
```

Env note: AnyPlace + IGR + DRO + capture + grid share **one** `anyplace_cu128` env (torch 2.7.1+cu128).
Molmo (transformers 4.57.1) and SAM3 (py3.12, gated) need separate envs. See
[../ANYPLACE_PORT.md](../ANYPLACE_PORT.md), [../ADAPOINTR_PORT.md](../ADAPOINTR_PORT.md).
</content>
