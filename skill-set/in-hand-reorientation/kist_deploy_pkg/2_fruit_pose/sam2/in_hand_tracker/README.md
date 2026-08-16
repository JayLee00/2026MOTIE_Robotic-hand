# in_hand_tracker

Standalone in-hand object tracking foundation (M1~M2): sequence IO +
deprojection. Clean-room reimplementation (numpy / cv2 / open3d / scipy) of the
sim perception stack; no sim-side imports.

## Layout

- `config/` — `camera.yaml` (intrinsics), `objects.yaml` (per-object shape map),
  `estimator.yaml` (RANSAC / ROI / segmenter knobs).
- `io/` — `types.py` (`Frame`, `GtShape`, `FitResult`), `replicator_reader.py`
  (sequence reader), `deproject.py` (euclidean deprojection + camera->world).
- `tests/` — structure, boundary (no-leak), and IO/geometry tests.

## Run tests

```bash
conda run -n dexsdr --no-capture-output python -m pytest in_hand_tracker/tests/ -x -q
```

torch / sam2 / pinocchio are optional and provided by the host `dexsdr` env.
