"""Demo CLI (US-007, AC-8): run the in-hand tracker over a recorded sequence.

Pipeline (one shared code path with the evaluator):

    ReplicatorReader -> segmenter(source) -> deproject -> cam_to_world
        -> roi_gate.roi_crop -> ellipsoid_fit (center) -> bbox output policy

Calibration of (a, c) / shape_class happens ONCE (objects.yaml -> gt_shape.json
-> free-fit of a clean early frame). Per frame we solve ONLY the ellipsoid center
(the AC-10 graded quantity) and turn it into a 3D box. A box is emitted for 100%
of frames -- on any fit failure a conservative box is written so output is never
empty. The bbox center is the ellipsoid_fit center (NOT the diagnostic surface
centroid).

Usage:
    python -m in_hand_tracker.run_demo \
        --seq <seq_dir> --config-dir <config_dir> \
        --source gt|sam2 --out <out_dir> [--viz] [--max-frames N]

Outputs to ``--out``:
    bbox_<idx>.json   per-frame 3D box (center, half_extent, R, AABB) + meta
    summary.json      run-level summary (calibration, coverage, timings)
    viz/<idx>.png     (with --viz) a 2D overlay of the projected box

Pure numpy / cv2 / scipy / yaml. No sim-side imports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    # Allow direct script execution (``python in_hand_tracker/run_demo.py``) by
    # putting the package parent on sys.path and importing the absolute path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from in_hand_tracker.pipeline import Pipeline
else:
    from .pipeline import Pipeline


def _project_box_to_image(box, K, cam_pos, cam_quat_wxyz):
    """Project a Box3D's 8 corners to image pixels (returns (8, 2) int or None)."""
    from scipy.spatial.transform import Rotation

    corners_w = box.corners()
    q = np.asarray(cam_quat_wxyz, dtype=np.float64).reshape(4)
    R_wc = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    gl2cv = np.diag([1.0, -1.0, -1.0])
    # invert cam_to_world: world -> camera (OpenCV).
    rel = corners_w - np.asarray(cam_pos, dtype=np.float64).reshape(3)
    cam = (gl2cv @ (R_wc @ rel.T)).T          # (8, 3) OpenCV camera frame
    z = cam[:, 2]
    if np.any(z <= 1e-6):
        return None
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def _draw_overlay(rgb, mask, box, K, cam_pos, cam_quat_wxyz):
    import cv2

    img = rgb.copy()
    overlay = img.copy()
    overlay[mask] = (0, 0, 255)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    pix = _project_box_to_image(box, K, cam_pos, cam_quat_wxyz)
    if pix is not None:
        pts = np.round(pix).astype(int)
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),       # one face (x fixed)
            (4, 5), (4, 6), (5, 7), (6, 7),       # opposite face
            (0, 4), (1, 5), (2, 6), (3, 7),       # connectors
        ]
        for i, j in edges:
            cv2.line(img, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 2)
    return img


def run_demo(
    seq: str,
    config_dir: str,
    source: Optional[str] = None,
    out: str = "out",
    viz: bool = False,
    max_frames: Optional[int] = None,
) -> dict:
    pipe = Pipeline(seq, config_dir, source=source)
    calib = pipe.calibrate()

    os.makedirs(out, exist_ok=True)
    viz_dir = os.path.join(out, "viz")
    if viz:
        os.makedirs(viz_dir, exist_ok=True)

    n = len(pipe)
    if max_frames is not None:
        n = min(n, int(max_frames))

    n_ok = 0
    n_fallback = 0
    geometry_times = []

    for i in range(n):
        frame = pipe.reader.read_frame(i)
        mask = pipe.mask_for(i, max_frames=n)
        est = pipe.estimate_frame(i, mask=mask, frame=frame, time_geometry=True)

        if est.fit_ok:
            n_ok += 1
        if est.used_fallback:
            n_fallback += 1
        geometry_times.append(est.timings.get("geometry_total", float("nan")))

        rec = {
            "idx": est.idx,
            "frame_id": int(frame.idx),
            "fit_ok": est.fit_ok,
            "used_fallback": est.used_fallback,
            "n_mask_px": est.n_mask_px,
            "n_points_world": est.n_points_world,
            "n_points_roi": est.n_points_roi,
            "inlier_ratio": est.inlier_ratio,
            "fit_center": None if est.fit_center is None else est.fit_center.tolist(),
            "surface_centroid_diagnostic": (
                None if est.surface_centroid is None
                else np.asarray(est.surface_centroid).tolist()
            ),
            "gt_object_pos": est.gt_object_pos.tolist(),
            "bbox": est.box.to_dict(),
        }
        with open(os.path.join(out, f"bbox_{est.idx:06d}.json"), "w") as f:
            json.dump(rec, f, indent=2)

        if viz:
            import cv2

            img = _draw_overlay(
                frame.rgb, mask, est.box, pipe.K, frame.cam_pos, frame.cam_quat_wxyz
            )
            cv2.imwrite(os.path.join(viz_dir, f"{est.idx:06d}.png"), img)

    gt = np.asarray(geometry_times, dtype=np.float64)
    gt = gt[np.isfinite(gt)]
    summary = {
        "seq": os.path.abspath(seq),
        "fruit_class_name": pipe.fruit_class_name,
        "source": pipe.seg_cfg.source,
        "calibration": {
            "a": calib.a, "c": calib.c, "shape_class": calib.shape_class,
            "source": calib.source, "ca_ratio": calib.ca_ratio,
        },
        "n_frames": n,
        "n_bbox_emitted": n,                       # 100% by construction
        "coverage_pct": 100.0,
        "n_fit_ok": n_ok,
        "n_fallback": n_fallback,
        "geometry_median_hz": (
            float(1.0 / np.median(gt)) if gt.size and np.median(gt) > 0 else None
        ),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="In-hand tracker demo: per-frame 3D bbox over a recorded sequence."
    )
    p.add_argument("--seq", required=True, help="sequence dir (e.g. .../Replicator_02000)")
    p.add_argument("--config-dir", required=True, help="dir with camera/objects/estimator yaml")
    p.add_argument("--source", choices=["gt", "sam2"], default=None,
                   help="segmentation source (default: estimator.yaml)")
    p.add_argument("--out", default="out", help="output directory for per-frame bbox JSON")
    p.add_argument("--viz", action="store_true", help="write 2D box overlays under <out>/viz")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames processed")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_demo(
        seq=args.seq,
        config_dir=args.config_dir,
        source=args.source,
        out=args.out,
        viz=args.viz,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
