"""REAL (RealSense) demo: live bbox-track a spherical fruit in the CAMERA frame.

Pipeline (per frame, all in the CAMERA frame -- no world extrinsic needed for a
bbox demo):

    RealSenseSource.capture()                       (BGR + planar Z-depth)
      -> RealSenseSegmenter.segment(center prompt)  (SAM2, or HSV fallback)
      -> deproject_zdepth(depth_m, mask, rs_intrin) (Z-depth, NOT euclidean)
      -> fit_ellipsoid_center(..., shape_class='spheroid', a=c=radius, R_wo=None)
      -> aabb_from_sphere(center, radius)           (3D AABB)
      -> project the 8 box corners back with K, draw the 2D rect + annotate.

The sphere radius is either fixed (``--radius``) or free-fit once from the first
good frame's camera-frame cloud via ``calibration.calibrate_object``. Annotated
PNGs + an MP4 + per-frame bbox JSON + a summary.json are written to ``--out``.

Pure numpy / cv2 / scipy / yaml + the in_hand_tracker package. No sim-side
(recorder / Isaac / omni) imports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from in_hand_tracker.io.realsense_source import (
        RealSenseSource,
        deproject_zdepth,
        project_points,
    )
    from in_hand_tracker.perception.realsense_segmenter import RealSenseSegmenter
    from in_hand_tracker.perception.bbox import aabb_from_sphere
    from in_hand_tracker.perception.ellipsoid_fit import fit_ellipsoid_center
    from in_hand_tracker.calibration.per_object_init import calibrate_object
else:
    from .io.realsense_source import (
        RealSenseSource,
        deproject_zdepth,
        project_points,
    )
    from .perception.realsense_segmenter import RealSenseSegmenter
    from .perception.bbox import aabb_from_sphere
    from .perception.ellipsoid_fit import fit_ellipsoid_center
    from .calibration.per_object_init import calibrate_object

_DEFAULT_ESTIMATOR_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "estimator.yaml"
)


def draw_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    box,
    K: np.ndarray,
    idx: int,
    center_depth: float,
    radius: float,
    method: str,
) -> np.ndarray:
    """Tint the mask, draw the projected 3D box's 2D bounding rect, annotate."""
    import cv2

    img = bgr.copy()
    if mask is not None and mask.any():
        overlay = img.copy()
        overlay[mask] = (0, 0, 255)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    if box is not None:
        # Project the 8 corners (camera frame -> pixels) and draw the 2D rect.
        pix = project_points(box.corners(), K)        # (8, 2), nan if behind cam
        good = np.isfinite(pix).all(axis=1)
        if good.sum() >= 2:
            pg = pix[good]
            u0, v0 = pg[:, 0].min(), pg[:, 1].min()
            u1, v1 = pg[:, 0].max(), pg[:, 1].max()
            cv2.rectangle(
                img,
                (int(round(u0)), int(round(v0))),
                (int(round(u1)), int(round(v1))),
                (0, 255, 0),
                2,
            )

    lines = [
        f"frame {idx}  [{method}]",
        f"depth {center_depth:.3f} m" if np.isfinite(center_depth) else "depth n/a",
        f"radius {radius * 100:.1f} cm" if np.isfinite(radius) else "radius n/a",
    ]
    y = 22
    for txt in lines:
        cv2.putText(
            img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        y += 26
    return img


def _fit_center_camframe(points_cam: np.ndarray, radius: float):
    """Axis-fixed SPHERE fit in the camera frame (R_wo=None -> object==camera)."""
    return fit_ellipsoid_center(
        points_cam,
        a=radius,
        c=radius,
        shape_class="spheroid",
        R_wo=None,
    )


def run_real_demo(
    frames: int = 30,
    out: str = "real_out",
    radius: Optional[float] = None,
    prompt: str = "center",
    estimator_yaml: str = _DEFAULT_ESTIMATOR_YAML,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: Optional[str] = None,
) -> dict:
    import cv2

    os.makedirs(out, exist_ok=True)

    segmenter = RealSenseSegmenter.from_yaml(estimator_yaml)
    source = RealSenseSource(width=width, height=height, fps=fps, serial=serial)
    intr = source.start()
    K = source.K
    print("[real_demo] intrinsics:", json.dumps(intr))
    if source.fallback_info is not None:
        print("[real_demo] PROFILE FALLBACK:", json.dumps(source.fallback_info))

    fitted_radius = float(radius) if radius is not None else None
    seg_methods = []
    coverage_flags = []
    radii = []
    fps_times = []
    writer = None
    frame_size = None

    try:
        for i in range(int(frames)):
            t0 = time.time()
            bgr, depth_m, rs_intrin = source.capture()
            if frame_size is None:
                frame_size = (bgr.shape[1], bgr.shape[0])

            seg = segmenter.segment(bgr)
            seg_methods.append(seg.method)
            mask = seg.mask
            n_mask = int(mask.sum())

            box = None
            center_depth = float("nan")
            frame_radius = float("nan")
            fit_ok = False
            n_points = 0

            if n_mask > 0:
                pts_cam = deproject_zdepth(depth_m, mask, rs_intrin)
                n_points = int(pts_cam.shape[0])
                if n_points >= 5:
                    # Calibrate the radius once from the first good frame.
                    if fitted_radius is None:
                        calib = calibrate_object(pts_cam)
                        if calib.ok and calib.a is not None:
                            # Spherical fruit: equatorial radius is the size.
                            fitted_radius = float(max(calib.a, calib.c))
                            print(
                                f"[real_demo] free-fit radius from frame {i}: "
                                f"{fitted_radius * 100:.1f} cm"
                            )
                    if fitted_radius is not None:
                        fit = _fit_center_camframe(pts_cam, fitted_radius)
                        if fit.ok and fit.center is not None:
                            fit_ok = True
                            frame_radius = fitted_radius
                            box = aabb_from_sphere(fit.center, fitted_radius)
                            center_depth = float(fit.center[2])

            coverage_flags.append(fit_ok)
            if np.isfinite(frame_radius):
                radii.append(frame_radius)

            img = draw_overlay(
                bgr, mask, box, K, i, center_depth, frame_radius, seg.method
            )
            cv2.imwrite(os.path.join(out, f"frame_{i:04d}.png"), img)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    os.path.join(out, "real_demo.mp4"), fourcc, float(fps),
                    frame_size,
                )
            writer.write(img)

            rec = {
                "idx": i,
                "method": seg.method,
                "sam2_score": seg.score,
                "n_mask_px": n_mask,
                "n_points": n_points,
                "fit_ok": fit_ok,
                "center": None if box is None else box.center.tolist(),
                "center_depth_m": (
                    None if not np.isfinite(center_depth) else center_depth
                ),
                "radius_m": (
                    None if not np.isfinite(frame_radius) else float(frame_radius)
                ),
                "bbox": None if box is None else box.to_dict(),
            }
            with open(os.path.join(out, f"bbox_{i:04d}.json"), "w") as f:
                json.dump(rec, f, indent=2)

            fps_times.append(time.time() - t0)
    finally:
        if writer is not None:
            writer.release()
        source.stop()

    n_cov = int(sum(coverage_flags))
    dt = np.asarray(fps_times, dtype=np.float64)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    summary = {
        "frames": int(frames),
        "n_fit_ok": n_cov,
        "coverage_pct": 100.0 * n_cov / max(1, int(frames)),
        "median_radius_m": (
            float(np.median(radii)) if radii else None
        ),
        "median_fps": float(1.0 / np.median(dt)) if dt.size else None,
        "segmenter_methods": {
            m: seg_methods.count(m) for m in sorted(set(seg_methods))
        },
        "intrinsics": intr,
        "profile_fallback": source.fallback_info,
        "out_dir": os.path.abspath(out),
        "video": os.path.abspath(os.path.join(out, "real_demo.mp4")),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live RealSense bbox demo for a spherical fruit (camera frame)."
    )
    p.add_argument("--frames", type=int, default=30, help="number of frames to capture")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--radius", type=float, default=None,
        help="fixed sphere radius (m); free-fit from frame 0 if omitted",
    )
    p.add_argument(
        "--prompt", default="center", choices=["center"],
        help="SAM2 prompt mode (default: image-center point)",
    )
    p.add_argument("--estimator-yaml", default=_DEFAULT_ESTIMATOR_YAML)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", default=None, help="RealSense serial (auto if omitted)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_real_demo(
        frames=args.frames,
        out=args.out,
        radius=args.radius,
        prompt=args.prompt,
        estimator_yaml=args.estimator_yaml,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
