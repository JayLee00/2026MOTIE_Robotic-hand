"""Shared per-frame estimation pipeline (US-007 wiring).

Wires the on-disk modules into one geometry path so both the demo CLI
(``run_demo.py``) and the evaluator (``eval/evaluate.py``) call the SAME code:

    ReplicatorReader  ->  segmenter(source)  ->  deproject (cam frame)
        ->  cam_to_world  ->  roi_gate.roi_crop  ->  ellipsoid_fit (center only)
        ->  bbox output policy

Calibration of ``(a, c)`` / ``shape_class`` happens ONCE: from objects.yaml if an
entry exists, else from gt_shape.json when present, else by fitting a clean early
frame (``calibrate_object``). Per frame we then solve ONLY the ellipsoid center
(the AC-10 graded quantity) and turn it into a 3D box. A box is emitted for 100%
of frames -- on any fit failure we fall back to a conservative box centered on the
ROI hint / surface centroid so the output is never empty.

The diagnostic ``surface_centroid`` is computed and carried for the AC-3b sanity
check ONLY; it never feeds the fit-center / bbox-center path.

Pure numpy / cv2 / scipy / yaml. No sim-side imports.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .io.deproject import (
    cam_to_world,
    deproject,
    intrinsics_from_yaml,
    surface_centroid,
)
from .io.replicator_reader import ReplicatorReader
from .perception.bbox import Box3D, box_from_fit, conservative_aabb
from .perception.ellipsoid_fit import fit_ellipsoid_center
from .perception.roi_gate import roi_crop
from .perception.segmenter import Segmenter, SegmenterConfig


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_yaml(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class Calibration:
    """The once-per-object shape estimate that gates the per-frame center fit."""

    a: float
    c: float
    shape_class: str
    source: str                       # "objects.yaml" | "gt_shape.json" | "fit"
    center0: Optional[np.ndarray] = None
    ca_ratio: Optional[float] = None


@dataclass
class FrameEstimate:
    """Per-frame pipeline output."""

    idx: int
    n_mask_px: int
    n_points_world: int
    n_points_roi: int
    fit_ok: bool
    fit_center: Optional[np.ndarray]          # (3,) ellipsoid_fit center (AC-10)
    surface_centroid: Optional[np.ndarray]    # (3,) DIAGNOSTIC (AC-3b)
    inlier_ratio: Optional[float]
    box: Box3D                                # never None (conservative fallback)
    gt_object_pos: np.ndarray
    used_fallback: bool = False
    timings: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    """End-to-end geometry pipeline for one recorded sequence.

    Args:
        seq_dir: sequence directory (e.g. .../Replicator_02000).
        config_dir: directory holding camera.yaml / objects.yaml / estimator.yaml.
        source: segmentation source override ("gt" | "sam2"); falls back to the
            estimator.yaml value when None.
        fruit_class_name: semantic class to segment. When None it is auto-detected
            from the first frame's sidecar (the single "object" class).
    """

    def __init__(
        self,
        seq_dir: str,
        config_dir: str,
        source: Optional[str] = None,
        fruit_class_name: Optional[str] = None,
    ) -> None:
        self.seq_dir = os.path.abspath(seq_dir)
        self.config_dir = os.path.abspath(config_dir)

        self.camera_yaml = os.path.join(self.config_dir, "camera.yaml")
        self.objects_yaml = os.path.join(self.config_dir, "objects.yaml")
        self.estimator_yaml = os.path.join(self.config_dir, "estimator.yaml")

        self.estimator_cfg = load_yaml(self.estimator_yaml)
        self.objects_cfg = load_yaml(self.objects_yaml)
        self.K = intrinsics_from_yaml(self.camera_yaml)

        self.fruit_class_name = (
            fruit_class_name
            if fruit_class_name is not None
            else self._auto_class_name()
        )

        seg_cfg = SegmenterConfig.from_yaml(self.estimator_cfg, self.fruit_class_name)
        if source is not None:
            seg_cfg.source = str(source)
        self.seg_cfg = seg_cfg
        self.segmenter = Segmenter(seg_cfg)

        self.reader = ReplicatorReader(
            self.seq_dir,
            camera_yaml=self.camera_yaml,
            fruit_class_name=self.fruit_class_name,
            color_tolerance=int(seg_cfg.color_tolerance),
        )

        self.ca_threshold = float(self.estimator_cfg.get("ca_threshold", 1.2))
        ransac = dict(self.estimator_cfg.get("ransac", {}) or {})
        self.ransac_iters = int(ransac.get("iters", 200))
        self.inlier_thresh = float(ransac.get("dist_thresh", 0.004))
        self.min_inlier_ratio = float(ransac.get("min_inlier_ratio", 0.6))

        # Camera->world convention. The recorded datasets disagree (original
        # soldering -> "transposed"; regenerated fruit -> "literal"), so default
        # to "auto": detect once per sequence from the GT object pose.
        self.convention_cfg = str(
            (self.estimator_cfg.get("extrinsic", {}) or {}).get("convention", "auto")
        )
        self._resolved_convention: Optional[str] = (
            self.convention_cfg
            if self.convention_cfg in ("literal", "transposed")
            else None
        )

        self._masks_cache: Optional[Dict[int, np.ndarray]] = None
        self.calibration: Optional[Calibration] = None

    # ------------------------------------------------------------------ #
    def _auto_class_name(self) -> str:
        """Detect the single 'object' semantic class from frame 0's sidecar."""
        fid = self.reader_frame_ids()[0]
        path = os.path.join(
            self.seq_dir, "semantic_segmentation",
            f"segmentation_labels_{fid}.json",
        )
        if os.path.isfile(path):
            import json

            with open(path) as f:
                labels = json.load(f)
            for name, entry in labels.items():
                if entry.get("class") == "object":
                    return str(name)
            if labels:
                return str(next(iter(labels)))
        return "apple"

    def reader_frame_ids(self) -> List[str]:
        import glob

        ids = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self.seq_dir, "rgb", "*.png"))
        )
        if not ids:
            raise FileNotFoundError(f"no rgb frames under {self.seq_dir}")
        return ids

    def __len__(self) -> int:
        return len(self.reader)

    # ------------------------------------------------------------------ #
    # Segmentation (cache the sam2 sequence run; gt is per-frame).
    # ------------------------------------------------------------------ #
    def mask_for(self, idx: int, max_frames: Optional[int] = None) -> np.ndarray:
        if self.seg_cfg.source == "gt":
            return self.segmenter.gt_mask(self.reader, idx)
        if self._masks_cache is None:
            self._masks_cache = self.segmenter.run_sequence(
                self.reader, max_frames=max_frames
            )
        m = self._masks_cache.get(idx)
        if m is None:
            m = self.segmenter.gt_mask(self.reader, idx)
        return m

    # ------------------------------------------------------------------ #
    # Calibration (once).
    # ------------------------------------------------------------------ #
    def _hints_for(self, frame) -> dict:
        return {
            "gt_object_pos": frame.gt_object_pos,
            "cam_pos": frame.cam_pos,
            "hand_pos": frame.hand_pos,
            "hand_quat_wxyz": frame.hand_quat_wxyz,
            "hand_joints": frame.hand_joints,
            "hand_joint_names": frame.hand_joint_names,
        }

    def get_convention(self) -> str:
        """Resolve the cam->world convention once per sequence.

        "literal"/"transposed" from estimator.yaml are used as-is; "auto"
        detects from the largest-mask early frame using the GT object pose
        (the two conventions differ by ~0.8 m, so the choice is unambiguous).
        Falls back to "transposed" only when no usable GT-masked frame exists.
        """
        if self._resolved_convention is not None:
            return self._resolved_convention
        from .io.deproject import detect_convention

        best = None
        n_probe = min(5, len(self.reader))
        for i in range(n_probe):
            frame = self.reader.read_frame(i)
            mask = self.segmenter.gt_mask(self.reader, i)
            n = int(np.asarray(mask, dtype=bool).sum())
            if n < 50 or not np.all(np.isfinite(frame.gt_object_pos)):
                continue
            if best is None or n > best[0]:
                best = (n, frame, mask)
        if best is None:
            self._resolved_convention = "transposed"
            return self._resolved_convention
        _, frame, mask = best
        pc = deproject(frame.distance, mask, self.K)
        conv, _dist = detect_convention(
            pc, frame.cam_pos, frame.cam_quat_wxyz, frame.gt_object_pos
        )
        self._resolved_convention = conv
        return conv

    def world_points(self, frame, mask: np.ndarray) -> np.ndarray:
        pc = deproject(frame.distance, mask, self.K)
        return cam_to_world(
            pc, frame.cam_pos, frame.cam_quat_wxyz, convention=self.get_convention()
        )

    def calibrate(self) -> Calibration:
        """Resolve (a, c, shape_class) ONCE, by precedence.

        objects.yaml entry -> gt_shape.json -> free-fit of a clean early frame.
        """
        if self.calibration is not None:
            return self.calibration

        # 1) objects.yaml entry for this class.
        objects = (self.objects_cfg or {}).get("objects") or {}
        entry = objects.get(self.fruit_class_name)
        if isinstance(entry, dict) and "a" in entry and "c" in entry:
            a = float(entry["a"])
            c = float(entry["c"])
            sc = str(entry.get("shape_class", "spheroid"))
            self.calibration = Calibration(a, c, sc, source="objects.yaml")
            return self.calibration

        # 2) gt_shape.json (fruit sequences only). a/c are ABSOLUTE radii;
        #    scale is metadata only and must NOT be multiplied in.
        if self.reader.gt_shape is not None:
            gs = self.reader.gt_shape
            a = float(gs.a)
            c = float(gs.c)
            from .calibration.per_object_init import classify_shape

            sc = classify_shape(a, c, self.ca_threshold)
            self.calibration = Calibration(a, c, sc, source="gt_shape.json")
            return self.calibration

        # 3) Free-fit a clean early frame (largest-mask of the first few frames).
        from .calibration.per_object_init import calibrate_object

        best = None
        n_probe = min(5, len(self.reader))
        for i in range(n_probe):
            frame = self.reader.read_frame(i)
            mask = self.mask_for(i, max_frames=n_probe)
            n = int(np.asarray(mask, dtype=bool).sum())
            if best is None or n > best[0]:
                best = (n, i, frame, mask)
        if best is None:
            self.calibration = Calibration(0.04, 0.04, "spheroid", source="default")
            return self.calibration

        _, i0, frame0, mask0 = best
        pw = self.world_points(frame0, mask0)
        fit = calibrate_object(pw, self.ca_threshold)
        if fit.ok and fit.a and fit.c:
            self.calibration = Calibration(
                float(fit.a), float(fit.c), str(fit.shape_class),
                source="fit", center0=fit.center, ca_ratio=fit.ca_ratio,
            )
        else:
            # Fall back to an isotropic estimate from the point-cloud spread.
            if pw.shape[0] >= 4:
                ctr = surface_centroid(pw)
                r = float(np.median(np.linalg.norm(pw - ctr, axis=1)))
                r = max(r, 0.01)
            else:
                ctr, r = None, 0.04
            self.calibration = Calibration(
                r, r, "spheroid", source="fit-fallback", center0=ctr,
            )
        return self.calibration

    # ------------------------------------------------------------------ #
    # Per-frame estimate (the AC-12 geometry path: deproject+roi+fit+bbox).
    # ------------------------------------------------------------------ #
    def estimate_frame(
        self,
        idx: int,
        mask: Optional[np.ndarray] = None,
        frame=None,
        time_geometry: bool = False,
    ) -> FrameEstimate:
        """Run the geometry path for one frame and emit a guaranteed box.

        When ``mask`` is supplied (e.g. a precomputed SAM2 mask) and ``frame`` is
        supplied, no IO is done here -- so the timed region is purely
        deproject+roi+fit+bbox (AC-12 EXCLUDES segmentation).
        """
        calib = self.calibrate()
        self.get_convention()  # resolve once, BEFORE the AC-12 timed region
        if frame is None:
            frame = self.reader.read_frame(idx)
        if mask is None:
            mask = self.mask_for(idx)
        mask = np.asarray(mask, dtype=bool)

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # deproject + cam->world (convention resolved once per sequence).
        pc = deproject(frame.distance, mask, self.K)
        pw = cam_to_world(
            pc, frame.cam_pos, frame.cam_quat_wxyz, convention=self.get_convention()
        )
        t_dep = time.perf_counter()

        # ROI crop (world frame; hint = GT object region for "fixed").
        roi = roi_crop(pw, mask=mask, cfg=self.estimator_cfg, hints=self._hints_for(frame))
        pts = roi.points
        t_roi = time.perf_counter()

        # Diagnostic surface centroid (AC-3b) -- NEVER feeds the fit center.
        diag_centroid = surface_centroid(pts) if pts.shape[0] else surface_centroid(pw)

        # Ellipsoid center fit (AC-10). Center only.
        fit_center = None
        inlier_ratio = None
        fit_ok = False
        major_axis = None
        if pts.shape[0] >= 3:
            fit = fit_ellipsoid_center(
                pts,
                a=calib.a,
                c=calib.c,
                shape_class=calib.shape_class,
                R_wo=None,
                n_iterations=self.ransac_iters,
                inlier_ratio_target=self.min_inlier_ratio,
            )
            if fit.ok and fit.center is not None and np.all(np.isfinite(fit.center)):
                fit_center = np.asarray(fit.center, dtype=np.float64)
                inlier_ratio = fit.inlier_ratio
                major_axis = fit.major_axis
                fit_ok = True
        t_fit = time.perf_counter()

        # bbox output policy (never empty).
        used_fallback = False
        if fit_ok:
            box = box_from_fit(
                fit_center, calib.a, calib.c, calib.shape_class, major_axis
            )
        else:
            used_fallback = True
            # Conservative box centered on the best available hint.
            if diag_centroid is not None and np.all(np.isfinite(diag_centroid)):
                ctr = diag_centroid
            elif np.all(np.isfinite(frame.gt_object_pos)):
                ctr = frame.gt_object_pos
            else:
                ctr = roi.center
            box = conservative_aabb(ctr, calib.a, calib.c)
        t_box = time.perf_counter()

        if time_geometry:
            timings = {
                "deproject": t_dep - t0,
                "roi": t_roi - t_dep,
                "fit": t_fit - t_roi,
                "bbox": t_box - t_fit,
                "geometry_total": t_box - t0,
            }

        return FrameEstimate(
            idx=int(idx),
            n_mask_px=int(mask.sum()),
            n_points_world=int(pw.shape[0]),
            n_points_roi=int(pts.shape[0]),
            fit_ok=fit_ok,
            fit_center=fit_center,
            surface_centroid=diag_centroid,
            inlier_ratio=inlier_ratio,
            box=box,
            gt_object_pos=np.asarray(frame.gt_object_pos, dtype=np.float64),
            used_fallback=used_fallback,
            timings=timings,
        )
