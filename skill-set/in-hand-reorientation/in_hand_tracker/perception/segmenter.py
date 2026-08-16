"""Object mask segmentation: a GT oracle and a SAM2 video tracker.

One interface (:class:`Segmenter`), two sources selected by
``estimator.yaml`` ``segmenter.source``:

  * ``"gt"``  — read the recorded colorized semantic-segmentation PNG and build
    the object mask by matching the class color (string-tuple parse + RGB->BGR
    swap, reusing the IO reader helpers). This is the bring-up path and the
    AC-9 oracle: a SAM2 mask is graded against it.
  * ``"sam2"`` — track the object across the sequence with the SAM2.1 video
    predictor. The first frame is *seeded with the exact GT mask* via
    ``predictor.add_new_mask`` and a single stable object id, then the mask is
    propagated forward through the video.

Clean-room: pure numpy / cv2 here. ``torch`` and ``sam2`` are imported lazily
(only when ``source == "sam2"``) and are provided by the host ``dexsdr`` env.
We NEVER import ``sam3`` (absent) or ``sam-3d-objects``, and we never import any
sim-side (recorder / Isaac) module.

Finger-outlier removal (:func:`remove_finger_outliers`) cleans the in-hand mask
by combining (a) mask-boundary erosion (drops the thin fingertip fringe) with
(b) depth-discontinuity rejection (drops pixels whose depth jumps away from the
robust object-surface median — typically the occluding fingers).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from ..io.replicator_reader import (
    ReplicatorReader,
    mask_from_seg_bgr,
    parse_color_string,
    rgb_to_bgr,
)

# Default config for the SAM2.1 hiera-tiny checkpoint. The hydra config name is
# resolved relative to the installed ``sam2`` package config tree.
#
# The *preferred* way to supply the checkpoint is ``estimator.yaml``
# (segmenter.sam2.ckpt). The fallback default below points to the checkpoint
# bundled at the repo root (``fruit-manipulation/sam2.1_hiera_tiny.pt``),
# resolved relative to this file so it carries no machine-specific home path.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_DEFAULT_SAM2_CKPT = os.path.join(_REPO_ROOT, "sam2.1_hiera_tiny.pt")
_DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
_STABLE_OBJ_ID = 1  # single stable object id for the whole sequence


# --------------------------------------------------------------------------- #
# Mask metrics + cleanup helpers
# --------------------------------------------------------------------------- #
def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks.

    Returns 1.0 when both masks are empty (vacuously identical), 0.0 when only
    one is empty. ``pred`` and ``gt`` must be the same shape.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return inter / union


def erode_mask_boundary(mask: np.ndarray, iters: int = 1, ksize: int = 3) -> np.ndarray:
    """Erode the mask boundary by ``iters`` rounds of a ``ksize`` square kernel.

    Removes the thin object/finger boundary fringe. ``iters <= 0`` returns the
    mask unchanged.
    """
    mask = np.asarray(mask, dtype=bool)
    if iters <= 0 or not mask.any():
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ksize), int(ksize)))
    eroded = cv2.erode(mask.astype(np.uint8), k, iterations=int(iters))
    return eroded.astype(bool)


def remove_finger_outliers(
    mask: np.ndarray,
    distance: np.ndarray,
    erode_iters: int = 2,
    erode_ksize: int = 3,
    depth_mad_k: float = 3.0,
    depth_abs_floor: float = 0.02,
    min_keep: int = 20,
) -> np.ndarray:
    """Clean an in-hand object mask of finger / boundary outliers.

    Two cues are combined:
      1. **Mask-boundary erosion** — ``erode_iters`` rounds of erosion peel off
         the thin fringe where the mask abuts the occluding fingers.
      2. **Depth-discontinuity rejection** — within the eroded mask, the robust
         object-surface depth is the median; pixels whose euclidean ray length
         deviates by more than ``depth_mad_k`` * (1.4826 * MAD) are dropped
         (fingers sit at a different depth than the held object surface).

    Args:
        mask: (H, W) bool object mask.
        distance: (H, W) float euclidean ray length (m), same shape as ``mask``.
        erode_iters: boundary-erosion rounds (0 disables erosion).
        erode_ksize: erosion structuring-element side.
        depth_mad_k: robust z-score gate on depth (0/inf disables the gate).
        depth_abs_floor: minimum band width (m). On a near-flat surface the MAD
            collapses to ~0; this floor still rejects pixels whose depth jumps
            by more than ``depth_abs_floor`` (a clear occluding finger) while
            tolerating sub-floor sensor noise. The effective band is
            ``max(depth_mad_k * 1.4826 * MAD, depth_abs_floor)``.
        min_keep: if the cleaned mask would fall below this many pixels, the
            depth gate is skipped (avoid erasing a small but valid object).

    Returns:
        (H, W) bool cleaned mask (always a subset of the eroded input mask).
    """
    mask = np.asarray(mask, dtype=bool)
    distance = np.asarray(distance, dtype=np.float64)
    if mask.shape != distance.shape:
        raise ValueError(
            f"mask {mask.shape} and distance {distance.shape} must match"
        )

    cleaned = erode_mask_boundary(mask, iters=erode_iters, ksize=erode_ksize)
    if not cleaned.any():
        return cleaned

    if depth_mad_k is None or not np.isfinite(depth_mad_k) or depth_mad_k <= 0:
        return cleaned

    d = distance[cleaned]
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() < 3:
        return cleaned
    med = float(np.median(d[valid]))
    mad = float(np.median(np.abs(d[valid] - med)))
    floor = max(0.0, float(depth_abs_floor))
    band = max(depth_mad_k * 1.4826 * mad, floor)
    if band <= 0:
        return cleaned

    out = cleaned.copy()
    vs, us = np.nonzero(cleaned)
    dv = distance[vs, us].astype(np.float64)
    reject = (~np.isfinite(dv)) | (np.abs(dv - med) > band)
    if reject.any():
        cand = out.copy()
        cand[vs[reject], us[reject]] = False
        # Guard against erasing a small-but-valid object.
        if int(cand.sum()) >= int(min_keep):
            out = cand
    return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SegmenterConfig:
    """Resolved segmenter configuration (subset of ``estimator.yaml``)."""

    source: str = "gt"                       # "gt" | "sam2"
    fruit_class_name: str = "apple"
    color_tolerance: int = 0                 # BGR per-channel tol for GT match
    reseg_every_n: int = 30                  # SAM2 re-seed cadence (frames)
    sam2_ckpt: str = _DEFAULT_SAM2_CKPT
    sam2_config: str = _DEFAULT_SAM2_CONFIG
    device: str = "cuda"
    # Finger-outlier removal knobs (applied on request, see Segmenter.clean()).
    erode_iters: int = 2
    erode_ksize: int = 3
    depth_mad_k: float = 3.0
    depth_abs_floor: float = 0.02

    @classmethod
    def from_yaml(
        cls,
        estimator_yaml: Union[str, dict],
        fruit_class_name: Optional[str] = None,
    ) -> "SegmenterConfig":
        """Build from an ``estimator.yaml`` path (or already-loaded dict)."""
        if isinstance(estimator_yaml, dict):
            cfg = estimator_yaml
        else:
            import yaml

            with open(estimator_yaml) as f:
                cfg = yaml.safe_load(f) or {}
        seg = dict(cfg.get("segmenter", {}) or {})
        sam2 = dict(seg.get("sam2", {}) or {})
        out = cls()
        out.source = str(seg.get("source", out.source))
        out.reseg_every_n = int(seg.get("reseg_every_n", out.reseg_every_n))
        out.color_tolerance = int(sam2.get("color_tolerance", out.color_tolerance))
        out.sam2_ckpt = str(sam2.get("ckpt", out.sam2_ckpt))
        if "config" in sam2:
            out.sam2_config = str(sam2["config"])
        if "device" in sam2:
            out.device = str(sam2["device"])
        if fruit_class_name is not None:
            out.fruit_class_name = str(fruit_class_name)
        return out


# --------------------------------------------------------------------------- #
# Segmenter
# --------------------------------------------------------------------------- #
class Segmenter:
    """Produce per-frame object masks from a recorded sequence.

    Args:
        config: a :class:`SegmenterConfig` (or call :meth:`from_yaml`).

    The GT path is stateless and works frame-by-frame (:meth:`segment_frame`).
    The SAM2 path tracks the whole sequence: call :meth:`run_sequence` (it seeds
    frame 0 with the exact GT mask, then propagates) and returns ``{idx: mask}``.
    """

    def __init__(self, config: Optional[SegmenterConfig] = None) -> None:
        self.config = config or SegmenterConfig()
        if self.config.source not in ("gt", "sam2"):
            raise ValueError(
                f"segmenter.source must be 'gt' or 'sam2'; got {self.config.source!r}"
            )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(
        cls,
        estimator_yaml: Union[str, dict],
        fruit_class_name: Optional[str] = None,
    ) -> "Segmenter":
        return cls(SegmenterConfig.from_yaml(estimator_yaml, fruit_class_name))

    # ------------------------------------------------------------------ #
    # GT-color mask (also the SAM2 seed + AC-9 oracle).
    # ------------------------------------------------------------------ #
    def gt_mask(self, reader: ReplicatorReader, idx: int) -> np.ndarray:
        """Object mask for frame ``idx`` from the recorded colorized seg PNG.

        Resolves the class color from the per-frame sidecar
        (``segmentation_labels_<fid>.json``), parses the RGB string, swaps to
        BGR, and matches the cv2-loaded seg PNG within ``color_tolerance``.
        """
        return reader.read_frame(idx).object_mask

    def gt_mask_from_seg(
        self, seg_bgr: np.ndarray, color_str: str
    ) -> np.ndarray:
        """Build a GT mask directly from a BGR seg image + RGB color string.

        ``color_str`` is the sidecar value like ``"(128, 0, 0)"`` (RGB). This is
        the same string-parse -> RGB->BGR -> tolerance-match path used by the IO
        reader, exposed for callers holding the seg image already.
        """
        target_bgr = rgb_to_bgr(parse_color_string(color_str))
        return mask_from_seg_bgr(
            seg_bgr, target_bgr, tol=self.config.color_tolerance
        )

    # ------------------------------------------------------------------ #
    # Single-frame interface.
    # ------------------------------------------------------------------ #
    def segment_frame(self, reader: ReplicatorReader, idx: int) -> np.ndarray:
        """Return the object mask for frame ``idx``.

        For ``source == "gt"`` this is the recorded mask. For ``source ==
        "sam2"`` a single frame cannot be tracked in isolation; callers should
        use :meth:`run_sequence`. As a convenience we fall back to the GT mask
        for frame 0 (the SAM2 seed) and otherwise raise.
        """
        if self.config.source == "gt":
            return self.gt_mask(reader, idx)
        # SAM2: a single isolated frame is the seed only.
        if idx == 0:
            return self.gt_mask(reader, idx)
        raise RuntimeError(
            "SAM2 source tracks the whole sequence; call run_sequence() "
            "instead of segment_frame() for idx > 0."
        )

    # ------------------------------------------------------------------ #
    # Whole-sequence interface.
    # ------------------------------------------------------------------ #
    def run_sequence(
        self,
        reader: ReplicatorReader,
        max_frames: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Return ``{frame_idx: bool mask}`` for the sequence.

        GT: per-frame recorded masks. SAM2: seed frame 0 with the exact GT mask
        (``add_new_mask``) under a single stable object id, then propagate
        forward through the video.
        """
        n = len(reader)
        if max_frames is not None:
            n = min(n, int(max_frames))
        if self.config.source == "gt":
            return {i: self.gt_mask(reader, i) for i in range(n)}
        return self._run_sam2(reader, n)

    # ------------------------------------------------------------------ #
    # SAM2 video tracking.
    # ------------------------------------------------------------------ #
    def _build_sam2_predictor(self):
        """Construct the SAM2 video predictor (lazy heavy imports)."""
        import torch  # noqa: F401  (presence + device check)
        from sam2.build_sam import build_sam2_video_predictor

        ckpt = self.config.sam2_ckpt
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")
        predictor = build_sam2_video_predictor(
            self.config.sam2_config,
            ckpt,
            device=self.config.device,
        )
        return predictor

    def _run_sam2(self, reader: ReplicatorReader, n: int) -> Dict[int, np.ndarray]:
        import torch
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401

        predictor = self._build_sam2_predictor()

        # SAM2's video predictor consumes a directory of "<int>.jpg" frames.
        # Dump the sequence RGB (converted BGR->RGB) to a temp dir; resolution
        # and frame order are preserved (sorted by integer stem).
        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmp:
            h = w = None
            for i in range(n):
                frame = reader.read_frame(i)
                rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB)
                if i == 0:
                    h, w = rgb.shape[:2]
                cv2.imwrite(
                    os.path.join(tmp, f"{i:05d}.jpg"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )

            inference_state = predictor.init_state(video_path=tmp)
            predictor.reset_state(inference_state)

            # Seed frame 0 with the EXACT GT mask under one stable object id.
            seed_mask = self.gt_mask(reader, 0).astype(bool)
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=_STABLE_OBJ_ID,
                mask=seed_mask,
            )

            masks: Dict[int, np.ndarray] = {}
            for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
                inference_state
            ):
                # Pick the stable object id's channel and threshold logits > 0.
                obj_pos = out_obj_ids.index(_STABLE_OBJ_ID)
                logit = out_logits[obj_pos]
                m = (logit > 0.0).squeeze().detach().cpu().numpy().astype(bool)
                if m.shape != (h, w):
                    m = cv2.resize(
                        m.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                masks[out_idx] = m

        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return masks

    # ------------------------------------------------------------------ #
    # Cleanup convenience.
    # ------------------------------------------------------------------ #
    def clean(self, mask: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Apply the configured finger-outlier removal to a mask."""
        return remove_finger_outliers(
            mask,
            distance,
            erode_iters=self.config.erode_iters,
            erode_ksize=self.config.erode_ksize,
            depth_mad_k=self.config.depth_mad_k,
            depth_abs_floor=self.config.depth_abs_floor,
        )
