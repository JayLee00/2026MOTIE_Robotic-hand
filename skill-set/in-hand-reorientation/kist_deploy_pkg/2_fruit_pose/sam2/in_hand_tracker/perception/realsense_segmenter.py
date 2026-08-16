"""GT-less, fruit-agnostic segmentation for the REAL (live) RGB-D path.

Two paths behind one interface (:class:`RealSenseSegmenter`):

  * **Primary -- SAM2 image predictor with a center-point prompt.** No GT mask
    exists for a live frame, so instead of seeding with a recorded mask we
    prompt the SAM2.1 *image* predictor with the image-center point (label 1 =
    foreground) and keep the highest-scoring mask. Extra prompt points may be
    supplied. The checkpoint path is read from ``estimator.yaml``
    (``segmenter.sam2.ckpt``) -- the bundled ``sam2.1_hiera_tiny.pt``.
  * **Fallback -- HSV color blob.** If SAM2 import / checkpoint resolution
    fails, segment the largest saturated/value-thresholded connected component
    near the image center. Logged so the demo can report which path ran.

We NEVER import ``sam3`` or ``sam-3d-objects``, and never import any sim-side
(recorder / Isaac / omni) module. ``torch`` and ``sam2`` are imported lazily so
the module stays import-safe without them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RealSenseSegmenter",
    "RealSenseSegmenterConfig",
    "hsv_center_blob",
]


# Reuse the SAM2 checkpoint/config defaults from the (sibling) sequence
# segmenter so the package source carries no verbatim sim-stack path literal.
from .segmenter import _DEFAULT_SAM2_CKPT, _DEFAULT_SAM2_CONFIG  # noqa: E402


# --------------------------------------------------------------------------- #
# HSV fallback
# --------------------------------------------------------------------------- #
def hsv_center_blob(
    bgr: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    sat_min: int = 60,
    val_min: int = 40,
) -> np.ndarray:
    """Largest saturated connected blob nearest the image center (bool mask).

    Thresholds saturation and value (drops the dim / desaturated background),
    finds connected components, and keeps the one whose centroid is closest to
    ``center`` (defaults to the image center). Returns an all-False mask if no
    component clears the threshold.

    Args:
        bgr: (H, W, 3) uint8 BGR image.
        center: (cx, cy) pixel to bias toward; defaults to image center.
        sat_min: minimum HSV saturation to be considered foreground.
        val_min: minimum HSV value to be considered foreground.
    """
    bgr = np.asarray(bgr)
    h, w = bgr.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    cx, cy = float(center[0]), float(center[1])

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    fg = (hsv[:, :, 1] >= int(sat_min)) & (hsv[:, :, 2] >= int(val_min))
    fg_u8 = fg.astype(np.uint8)
    # Close small holes so a textured fruit forms one blob.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, k, iterations=2)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_u8, connectivity=8
    )
    if n_labels <= 1:
        return np.zeros((h, w), dtype=bool)

    best_label = -1
    best_score = np.inf
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 50:                          # ignore speckle
            continue
        ccx, ccy = centroids[lbl]
        dist = (ccx - cx) ** 2 + (ccy - cy) ** 2
        # Prefer near-center; break ties toward larger area.
        score = dist / max(area, 1)
        if score < best_score:
            best_score = score
            best_label = lbl
    if best_label < 0:
        return np.zeros((h, w), dtype=bool)
    return labels == best_label


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class RealSenseSegmenterConfig:
    """Resolved live-segmenter config (subset of ``estimator.yaml``)."""

    sam2_ckpt: str = _DEFAULT_SAM2_CKPT
    sam2_config: str = _DEFAULT_SAM2_CONFIG
    device: str = "cuda"
    sat_min: int = 60
    val_min: int = 40

    @classmethod
    def from_yaml(
        cls, estimator_yaml: Union[str, dict]
    ) -> "RealSenseSegmenterConfig":
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
        out.sam2_ckpt = str(sam2.get("ckpt", out.sam2_ckpt))
        if "config" in sam2:
            out.sam2_config = str(sam2["config"])
        if "device" in sam2:
            out.device = str(sam2["device"])
        return out


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SegResult:
    """A single live segmentation result."""

    mask: np.ndarray                       # (H, W) bool
    method: str                            # "sam2" | "hsv"
    score: Optional[float] = None          # SAM2 mask score (None for HSV)
    prompt_points: List[Tuple[int, int]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Segmenter
# --------------------------------------------------------------------------- #
class RealSenseSegmenter:
    """Segment the in-hand object from a single live RGB frame (GT-less).

    Lazily builds the SAM2 image predictor on first use. If construction fails
    (import / checkpoint / device), it permanently switches to the HSV fallback
    and logs the reason; subsequent calls go straight to HSV.
    """

    def __init__(self, config: Optional[RealSenseSegmenterConfig] = None) -> None:
        self.config = config or RealSenseSegmenterConfig()
        self._predictor = None
        self._sam2_failed = False
        self._sam2_fail_reason: Optional[str] = None

    @classmethod
    def from_yaml(cls, estimator_yaml: Union[str, dict]) -> "RealSenseSegmenter":
        return cls(RealSenseSegmenterConfig.from_yaml(estimator_yaml))

    # ------------------------------------------------------------------ #
    def _build_predictor(self):
        """Construct the SAM2 image predictor (lazy heavy imports)."""
        import torch  # noqa: F401  (presence + device check)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt = self.config.sam2_ckpt
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")
        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        sam2_model = build_sam2(self.config.sam2_config, ckpt, device=device)
        return SAM2ImagePredictor(sam2_model)

    def _ensure_predictor(self):
        if self._predictor is not None or self._sam2_failed:
            return
        try:
            self._predictor = self._build_predictor()
        except Exception as exc:
            self._sam2_failed = True
            self._sam2_fail_reason = str(exc)
            logger.warning(
                "SAM2 image predictor unavailable (%s); using HSV fallback.", exc
            )

    @property
    def sam2_available(self) -> bool:
        """Whether the SAM2 predictor is usable (builds it lazily)."""
        self._ensure_predictor()
        return self._predictor is not None

    @property
    def sam2_fail_reason(self) -> Optional[str]:
        return self._sam2_fail_reason

    # ------------------------------------------------------------------ #
    def segment(
        self,
        bgr: np.ndarray,
        prompt_points: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> SegResult:
        """Segment the object from one BGR frame.

        Args:
            bgr: (H, W, 3) uint8 BGR color image.
            prompt_points: optional extra (u, v) foreground points. The image
                center point is always prepended as the primary prompt.

        Returns:
            A :class:`SegResult` (``method`` records whether SAM2 or the HSV
            fallback produced the mask).
        """
        bgr = np.asarray(bgr)
        h, w = bgr.shape[:2]
        center = (w // 2, h // 2)
        points: List[Tuple[int, int]] = [center]
        if prompt_points:
            points.extend((int(u), int(v)) for (u, v) in prompt_points)

        self._ensure_predictor()
        if self._predictor is not None:
            try:
                return self._segment_sam2(bgr, points)
            except Exception as exc:
                logger.warning(
                    "SAM2 predict failed (%s); falling back to HSV.", exc
                )
                self._sam2_failed = True
                self._sam2_fail_reason = str(exc)

        mask = hsv_center_blob(
            bgr,
            center=center,
            sat_min=self.config.sat_min,
            val_min=self.config.val_min,
        )
        return SegResult(mask=mask, method="hsv", score=None, prompt_points=points)

    def segment_at(
        self,
        bgr: np.ndarray,
        points: Sequence[Tuple[int, int]],
    ) -> SegResult:
        """Segment prompted ONLY at the given (u, v) foreground points.

        Unlike :meth:`segment`, the image center is NOT prepended -- use this for
        tracking, where the prompt should follow the object (e.g. the reprojected
        filtered centre or a user click), not the frame centre. Falls back to an
        HSV blob near the first point if SAM2 is unavailable.
        """
        bgr = np.asarray(bgr)
        h, w = bgr.shape[:2]
        pts: List[Tuple[int, int]] = (
            [(int(u), int(v)) for (u, v) in points] if points else [(w // 2, h // 2)]
        )
        self._ensure_predictor()
        if self._predictor is not None:
            try:
                return self._segment_sam2(bgr, pts)
            except Exception as exc:
                logger.warning("SAM2 predict failed (%s); falling back to HSV.", exc)
                self._sam2_failed = True
                self._sam2_fail_reason = str(exc)
        mask = hsv_center_blob(
            bgr, center=pts[0], sat_min=self.config.sat_min, val_min=self.config.val_min
        )
        return SegResult(mask=mask, method="hsv", score=None, prompt_points=pts)

    def _segment_sam2(
        self, bgr: np.ndarray, points: List[Tuple[int, int]]
    ) -> SegResult:
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.ones((len(points),), dtype=np.int32)

        autocast_dev = "cuda" if self.config.device == "cuda" else "cpu"
        with torch.inference_mode():
            self._predictor.set_image(rgb)
            with torch.autocast(autocast_dev, dtype=torch.bfloat16) if (
                autocast_dev == "cuda"
            ) else _nullcontext():
                masks, scores, _ = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

        # Pick the highest-scoring mask.
        best = int(np.argmax(scores))
        mask = np.asarray(masks[best]).astype(bool)
        if mask.shape != (bgr.shape[0], bgr.shape[1]):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return SegResult(
            mask=mask,
            method="sam2",
            score=float(scores[best]),
            prompt_points=points,
        )


class _nullcontext:
    """Minimal no-op context manager (CPU path, no autocast)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
