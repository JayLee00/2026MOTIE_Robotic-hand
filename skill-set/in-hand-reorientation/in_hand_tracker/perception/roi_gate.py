"""Region-of-interest (ROI) gating before the fit (US-003, AC-4 part 1).

The ROI gate shrinks the candidate point set / image region that is fed to the
ellipsoid fit, so stray background / table / arm points do not pollute RANSAC.
It is a pure-geometry, axis-aligned 3D box crop plus the matching image mask.

Two modes (``cfg.roi.mode``):

  * ``"fixed"`` (M1 default): an axis-aligned workspace box centered on a *hint*
    point (the object GT region, or a camera-aim point), with half-extent taken
    from cfg. This is the dependable default and never requires extra deps.

  * ``"fk"``: derive the workspace box from the hand. Given ``hand_pos`` +
    ``hand_joints`` and a KISTAR URDF (path supplied at runtime via cfg/hints),
    forward kinematics (pinocchio) gives the fingertip/palm world positions; the
    box is their AABB grown by a margin. If the URDF path is unavailable, the
    URDF cannot be parsed, or pinocchio is missing, this logs a warning and
    falls back to ``"fixed"``.

Clean-room: this module never imports the sim-side stack (recorder / Isaac /
omni). The KISTAR URDF is a *runtime input* (a path string the caller passes),
never a hardcoded source literal, so the AC-2 boundary grep stays at 0 hits.

The crop operates on a point set that is either world-frame or camera-frame; the
box center hint must be expressed in the *same* frame as the points. The image
``mask`` (and ``depth``) are carried through and a cropped mask is returned with
the per-pixel kept flags aligned to the surviving points, so a downstream
consumer can re-rasterize if needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default 3D workspace half-extent (meters) for the fixed box when cfg omits it.
# An in-hand fruit/tool plus a little slack comfortably fits in ~15 cm.
_DEFAULT_HALF_EXTENT_M = 0.15

# Default extra margin (meters) grown around the FK fingertip AABB.
_DEFAULT_FK_MARGIN_M = 0.05


@dataclass
class RoiResult:
    """Output of :func:`roi_crop`.

    Attributes:
        points: (M, 3) the kept point subset (same frame as the input points).
        kept_index: (M,) int indices into the *input* point array that survived.
        center: (3,) the box center actually used (in the points' frame).
        half_extent: (3,) per-axis box half-size (meters).
        cropped_mask: (H, W) bool image mask after cropping, or None when no
            per-point pixel correspondence was supplied. When the input ``mask``
            is given and its True-count equals the number of input points (the
            usual deproject contract: one point per masked pixel, in row-major
            ``np.nonzero`` order), the surviving points re-light their pixels.
        mode: the mode actually applied ("fixed" or "fk").
        n_in: number of input points.
        n_out: number of kept points (== len(points)).
    """

    points: np.ndarray
    kept_index: np.ndarray
    center: np.ndarray
    half_extent: np.ndarray
    cropped_mask: Optional[np.ndarray] = None
    mode: str = "fixed"
    n_in: int = 0
    n_out: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# cfg / hint helpers
# --------------------------------------------------------------------------- #
def _get(d: Any, key: str, default: Any = None) -> Any:
    """dict-or-attr getter so both dicts and simple namespaces work."""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _roi_cfg(cfg: Any) -> Any:
    """Extract the ``roi`` sub-config, tolerating cfg being the roi block itself."""
    if cfg is None:
        return {}
    roi = _get(cfg, "roi", None)
    return roi if roi is not None else cfg


def _as_half_extent(value: Any, default: float = _DEFAULT_HALF_EXTENT_M) -> np.ndarray:
    """Coerce a scalar or 3-vector half-extent into a (3,) float array."""
    if value is None:
        return np.full(3, float(default), dtype=np.float64)
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    if arr.size == 1:
        return np.full(3, float(arr[0]), dtype=np.float64)
    if arr.size >= 3:
        return arr[:3].astype(np.float64)
    raise ValueError(f"half_extent must be scalar or length-3, got {value!r}")


def _resolve_center(fixed_cfg: Any, hints: Any, points: np.ndarray) -> np.ndarray:
    """Pick the fixed-box center in the points' frame.

    Priority: explicit cfg ``center`` -> hints ``center`` -> hints
    ``gt_object_pos`` -> hints ``cam_aim`` -> hints ``cam_pos`` -> the median of
    the input points (a robust fallback that always keeps the box over the data).
    """
    for src, key in (
        (fixed_cfg, "center"),
        (hints, "center"),
        (hints, "gt_object_pos"),
        (hints, "cam_aim"),
        (hints, "cam_pos"),
    ):
        v = _get(src, key, None)
        if v is not None:
            return np.asarray(v, dtype=np.float64).reshape(3)
    if points.shape[0] > 0:
        return np.median(points, axis=0)
    return np.zeros(3, dtype=np.float64)


# --------------------------------------------------------------------------- #
# box crop core
# --------------------------------------------------------------------------- #
def _box_keep_mask(
    points: np.ndarray, center: np.ndarray, half_extent: np.ndarray
) -> np.ndarray:
    """(N,) bool: True where ``points`` lie inside the axis-aligned box."""
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    lo = center - half_extent
    hi = center + half_extent
    return np.all((points >= lo) & (points <= hi), axis=1)


def _rebuild_cropped_mask(
    mask: Optional[np.ndarray], keep: np.ndarray
) -> Optional[np.ndarray]:
    """Re-light the surviving pixels of ``mask`` given the per-point keep flags.

    Relies on the deproject contract: the i-th point corresponds to the i-th
    True pixel of ``mask`` in row-major ``np.nonzero`` order. Returns None when
    no mask was supplied or its True-count does not match the point count (so we
    never silently misalign).
    """
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    vs, us = np.nonzero(mask)
    if vs.size != keep.size:
        return None
    out = np.zeros_like(mask, dtype=bool)
    kv, ku = vs[keep], us[keep]
    out[kv, ku] = True
    return out


# --------------------------------------------------------------------------- #
# FK box (optional; pinocchio)
# --------------------------------------------------------------------------- #
def _fk_box(
    cfg_fk: Any, hints: Any
) -> Optional[tuple]:
    """Compute an FK-derived (center, half_extent) AABB, or None to fall back.

    Returns None (and logs) whenever the FK path cannot be honored: missing
    pinocchio, missing/unparseable URDF, or missing hand state. The caller then
    uses the fixed box. The URDF path is taken at runtime from cfg/hints
    (``urdf_path``), never hardcoded, to keep the clean boundary.
    """
    urdf_path = _get(cfg_fk, "urdf_path", None) or _get(hints, "urdf_path", None)
    if not urdf_path:
        logger.warning("roi_gate: fk mode requested but no urdf_path; falling back to fixed")
        return None

    hand_pos = _get(hints, "hand_pos", None)
    hand_quat_wxyz = _get(hints, "hand_quat_wxyz", None)
    hand_joints = _get(hints, "hand_joints", None)
    hand_joint_names = _get(hints, "hand_joint_names", None)
    if hand_pos is None or hand_joints is None:
        logger.warning("roi_gate: fk mode missing hand_pos/hand_joints; falling back to fixed")
        return None

    try:
        import os

        import pinocchio as pin
    except Exception as exc:  # pragma: no cover - depends on host env
        logger.warning("roi_gate: pinocchio unavailable (%s); falling back to fixed", exc)
        return None

    if not os.path.isfile(str(urdf_path)):
        logger.warning("roi_gate: urdf not found at %s; falling back to fixed", urdf_path)
        return None

    try:
        model = pin.buildModelFromUrdf(str(urdf_path))
        data = model.createData()

        # Build the configuration vector q (length model.nq) by name-matching the
        # recorded joint names; unmatched joints stay at 0.
        q = pin.neutral(model) if hasattr(pin, "neutral") else np.zeros(model.nq)
        q = np.asarray(q, dtype=np.float64).copy()
        names = list(hand_joint_names) if hand_joint_names is not None else []
        joints = np.asarray(hand_joints, dtype=np.float64).ravel()
        for jname, jval in zip(names, joints):
            if model.existJointName(jname):
                jid = model.getJointId(jname)
                idx_q = model.idx_qs[jid]
                # revolute joints occupy a single q slot
                if 0 <= idx_q < q.shape[0]:
                    q[idx_q] = float(jval)

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # Collect link/frame translations in the model's root frame.
        pts_local = []
        for fr in data.oMf:
            pts_local.append(np.asarray(fr.translation, dtype=np.float64))
        if not pts_local:
            logger.warning("roi_gate: fk produced no frames; falling back to fixed")
            return None
        pts_local = np.asarray(pts_local, dtype=np.float64)

        # Place into world using the hand root pose (rotation optional).
        hand_pos = np.asarray(hand_pos, dtype=np.float64).reshape(3)
        if hand_quat_wxyz is not None:
            from scipy.spatial.transform import Rotation

            qw = np.asarray(hand_quat_wxyz, dtype=np.float64).reshape(4)
            R = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
            pts_world = (R @ pts_local.T).T + hand_pos
        else:
            pts_world = pts_local + hand_pos

        lo = pts_world.min(axis=0)
        hi = pts_world.max(axis=0)
        margin = float(_get(cfg_fk, "margin_m", _get(cfg_fk, "margin", _DEFAULT_FK_MARGIN_M)))
        center = 0.5 * (lo + hi)
        half_extent = 0.5 * (hi - lo) + margin
        return center, half_extent
    except Exception as exc:  # pragma: no cover - depends on host env / urdf
        logger.warning("roi_gate: fk computation failed (%s); falling back to fixed", exc)
        return None


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def roi_crop(
    points: np.ndarray,
    depth: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    cfg: Any = None,
    hints: Any = None,
) -> RoiResult:
    """Crop a point set to a workspace ROI box (US-003 / AC-4 part 1).

    Args:
        points: (N, 3) point set in world OR camera frame. The box center (from
            cfg/hints) must be in the *same* frame.
        depth: optional (H, W) depth/distance map. Carried for API symmetry;
            unused by the box crop itself (the crop is in 3D).
        mask: optional (H, W) bool image mask. When its True-count matches the
            number of input points (the deproject contract), a cropped mask with
            the surviving pixels is returned in ``RoiResult.cropped_mask``.
        cfg: config with a ``roi`` block: ``roi.mode`` ("fixed"|"fk"),
            ``roi.fixed`` (``center`` optional, ``half_extent`` scalar/3-vec in
            meters) and ``roi.fk`` (``urdf_path``, ``margin_m``). Plain dict or
            attribute-style objects both work; ``cfg`` may also be the roi block.
        hints: per-frame hints: ``gt_object_pos`` / ``cam_pos`` / ``cam_aim`` /
            ``center`` (box center candidates, in the points' frame), and for FK:
            ``hand_pos`` / ``hand_quat_wxyz`` / ``hand_joints`` /
            ``hand_joint_names`` / ``urdf_path``.

    Returns:
        :class:`RoiResult` with the filtered point set, the kept indices, the box
        geometry actually used, and (when derivable) the cropped image mask.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        if points.size == 0:
            points = points.reshape(0, 3)
        else:
            raise ValueError(f"points must be (N, 3), got {points.shape}")
    n_in = int(points.shape[0])

    roi = _roi_cfg(cfg)
    mode = str(_get(roi, "mode", "fixed")).lower()
    fixed_cfg = _get(roi, "fixed", {}) or {}
    fk_cfg = _get(roi, "fk", {}) or {}

    center = None
    half_extent = None
    applied_mode = mode

    if mode == "fk":
        fk = _fk_box(fk_cfg, hints)
        if fk is not None:
            center, half_extent = fk
            applied_mode = "fk"
        else:
            applied_mode = "fixed"  # graceful fallback

    if center is None or half_extent is None:
        applied_mode = "fixed" if mode != "fk" else applied_mode
        center = _resolve_center(fixed_cfg, hints, points)
        half_extent = _as_half_extent(_get(fixed_cfg, "half_extent", None))

    center = np.asarray(center, dtype=np.float64).reshape(3)
    half_extent = np.asarray(half_extent, dtype=np.float64).reshape(3)

    keep = _box_keep_mask(points, center, half_extent)
    kept_index = np.nonzero(keep)[0]
    kept_points = points[keep]
    cropped_mask = _rebuild_cropped_mask(mask, keep)

    return RoiResult(
        points=kept_points,
        kept_index=kept_index,
        center=center,
        half_extent=half_extent,
        cropped_mask=cropped_mask,
        mode=applied_mode,
        n_in=n_in,
        n_out=int(kept_points.shape[0]),
        meta={"requested_mode": mode},
    )
