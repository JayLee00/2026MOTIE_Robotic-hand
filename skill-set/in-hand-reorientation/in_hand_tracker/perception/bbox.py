"""3D bounding-box output policy (US-007, AC-8 / AC-11).

Turns an ellipsoid-center fit ``(center C, a, c, shape_class, major_axis)`` into a
3D bounding box. Two policies, selected by shape class:

  * **spheroid** (a ~= c, round): an axis-aligned box centered on ``C`` with
    half-extent ``max(a, c)`` on every axis -- :func:`aabb_from_sphere`. The
    bounding sphere always contains the object, so the AABB contains it too.
  * **elongated** (prolate): when the major-axis direction is known, a *tight*
    oriented box (OBB) whose local +z is the major axis, with half-extents
    ``(a, a, c)`` -- :func:`tight_obb`. When the direction is unavailable a
    conservative AABB with isotropic half-extent ``max(a, c)`` is emitted so the
    GT box is still contained on the worst (end-on) frame --
    :func:`conservative_aabb`.

A box is represented by :class:`Box3D`: a world-frame ``center`` (3,), a per-axis
``half_extent`` (3,) and a ``R`` (3, 3) world<-box rotation (identity for an
AABB). Helpers compute the 8 corners, the enclosing AABB, and 3D IoU (axis-aligned
and via voxel sampling for the oriented case).

Pure numpy. No sim-side imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

__all__ = [
    "Box3D",
    "aabb_from_sphere",
    "tight_obb",
    "conservative_aabb",
    "box_from_fit",
    "aabb_iou",
    "box_iou_voxel",
]


# --------------------------------------------------------------------------- #
# Box container
# --------------------------------------------------------------------------- #
@dataclass
class Box3D:
    """A 3D box: center + per-axis half-extent + world<-box rotation.

    For an axis-aligned box ``R`` is the identity. ``kind`` records the policy
    that produced the box ("aabb_sphere" | "obb_tight" | "aabb_conservative").
    """

    center: np.ndarray              # (3,) world
    half_extent: np.ndarray         # (3,) box-local half sizes (>0)
    R: np.ndarray = field(          # (3, 3) world<-box (columns = box axes)
        default_factory=lambda: np.eye(3)
    )
    kind: str = "aabb_sphere"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.half_extent = np.abs(
            np.asarray(self.half_extent, dtype=np.float64).reshape(3)
        )
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)

    # ----- geometry ----- #
    def corners(self) -> np.ndarray:
        """(8, 3) world-frame corners."""
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=np.float64,
        )
        local = signs * self.half_extent[None, :]      # (8, 3)
        return (self.R @ local.T).T + self.center

    def aabb(self) -> "Box3D":
        """The smallest axis-aligned box enclosing this (possibly oriented) box."""
        c = self.corners()
        lo = c.min(axis=0)
        hi = c.max(axis=0)
        return Box3D(
            center=0.5 * (lo + hi),
            half_extent=0.5 * (hi - lo),
            R=np.eye(3),
            kind="aabb",
        )

    def bounds(self) -> tuple:
        """Axis-aligned (lo (3,), hi (3,)) of this box's enclosing AABB."""
        c = self.corners()
        return c.min(axis=0), c.max(axis=0)

    def volume(self) -> float:
        return float(8.0 * np.prod(self.half_extent))

    def contains(self, points: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """(N,) bool: which world points lie inside the box (within ``tol``)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        local = (pts - self.center) @ self.R          # world->box
        he = self.half_extent + float(tol)
        return np.all(np.abs(local) <= he[None, :], axis=1)

    def to_dict(self) -> Dict:
        lo, hi = self.bounds()
        return {
            "center": self.center.tolist(),
            "half_extent": self.half_extent.tolist(),
            "R": self.R.tolist(),
            "kind": self.kind,
            "aabb_min": lo.tolist(),
            "aabb_max": hi.tolist(),
        }


# --------------------------------------------------------------------------- #
# Output policies
# --------------------------------------------------------------------------- #
def aabb_from_sphere(center: np.ndarray, radius: float) -> Box3D:
    """Axis-aligned box from a bounding sphere (spheroid policy).

    Half-extent is ``radius`` on every axis; the box therefore contains the
    bounding sphere and thus the object.
    """
    r = abs(float(radius))
    return Box3D(
        center=np.asarray(center, dtype=np.float64).reshape(3),
        half_extent=np.full(3, r),
        R=np.eye(3),
        kind="aabb_sphere",
    )


def _basis_from_major(major_axis: np.ndarray) -> np.ndarray:
    """Right-handed (3, 3) basis whose 3rd column is the unit ``major_axis``."""
    z = np.asarray(major_axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(z)
    if not np.isfinite(n) or n < 1e-9:
        return np.eye(3)
    z = z / n
    # Pick a helper not parallel to z to seed the orthogonal complement.
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = helper - np.dot(helper, z) * z
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def tight_obb(
    center: np.ndarray, a: float, c: float, major_axis: np.ndarray
) -> Box3D:
    """Tight oriented box for a prolate spheroid with a known major-axis dir.

    Local +z is ``major_axis`` (the polar/elongation axis, half-length ``c``);
    the two equatorial axes get half-length ``a``. Encloses the spheroid tightly.
    """
    R = _basis_from_major(major_axis)
    a = abs(float(a))
    c = abs(float(c))
    return Box3D(
        center=np.asarray(center, dtype=np.float64).reshape(3),
        half_extent=np.array([a, a, c], dtype=np.float64),
        R=R,
        kind="obb_tight",
    )


def conservative_aabb(center: np.ndarray, *radii: float) -> Box3D:
    """Conservative AABB: isotropic half-extent = max(|radii|) on every axis.

    Used when the elongation direction is unknown: an axis-aligned box big
    enough to contain the spheroid in ANY orientation (including the worst
    end-on frame).
    """
    r = max(abs(float(x)) for x in radii) if radii else 0.0
    return Box3D(
        center=np.asarray(center, dtype=np.float64).reshape(3),
        half_extent=np.full(3, r),
        R=np.eye(3),
        kind="aabb_conservative",
    )


def box_from_fit(
    center: np.ndarray,
    a: float,
    c: float,
    shape_class: str,
    major_axis: Optional[np.ndarray] = None,
) -> Box3D:
    """Apply the output policy for a fitted ellipsoid center.

    * spheroid -> :func:`aabb_from_sphere` with radius ``max(a, c)``.
    * elongated + major_axis known -> :func:`tight_obb`.
    * elongated + no direction -> :func:`conservative_aabb`.
    """
    a = abs(float(a))
    c = abs(float(c))
    sc = str(shape_class).lower()
    if sc in ("spheroid", "sphere", "round"):
        return aabb_from_sphere(center, max(a, c))
    # elongated / prolate / ellipsoid
    if major_axis is not None and np.all(np.isfinite(np.asarray(major_axis))):
        return tight_obb(center, a, c, major_axis)
    return conservative_aabb(center, a, c)


# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #
def aabb_iou(box_a: Box3D, box_b: Box3D) -> float:
    """Axis-aligned 3D IoU using each box's enclosing AABB.

    Exact when both boxes are axis-aligned; an upper bound otherwise (use
    :func:`box_iou_voxel` for the oriented case).
    """
    lo_a, hi_a = box_a.bounds()
    lo_b, hi_b = box_b.bounds()
    lo = np.maximum(lo_a, lo_b)
    hi = np.minimum(hi_a, hi_b)
    inter_dims = np.clip(hi - lo, 0.0, None)
    inter = float(np.prod(inter_dims))
    vol_a = float(np.prod(hi_a - lo_a))
    vol_b = float(np.prod(hi_b - lo_b))
    union = vol_a + vol_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def box_iou_voxel(box_a: Box3D, box_b: Box3D, n: int = 40) -> float:
    """3D IoU of two (possibly oriented) boxes by uniform voxel sampling.

    Samples an ``n^3`` grid over the union AABB and counts membership in each
    box. Robust to orientation; accuracy improves with ``n``.
    """
    lo_a, hi_a = box_a.bounds()
    lo_b, hi_b = box_b.bounds()
    lo = np.minimum(lo_a, lo_b)
    hi = np.maximum(hi_a, hi_b)
    span = hi - lo
    if np.any(span <= 0):
        return 0.0
    axes = [np.linspace(lo[d] + span[d] / (2 * n), hi[d] - span[d] / (2 * n), n)
            for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    in_a = box_a.contains(pts)
    in_b = box_b.contains(pts)
    inter = int(np.logical_and(in_a, in_b).sum())
    union = int(np.logical_or(in_a, in_b).sum())
    if union == 0:
        return 0.0
    return inter / union
