"""Axis-fixed RANSAC ellipsoid-center fit (US-004, AC-6 / AC-7).

Given a world-frame point cloud sampled from the *visible* (camera-facing)
surface of a rotation-symmetric ellipsoid with KNOWN semi-axes ``(a, c)`` and a
KNOWN (or assumed) orientation, recover ONLY the ellipsoid center ``C``.

The whole point of this estimator is that the visible surface is a
self-occluded cap (roughly a hemisphere). The mean of those visible points (the
"surface centroid") sits ~one radius *toward the camera* from the true center
and is therefore the WRONG answer. By minimizing the rotation-symmetric
ellipsoid implicit residual

    f(p; C, a, c) = (dx^2 + dy^2) / a^2 + dz^2 / c^2 - 1

(where ``[dx, dy, dz] = R_ow @ (p - C)`` are object-frame offsets and ``a == b``
is the equatorial semi-axis, ``c`` the polar semi-axis), the fit back-solves the
hidden hemisphere and lands on the true center, not the visible centroid.

Single code path. ``shape_class`` only gates whether a cap-PCA major axis is
computed:
  * ``"spheroid"`` (a == c, sphere) / round  -> ``major_axis = None``.
  * ``"elongated"`` / prolate                 -> PCA on the cap inliers gives a
    unit major-axis direction (world frame).

Pure numpy + scipy. No sim-side (recorder / Isaac) imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..io.types import FitResult

__all__ = [
    "EllipsoidFitResult",
    "fit_ellipsoid_center",
    "ellipsoid_residual",
]


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class EllipsoidFitResult:
    """Output of the axis-fixed ellipsoid-center fit.

    Attributes:
        center: (3,) fitted ellipsoid center in WORLD frame. This is the AC-10
            graded quantity (the ``ellipsoid_fit`` center), distinct from the
            diagnostic surface centroid.
        cov: (3, 3) covariance of the center estimate (Gauss-Newton normal-
            equation inverse scaled by residual variance); ``None`` on failure.
        inliers: (N,) bool mask over the input points (RANSAC + refit inliers).
        major_axis: (3,) unit vector (world frame) of the cap's principal axis
            for elongated shapes, else ``None``.
        a: equatorial semi-axis used (echoed from input).
        c: polar semi-axis used (echoed from input).
        R_wo: (3, 3) world<-object rotation used (identity if none supplied).
        inlier_ratio: fraction of points kept as inliers.
        n_points: number of input points.
        n_iterations: RANSAC iterations actually run.
        ok: True if a center was produced.
    """

    center: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    inliers: Optional[np.ndarray] = None
    major_axis: Optional[np.ndarray] = None
    a: Optional[float] = None
    c: Optional[float] = None
    R_wo: Optional[np.ndarray] = None
    inlier_ratio: Optional[float] = None
    n_points: Optional[int] = None
    n_iterations: int = 0
    ok: bool = False

    def to_fit_result(self) -> FitResult:
        """Bridge to the shared, import-safe ``io.types.FitResult`` contract."""
        return FitResult(
            center=None if self.center is None else np.asarray(self.center),
            R_wo=None if self.R_wo is None else np.asarray(self.R_wo),
            a=self.a,
            c=self.c,
            inlier_ratio=self.inlier_ratio,
            n_points=self.n_points,
            ok=self.ok,
        )


# ----------------------------------------------------------------------------
# Core geometry
# ----------------------------------------------------------------------------
def ellipsoid_residual(
    points: np.ndarray,
    center: np.ndarray,
    a: float,
    c: float,
    R_ow: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rotation-symmetric ellipsoid implicit residual ``f = q - 1``.

    ``q = (dx^2 + dy^2) / a^2 + dz^2 / c^2`` where ``[dx, dy, dz]`` are the
    object-frame offsets of each point from ``center``. ``f == 0`` on the
    surface, ``< 0`` inside, ``> 0`` outside.

    Args:
        points: (N, 3) world-frame points.
        center: (3,) candidate center (world frame).
        a: equatorial semi-axis (a == b).
        c: polar semi-axis.
        R_ow: (3, 3) object<-world rotation (transforms world offsets into the
            object frame). ``None`` -> identity (object axes == world axes,
            polar axis == world z).

    Returns:
        (N,) residual array.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    d = points - center                              # (N, 3) world offsets
    if R_ow is not None:
        d = d @ np.asarray(R_ow, dtype=np.float64).T  # -> object frame
    inv = np.array([1.0 / (a * a), 1.0 / (a * a), 1.0 / (c * c)])
    q = (d * d) @ inv
    return q - 1.0


def _solve_center_lls(
    points: np.ndarray, a: float, c: float, R_ow: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Closed-form least-squares center for fixed (a, c) and orientation.

    With object-frame coords ``y = R_ow @ (p - C)`` and weights
    ``w = (1/a^2, 1/a^2, 1/c^2)``, the surface constraint ``sum_k w_k y_k^2 = 1``
    is, in object-frame point coords ``x = R_ow @ p`` and center
    ``b = R_ow @ C``:

        sum_k w_k (x_k - b_k)^2 = 1
        => -2 sum_k w_k x_k b_k + sum_k w_k b_k^2 = 1 - sum_k w_k x_k^2

    The quadratic term ``sum_k w_k b_k^2`` is a constant shared by every point,
    so subtract the per-point mean to cancel it and obtain a LINEAR system in
    ``b``. We solve for ``b`` then map back ``C = R_ow.T @ b``. This is the
    standard algebraic (Bookstein-style) center solve specialized to a fixed,
    axis-aligned, rotation-symmetric ellipsoid.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 3:
        return None

    if R_ow is not None:
        x = pts @ np.asarray(R_ow, dtype=np.float64).T   # object frame coords
    else:
        x = pts
    w = np.array([1.0 / (a * a), 1.0 / (a * a), 1.0 / (c * c)])

    # Per-point: A_i . b = g_i, with the shared b-quadratic removed by centering.
    A = 2.0 * (x * w)                       # (N, 3)  coefficient on b
    g = (x * x) @ w                         # (N,)    = sum_k w_k x_k^2

    A_c = A - A.mean(axis=0, keepdims=True)
    g_c = g - g.mean()

    try:
        b, *_ = np.linalg.lstsq(A_c, g_c, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(b)):
        return None

    if R_ow is not None:
        center = np.asarray(R_ow, dtype=np.float64).T @ b
    else:
        center = b
    return center


def _center_cov(
    points: np.ndarray,
    center: np.ndarray,
    a: float,
    c: float,
    R_ow: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Approximate (3, 3) center covariance from the residual Jacobian.

    ``J_i = d f_i / d C`` for the implicit residual; ``cov ~= s2 (J^T J)^-1``
    with ``s2`` the residual variance. Diagnostic-grade only.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 4:
        return None
    d = pts - np.asarray(center, dtype=np.float64).reshape(3)
    if R_ow is not None:
        Rm = np.asarray(R_ow, dtype=np.float64)
        d_obj = d @ Rm.T
    else:
        Rm = np.eye(3)
        d_obj = d
    w = np.array([1.0 / (a * a), 1.0 / (a * a), 1.0 / (c * c)])
    # f = sum_k w_k (R(p-C))_k^2 - 1 ; df/dC = -2 R^T diag(w) R (p - C)
    # J_i (1x3) = -2 (R^T (w * d_obj_i))
    J = -2.0 * (d_obj * w) @ Rm              # (N, 3)
    f = (d_obj * d_obj) @ w - 1.0            # (N,)
    JTJ = J.T @ J
    try:
        JTJ_inv = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        return None
    dof = max(n - 3, 1)
    s2 = float(f @ f) / dof
    return s2 * JTJ_inv


def _cap_major_axis(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Unit principal-axis direction of the cap inliers (world frame).

    PCA on the offsets from the fitted center; the largest-eigenvalue direction
    is the elongation (polar) axis. Sign is disambiguated to point away from the
    centroid offset for determinism.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    d = pts - np.asarray(center, dtype=np.float64).reshape(3)
    cov = np.cov(d.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # Deterministic sign: align with the mean offset (cap points lie on one side
    # of the center, so this picks a stable hemisphere-facing orientation).
    if np.dot(axis, d.mean(axis=0)) < 0:
        axis = -axis
    return axis


# ----------------------------------------------------------------------------
# Public RANSAC fit
# ----------------------------------------------------------------------------
def fit_ellipsoid_center(
    points: np.ndarray,
    a: float,
    c: float,
    shape_class: str = "spheroid",
    R_wo: Optional[np.ndarray] = None,
    *,
    n_iterations: int = 200,
    min_sample: int = 6,
    inlier_thresh: Optional[float] = None,
    inlier_ratio_target: float = 0.6,
    seed: Optional[int] = 0,
) -> EllipsoidFitResult:
    """Axis-fixed RANSAC fit of an ellipsoid CENTER only.

    Args:
        points: (N, 3) world-frame surface points (e.g. a self-occluded cap).
        a: equatorial semi-axis (a == b), meters.
        c: polar semi-axis, meters.
        shape_class: ``"spheroid"`` / ``"sphere"`` -> no major axis; anything
            else (``"elongated"`` / ``"prolate"`` / ``"ellipsoid"``) -> compute
            a cap-PCA major axis.
        R_wo: (3, 3) world<-object rotation. If given, the polar axis is
            ``R_wo[:, 2]``; if ``None`` the object frame is assumed aligned with
            the world (polar axis == world z). For M1 the default is used.
        n_iterations: RANSAC iteration budget.
        min_sample: minimal-set size per RANSAC hypothesis.
        inlier_thresh: |residual| inlier threshold. Defaults to a value scaled
            by the ellipsoid size (``0.25``, dimensionless in residual units).
        inlier_ratio_target: target inlier fraction (early-accept); also recorded.
        seed: RNG seed for reproducibility (``None`` -> nondeterministic).

    Returns:
        EllipsoidFitResult.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    a = float(a)
    c = float(c)

    # world<-object rotation; R_ow = R_wo.T transforms world offsets to object.
    if R_wo is not None:
        R_wo_m = np.asarray(R_wo, dtype=np.float64).reshape(3, 3)
        R_ow = R_wo_m.T
    else:
        R_wo_m = np.eye(3)
        R_ow = None  # identity fast-path

    result = EllipsoidFitResult(
        a=a, c=c, R_wo=R_wo_m, n_points=n, n_iterations=0, ok=False
    )

    if n < 3:
        return result

    # Dimensionless residual threshold (q is ~ (r/axis)^2 near the surface).
    if inlier_thresh is None:
        inlier_thresh = 0.25

    rng = np.random.default_rng(seed)
    sample_size = int(min(max(min_sample, 3), n))

    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    iters_run = 0

    # Degenerate / tiny clouds: skip RANSAC, fit on everything.
    if n <= sample_size:
        center0 = _solve_center_lls(pts, a, c, R_ow)
        if center0 is None:
            return result
        res = np.abs(ellipsoid_residual(pts, center0, a, c, R_ow))
        best_inliers = res <= inlier_thresh
        best_count = int(best_inliers.sum())
        iters_run = 1
    else:
        for it in range(int(n_iterations)):
            iters_run = it + 1
            idx = rng.choice(n, size=sample_size, replace=False)
            center_h = _solve_center_lls(pts[idx], a, c, R_ow)
            if center_h is None:
                continue
            res = np.abs(ellipsoid_residual(pts, center_h, a, c, R_ow))
            inliers = res <= inlier_thresh
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers
                if count >= inlier_ratio_target * n:
                    break

    if best_inliers is None or best_count < 3:
        # Fall back to a single all-points algebraic solve.
        center_all = _solve_center_lls(pts, a, c, R_ow)
        if center_all is None:
            result.n_iterations = iters_run
            return result
        res = np.abs(ellipsoid_residual(pts, center_all, a, c, R_ow))
        best_inliers = res <= inlier_thresh
        if int(best_inliers.sum()) < 3:
            best_inliers = np.ones(n, dtype=bool)

    # Refit on all inliers (re-weighted, single algebraic solve).
    inlier_pts = pts[best_inliers]
    center = _solve_center_lls(inlier_pts, a, c, R_ow)
    if center is None:
        center = _solve_center_lls(pts, a, c, R_ow)
        inlier_pts = pts
        best_inliers = np.ones(n, dtype=bool)
    if center is None:
        result.n_iterations = iters_run
        return result

    # Recompute inliers around the refined center for a clean final mask.
    final_res = np.abs(ellipsoid_residual(pts, center, a, c, R_ow))
    final_inliers = final_res <= inlier_thresh
    if int(final_inliers.sum()) >= 3:
        center2 = _solve_center_lls(pts[final_inliers], a, c, R_ow)
        if center2 is not None:
            center = center2
            best_inliers = final_inliers
    else:
        best_inliers = np.asarray(best_inliers, dtype=bool)

    inlier_count = int(best_inliers.sum())
    cov = _center_cov(pts[best_inliers], center, a, c, R_ow)

    major_axis = None
    if str(shape_class).lower() not in ("spheroid", "sphere", "round"):
        if R_wo is not None:
            # Known orientation: the polar (elongation) axis IS R_wo[:, 2].
            major_axis = R_wo_m[:, 2] / (np.linalg.norm(R_wo_m[:, 2]) + 1e-12)
        else:
            major_axis = _cap_major_axis(pts[best_inliers], center)

    result.center = np.asarray(center, dtype=np.float64)
    result.cov = cov
    result.inliers = np.asarray(best_inliers, dtype=bool)
    result.major_axis = major_axis
    result.inlier_ratio = inlier_count / float(n) if n else 0.0
    result.n_iterations = iters_run
    result.ok = True
    return result
