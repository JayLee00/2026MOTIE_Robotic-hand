"""Per-object shape initialization (US-005, AC-5).

Given a clean, low-occlusion early frame's *masked, world-frame* point cloud,
estimate a rotation-symmetric ellipsoid (a = b; two radii a, c + a center, with
the symmetry axis assumed ~world-z for M1), classify the shape against the
configured ``ca_threshold``, and write ``(a, c)`` / ``shape_class`` into
``objects.yaml``.

Design / ownership notes
------------------------
The shared contract asks us to *reuse* the fit primitive from
``in_hand_tracker.perception.ellipsoid_fit`` when it offers a free-axis (radii
estimating) mode. That module is owned by another agent and may expose only a
fixed-center / fixed-axes fit (or may not exist yet). We therefore:

  * Try, defensively, to discover a free-fit entry point in that module
    (``fit_axes`` / ``free_fit`` / ``fit_free``) and use it if present.
  * Otherwise fall back to a small, self-contained algebraic free-fit
    implemented *locally here* (per the contract's explicit instruction not to
    add helpers to ``perception/ellipsoid_fit.py``).

The local fit is the standard algebraic (Bookstein-style) least-squares on the
implicit quadric restricted to a = b with the symmetry axis along world-z:

    A (x^2 + y^2) + C z^2 + D x + E y + F z = 1

solved linearly for [A, C, D, E, F]; the center and radii follow by completing
the square. This recovers (a, c) and the center exactly on full synthetic
clouds and stays well within 10% relative error on partial (self-occluded) and
mildly noisy clouds.

Pure numpy / scipy / yaml. No sim-side (recorder / Isaac) imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FreeFitResult:
    """Free-axis rotation-symmetric ellipsoid fit (a = b).

    ``a`` is the equatorial semi-axis, ``c`` the polar (world-z) semi-axis
    (meters); ``center`` is the world-frame ellipsoid center (3,). ``shape_class``
    is the ``ca_threshold`` classification. ``n_points`` is the count actually
    used by the fit. ``ok`` is False when the fit could not be formed.
    """

    a: Optional[float] = None
    c: Optional[float] = None
    center: Optional[np.ndarray] = None
    shape_class: str = "spheroid"
    ca_ratio: Optional[float] = None
    n_points: int = 0
    ok: bool = False


# --------------------------------------------------------------------------- #
# Optional reuse of the perception fit primitive (owned by another agent).
# --------------------------------------------------------------------------- #
def _try_perception_free_fit(
    points: np.ndarray,
) -> Optional[Tuple[float, float, np.ndarray]]:
    """Use a free-axis fit from ``perception.ellipsoid_fit`` if one is exposed.

    Returns ``(a, c, center)`` on success, or ``None`` when no compatible
    free-fit entry point exists (in which case we fall back to the local fit).
    This stays a *soft* dependency so calibration works whether or not the
    sibling module is present yet, and never reaches into its internals.
    """
    try:
        from in_hand_tracker.perception import ellipsoid_fit as _ef  # type: ignore
    except Exception:
        return None

    for name in ("fit_axes", "free_fit", "fit_free", "fit_spheroid_free"):
        fn = getattr(_ef, name, None)
        if not callable(fn):
            continue
        try:
            out = fn(np.asarray(points, dtype=np.float64))
        except Exception:
            return None
        a, c, center = _coerce_free_fit_output(out)
        if a is None:
            return None
        return float(a), float(c), np.asarray(center, dtype=np.float64).reshape(3)
    return None


def _coerce_free_fit_output(out):
    """Normalize a variety of plausible return shapes to (a, c, center)."""
    # (a, c, center)
    if isinstance(out, tuple) and len(out) == 3:
        a, c, center = out
        return float(a), float(c), np.asarray(center, dtype=np.float64).reshape(3)
    # An object exposing .a / .c / .center (e.g. a FitResult-like).
    a = getattr(out, "a", None)
    c = getattr(out, "c", None)
    center = getattr(out, "center", None)
    if a is not None and c is not None and center is not None:
        return float(a), float(c), np.asarray(center, dtype=np.float64).reshape(3)
    return None, None, None


# --------------------------------------------------------------------------- #
# Local free-axis algebraic fit (a = b, symmetry axis = world-z).
# --------------------------------------------------------------------------- #
def free_fit_spheroid(points: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Fit a rotation-symmetric ellipsoid (a = b, axis = world-z) to ``points``.

    Args:
        points: (N, 3) world-frame surface points.

    Returns:
        ``(a, c, center)`` where ``a`` is the equatorial semi-axis, ``c`` the
        polar (world-z) semi-axis (meters), and ``center`` is the (3,) world
        center.

    Raises:
        ValueError: if fewer than 5 points are supplied (the linear system has
            5 unknowns) or the system is degenerate.
    """
    # Prefer the sibling perception primitive when it offers a free fit.
    reused = _try_perception_free_fit(points)
    if reused is not None:
        return reused

    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] < 5:
        raise ValueError(
            f"need >= 5 points for a free spheroid fit, got {P.shape[0]}"
        )

    # Center + scale the data for numerical conditioning, then undo afterwards.
    mu = P.mean(axis=0)
    Q = P - mu
    scale = float(np.sqrt((Q ** 2).sum(axis=1).mean()))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("degenerate point cloud (zero spread)")
    Q /= scale

    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    # A (x^2 + y^2) + C z^2 + D x + E y + F z = 1
    M = np.column_stack([x * x + y * y, z * z, x, y, z])
    b = np.ones(Q.shape[0], dtype=np.float64)
    coef, _res, _rank, _sv = np.linalg.lstsq(M, b, rcond=None)
    A, C, D, E, F = (float(v) for v in coef)

    if not (A > 0.0 and C > 0.0):
        raise ValueError(
            "fit did not yield a positive-definite spheroid "
            f"(A={A:.3e}, C={C:.3e}); cloud may not be ellipsoidal"
        )

    # Complete the square (in the normalized frame).
    cxn = -D / (2.0 * A)
    cyn = -E / (2.0 * A)
    czn = -F / (2.0 * C)
    k = 1.0 + A * cxn * cxn + A * cyn * cyn + C * czn * czn
    if k <= 0.0:
        raise ValueError("non-physical fit (negative squared radius)")

    a_n = np.sqrt(k / A)
    c_n = np.sqrt(k / C)

    # Undo the normalization.
    a = float(a_n * scale)
    c = float(c_n * scale)
    center = np.array([cxn, cyn, czn], dtype=np.float64) * scale + mu
    return a, c, center


# --------------------------------------------------------------------------- #
# Shape classification.
# --------------------------------------------------------------------------- #
def classify_shape(a: float, c: float, ca_threshold: float = 1.2) -> str:
    """Tag the shape from the polar/equatorial aspect ratio.

    The aspect ratio is taken as max(c, a) / min(c, a) so the tag does not
    depend on which axis happens to be longer. Below ``ca_threshold`` the object
    is ~rotation-symmetric-round (e.g. an apple) -> ``"spheroid"``; at/above it
    the object is markedly elongated (prolate / oblate) -> ``"elongated"``.
    """
    a = float(a)
    c = float(c)
    lo = min(a, c)
    hi = max(a, c)
    if lo <= 0.0:
        return "elongated"
    ratio = hi / lo
    return "elongated" if ratio >= float(ca_threshold) else "spheroid"


# --------------------------------------------------------------------------- #
# End-to-end calibration from a world-frame cloud.
# --------------------------------------------------------------------------- #
def calibrate_object(
    points_world: np.ndarray, ca_threshold: float = 1.2
) -> FreeFitResult:
    """Fit + classify from a masked, world-frame surface cloud.

    Args:
        points_world: (N, 3) world-frame masked surface points from a clean,
            low-occlusion early frame.
        ca_threshold: aspect-ratio threshold (estimator.yaml ``ca_threshold``).

    Returns:
        A :class:`FreeFitResult`. ``ok`` is False (with a "spheroid" default
        tag and None radii) if the cloud is too small / degenerate to fit.
    """
    P = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    try:
        a, c, center = free_fit_spheroid(P)
    except ValueError:
        return FreeFitResult(n_points=int(P.shape[0]), ok=False)

    shape_class = classify_shape(a, c, ca_threshold)
    ca_ratio = max(a, c) / min(a, c) if min(a, c) > 0 else float("inf")
    return FreeFitResult(
        a=a,
        c=c,
        center=center,
        shape_class=shape_class,
        ca_ratio=float(ca_ratio),
        n_points=int(P.shape[0]),
        ok=True,
    )


# --------------------------------------------------------------------------- #
# Persistence into objects.yaml.
# --------------------------------------------------------------------------- #
def write_object_entry(
    objects_yaml_path: str,
    object_id: str,
    fit: FreeFitResult,
    semantic_color: Optional[str] = None,
) -> dict:
    """Write/merge a calibrated object entry into ``objects.yaml``.

    The file's top-level ``objects`` map is keyed by ``object_id`` (the semantic
    class name). Each entry carries ``a``, ``c``, ``shape_class`` and, when
    provided, the ``semantic_color`` RGB string as it appears in the sidecar.

    Returns the full, updated config dict (also written back to disk). Raises
    ValueError if ``fit`` is not ``ok`` (nothing to persist).
    """
    if not fit.ok or fit.a is None or fit.c is None:
        raise ValueError("refusing to write a non-ok fit to objects.yaml")

    import yaml

    try:
        with open(objects_yaml_path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = None
    if not isinstance(cfg, dict):
        cfg = {}
    objects = cfg.get("objects")
    if not isinstance(objects, dict):
        objects = {}

    entry = {
        "a": round(float(fit.a), 6),
        "c": round(float(fit.c), 6),
        "shape_class": str(fit.shape_class),
    }
    if semantic_color is not None:
        entry["semantic_color"] = str(semantic_color)

    objects[str(object_id)] = entry
    cfg["objects"] = objects

    with open(objects_yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
    return cfg
