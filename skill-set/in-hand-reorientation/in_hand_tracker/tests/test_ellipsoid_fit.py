"""US-004 / AC-6 / AC-7: axis-fixed RANSAC ellipsoid-center fit.

These tests construct a NEAR-HEMISPHERE point cloud: only the camera-facing cap
of a known ellipsoid is sampled (self-occlusion, exactly like a single depth
view). The fit must back-solve the HIDDEN hemisphere and recover the true
center -- NOT return the visible-surface centroid, which is biased ~one radius
toward the camera. We assert both:
  * fitted center within a tight tolerance of the TRUE center, and
  * the naive surface centroid is ~one radius OFF (the contrast that proves the
    fit is not just averaging visible points).
"""

import numpy as np
import pytest

from in_hand_tracker.perception.ellipsoid_fit import (
    EllipsoidFitResult,
    fit_ellipsoid_center,
    ellipsoid_residual,
)
from in_hand_tracker.io.deproject import surface_centroid


def _sample_cap(
    center, a, c, view_dir, n=1200, cap_frac=0.5, noise=0.0, seed=0
):
    """Sample the camera-FACING cap of an axis-aligned (polar==z) ellipsoid.

    A point on the ellipsoid is ``p = center + [a*sx*cosφ, a*sy*sinφ, c*sz]``
    parameterized by a unit sphere direction. We keep only points whose OUTWARD
    surface normal faces the camera (``normal . (-view_dir) > cos(cap)``), i.e.
    the cap the camera can actually see. ``view_dir`` points FROM the camera
    TOWARD the object, so ``-view_dir`` points back at the camera.
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float64)
    view_dir = np.asarray(view_dir, dtype=np.float64)
    view_dir = view_dir / np.linalg.norm(view_dir)
    to_cam = -view_dir

    pts = []
    # oversample unit directions, keep camera-facing ones, build surface points.
    guard = 0
    while len(pts) < n and guard < 200:
        guard += 1
        u = rng.normal(size=(4 * n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        # surface point on the ellipsoid for direction u (scaled axes).
        surf = u * np.array([a, a, c])
        # outward normal of f=(x/a)^2+(y/a)^2+(z/c)^2-1 is grad = (2x/a^2, ...).
        grad = surf * np.array([1.0 / a**2, 1.0 / a**2, 1.0 / c**2])
        nrm = grad / np.linalg.norm(grad, axis=1, keepdims=True)
        facing = nrm @ to_cam
        keep = facing > (1.0 - 2.0 * cap_frac)  # cap_frac=0.5 -> facing>0 (hemisphere)
        sel = surf[keep]
        for s in sel:
            pts.append(s)
            if len(pts) >= n:
                break
    arr = np.asarray(pts[:n], dtype=np.float64)
    if noise > 0:
        arr = arr + rng.normal(scale=noise, size=arr.shape)
    return arr + center


@pytest.mark.parametrize(
    "a,c,shape_class",
    [
        (0.04, 0.04, "spheroid"),     # sphere, radius 4 cm (apple-ish)
        (0.04, 0.06, "elongated"),    # prolate, c/a = 1.5
    ],
)
def test_fit_recovers_hidden_hemisphere_center(a, c, shape_class):
    true_center = np.array([0.31, -0.12, 0.84])
    view_dir = np.array([0.2, 0.1, 1.0])          # camera looks roughly +Z
    radius = max(a, c)
    noise = 0.002                                  # 2 mm depth noise

    cloud = _sample_cap(
        true_center, a, c, view_dir, n=1500, cap_frac=0.5, noise=noise, seed=7
    )
    assert cloud.shape[0] >= 500

    res = fit_ellipsoid_center(cloud, a, c, shape_class=shape_class, seed=0)
    assert isinstance(res, EllipsoidFitResult)
    assert res.ok
    assert res.center is not None

    fit_err = np.linalg.norm(res.center - true_center)
    centroid = surface_centroid(cloud)
    centroid_err = np.linalg.norm(centroid - true_center)

    # The fit back-solves the hidden hemisphere: tight on the TRUE center.
    assert fit_err < 0.15 * radius, (
        f"fit center off by {fit_err:.4f} m (> 0.15*r={0.15*radius:.4f}); "
        f"shape={shape_class}"
    )

    # Contrast: the naive visible-surface centroid is ~one radius off and is
    # MUCH worse than the fit -- proving the fit is not just averaging points.
    assert centroid_err > 0.4 * radius, (
        f"surface centroid only {centroid_err:.4f} m off; expected ~one radius"
    )
    assert centroid_err > 3.0 * fit_err

    # RANSAC inlier-ratio target.
    assert res.inlier_ratio is not None and res.inlier_ratio >= 0.6
    assert res.inliers is not None and res.inliers.shape[0] == cloud.shape[0]
    assert res.cov is not None and res.cov.shape == (3, 3)

    # major_axis gating: spheroid -> None; elongated -> unit 3-vector.
    if shape_class == "spheroid":
        assert res.major_axis is None
    else:
        assert res.major_axis is not None
        assert res.major_axis.shape == (3,)
        assert np.isclose(np.linalg.norm(res.major_axis), 1.0, atol=1e-6)


def test_residual_zero_on_surface():
    a, c = 0.05, 0.05
    center = np.array([0.1, 0.2, 0.3])
    rng = np.random.default_rng(1)
    u = rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    surf = center + u * np.array([a, a, c])
    r = ellipsoid_residual(surf, center, a, c)
    assert np.allclose(r, 0.0, atol=1e-9)


def test_known_orientation_uses_polar_axis_as_major():
    # Tilted prolate: object polar axis is world Z rotated 30deg about X.
    from scipy.spatial.transform import Rotation

    a, c = 0.03, 0.05
    R_wo = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    true_center = np.array([0.0, 0.05, 0.7])
    # build a cap in OBJECT frame then rotate into world.
    obj_cloud = _sample_cap(
        np.zeros(3), a, c, np.array([0.0, 0.0, 1.0]),
        n=1200, cap_frac=0.5, noise=0.0015, seed=3,
    )
    world_cloud = obj_cloud @ R_wo.T + true_center

    res = fit_ellipsoid_center(
        world_cloud, a, c, shape_class="elongated", R_wo=R_wo, seed=0
    )
    assert res.ok
    fit_err = np.linalg.norm(res.center - true_center)
    assert fit_err < 0.15 * max(a, c)
    # With a known orientation the major axis is the object polar axis R_wo[:,2].
    expected = R_wo[:, 2]
    cos = abs(float(np.dot(res.major_axis, expected)))
    assert cos > 0.999


def test_empty_and_tiny_inputs_are_safe():
    empty = fit_ellipsoid_center(np.empty((0, 3)), 0.04, 0.04)
    assert not empty.ok and empty.center is None
    tiny = fit_ellipsoid_center(np.zeros((2, 3)), 0.04, 0.04)
    assert not tiny.ok
