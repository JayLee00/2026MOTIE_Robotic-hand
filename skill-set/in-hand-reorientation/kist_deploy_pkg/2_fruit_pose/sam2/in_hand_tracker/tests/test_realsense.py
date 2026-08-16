"""Hardware-free unit tests for the REAL (RealSense) bbox path.

These exercise the geometry/math that does NOT need a camera:

  (1) Z-depth deprojection vs a synthetic K round-trips to the source pixels and
      recovers Z exactly (and is DISTINCT from the euclidean sim deproject for
      off-center pixels: |point| > Z there).
  (2) project_points is the exact inverse of deproject_zdepth for the pinhole
      model (sub-pixel round-trip).
  (3) A sphere's deprojected camera cloud + aabb_from_sphere projects to a 2D
      rect that brackets the mask, and the box AABB contains the camera-frame
      sphere-surface points.
  (4) The HSV center-blob fallback returns the largest near-center blob and an
      empty mask when nothing clears the threshold.

The live pieces (pyrealsense2, SAM2) are NOT touched here; the live dry-test is
run separately against the real D435.
"""

import numpy as np
import pytest

from in_hand_tracker.io.realsense_source import (
    deproject_zdepth,
    project_points,
)
from in_hand_tracker.io.deproject import deproject as euclidean_deproject
from in_hand_tracker.perception.bbox import aabb_from_sphere
from in_hand_tracker.perception.realsense_segmenter import hsv_center_blob


def _K(fx=615.0, fy=615.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


# --------------------------------------------------------------------------- #
# (1) Z-depth deproject vs synthetic K
# --------------------------------------------------------------------------- #
def test_zdepth_deproject_known_pixel():
    K = _K()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = 480, 640
    u, v = 400, 300
    Z = 0.62  # planar depth (m)

    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    depth[v, u] = Z
    mask[v, u] = True

    pts = deproject_zdepth(depth, mask, K)
    assert pts.shape == (1, 3)
    p = pts[0]

    # Planar depth: the Z component IS the depth value (NOT the euclidean norm).
    assert abs(p[2] - Z) < 1e-6
    # Back-projection: X = (u-cx)/fx*Z, Y = (v-cy)/fy*Z.
    assert abs(p[0] - (u - cx) / fx * Z) < 1e-9
    assert abs(p[1] - (v - cy) / fy * Z) < 1e-9
    # Reprojection recovers the source pixel exactly.
    u_rt = fx * p[0] / p[2] + cx
    v_rt = fy * p[1] / p[2] + cy
    assert abs(u_rt - u) < 1e-9
    assert abs(v_rt - v) < 1e-9
    # For an off-center pixel the euclidean ray length exceeds Z.
    assert np.linalg.norm(p) > Z + 1e-3


def test_zdepth_differs_from_euclidean_deproject():
    """Same value, two interpretations: Z-depth Z == value; euclidean |p| == value."""
    K = _K()
    H, W = 480, 640
    u, v = 500, 100
    val = 0.8

    grid = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    grid[v, u] = val
    mask[v, u] = True

    p_z = deproject_zdepth(grid, mask, K)[0]
    p_e = euclidean_deproject(grid, mask, K)[0]

    assert abs(p_z[2] - val) < 1e-6              # z-depth: Z == value
    assert abs(np.linalg.norm(p_e) - val) < 1e-6  # euclidean: |p| == value
    # They are genuinely different points for an off-center pixel.
    assert np.linalg.norm(p_z - p_e) > 1e-3
    assert p_z[2] > p_e[2]                       # z-depth Z is the larger one


def test_zdepth_drops_invalid_depth():
    """Zero / non-finite depth pixels are dropped even if masked."""
    K = _K()
    H, W = 20, 20
    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    mask[5:15, 5:15] = True            # 100 masked pixels
    depth[5:15, 5:15] = 0.5
    depth[7, 7] = 0.0                  # invalid (RealSense uses 0)
    depth[8, 8] = np.nan
    pts = deproject_zdepth(depth, mask, K)
    assert pts.shape[0] == 100 - 2
    assert np.isfinite(pts).all()


def test_zdepth_dict_and_matrix_intrinsics_agree():
    K = _K()
    intr = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
    H, W = 60, 80
    depth = np.full((H, W), 0.4, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    mask[20:40, 30:50] = True
    a = deproject_zdepth(depth, mask, K)
    b = deproject_zdepth(depth, mask, intr)
    assert np.allclose(a, b)


# --------------------------------------------------------------------------- #
# (2) project_points is the inverse of deproject_zdepth
# --------------------------------------------------------------------------- #
def test_project_points_inverts_deproject():
    K = _K()
    H, W = 120, 160
    rng = np.random.default_rng(0)
    mask = np.zeros((H, W), dtype=bool)
    vs = rng.integers(0, H, size=200)
    us = rng.integers(0, W, size=200)
    mask[vs, us] = True
    depth = np.zeros((H, W), dtype=np.float32)
    depth[mask] = rng.uniform(0.2, 1.2, size=int(mask.sum())).astype(np.float32)

    pts = deproject_zdepth(depth, mask, K)
    pix = project_points(pts, K)
    # deproject_zdepth iterates np.nonzero(valid) in row-major (v, u) order, and
    # project_points preserves that order, so the i-th projected pixel must be
    # the i-th masked (u, v). Compare element-wise in that native order.
    vs, us = np.nonzero(mask)
    src = np.stack([us, vs], axis=1).astype(float)       # (u, v) row-major
    assert pix.shape == src.shape
    assert np.allclose(src, pix, atol=1e-6)


def test_project_points_marks_behind_camera_nan():
    K = _K()
    pts = np.array([[0.1, 0.0, 0.5], [0.1, 0.0, -0.5], [0.0, 0.0, 0.0]])
    pix = project_points(pts, K)
    assert np.isfinite(pix[0]).all()
    assert np.isnan(pix[1]).all()
    assert np.isnan(pix[2]).all()


# --------------------------------------------------------------------------- #
# (3) sphere cloud -> AABB -> 2D rect brackets the mask
# --------------------------------------------------------------------------- #
def test_sphere_aabb_projects_to_bracketing_rect():
    K = _K()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # A sphere centered in front of the camera.
    C = np.array([0.0, 0.0, 0.6])
    r = 0.05

    # Sample visible (camera-facing) surface points, project to a mask.
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    surf = C + r * dirs
    surf = surf[surf[:, 2] < C[2]]                       # camera-facing cap
    pix = project_points(surf, K)
    pix = pix[np.isfinite(pix).all(axis=1)]
    u_min, v_min = pix[:, 0].min(), pix[:, 1].min()
    u_max, v_max = pix[:, 0].max(), pix[:, 1].max()

    box = aabb_from_sphere(C, r)
    bpix = project_points(box.corners(), K)
    bu0, bv0 = bpix[:, 0].min(), bpix[:, 1].min()
    bu1, bv1 = bpix[:, 0].max(), bpix[:, 1].max()

    # The box's projected rect brackets the cap silhouette (with a small margin).
    assert bu0 <= u_min + 1.0 and bu1 >= u_max - 1.0
    assert bv0 <= v_min + 1.0 and bv1 >= v_max - 1.0
    # And the 3D box actually contains the sampled surface points.
    assert box.contains(surf, tol=1e-6).all()


# --------------------------------------------------------------------------- #
# (4) HSV center-blob fallback
# --------------------------------------------------------------------------- #
def test_hsv_center_blob_picks_central_blob():
    H, W = 200, 200
    img = np.zeros((H, W, 3), dtype=np.uint8)         # black background, sat=0
    # A saturated red disk near the center.
    import cv2

    cv2.circle(img, (100, 100), 30, (0, 0, 255), -1)
    # A second, off-center blue blob.
    cv2.circle(img, (180, 20), 12, (255, 0, 0), -1)

    mask = hsv_center_blob(img)
    assert mask.dtype == bool
    assert mask.shape == (H, W)
    assert mask.sum() > 0
    # The chosen blob's centroid is near the center, not the corner.
    vs, us = np.nonzero(mask)
    cx, cy = us.mean(), vs.mean()
    assert abs(cx - 100) < 20 and abs(cy - 100) < 20


def test_hsv_center_blob_empty_on_blank():
    img = np.zeros((100, 100, 3), dtype=np.uint8)     # nothing saturated
    mask = hsv_center_blob(img)
    assert mask.shape == (100, 100)
    assert not mask.any()
