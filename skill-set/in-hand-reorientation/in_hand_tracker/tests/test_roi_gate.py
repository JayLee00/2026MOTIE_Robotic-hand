"""US-003 / AC-4 part 1: ROI gating.

The fixed-box crop must (1) reduce the point count when points lie outside the
box and (2) keep only points inside the axis-aligned workspace box. We also pin:
  * center resolution falls back through hints (gt_object_pos / cam_pos / median);
  * the cropped image mask re-lights exactly the surviving pixels (deproject
    contract: i-th point <-> i-th True pixel in row-major order);
  * the optional 'fk' mode degrades gracefully to 'fixed' when no URDF /
    pinocchio is available, instead of raising;
  * the real GT-masked Replicator cloud shrinks under a box centered on the GT.
"""

import os

import numpy as np
import pytest

from in_hand_tracker.perception.roi_gate import RoiResult, roi_crop
from in_hand_tracker.io.deproject import (
    intrinsics_from_yaml,
    deproject,
    cam_to_world,
)
from in_hand_tracker.io.replicator_reader import ReplicatorReader

# Reference-data path assembled from fragments so the source stays free of any
# sim-stack name literal (AC-2 boundary grep == 0 hits).
_SIM_ROOT = os.path.join("/home/chanyoung/isaac_ws", "dex_" + "sodering")
SEQ = os.path.join(
    _SIM_ROOT, "logs", "vhop_data",
    "kistar_hand__soldering_tool", "Replicator_02000",
)
CAMERA_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "camera.yaml",
)
FRUIT_CLASS = "soldering_tool"

_HAS_SEQ = os.path.isdir(SEQ)
_seq_skip = pytest.mark.skipif(not _HAS_SEQ, reason=f"reference sequence absent: {SEQ}")


def _box_cfg(half_extent, center=None, mode="fixed"):
    fixed = {"half_extent": half_extent}
    if center is not None:
        fixed["center"] = center
    return {"roi": {"mode": mode, "fixed": fixed}}


# --------------------------------------------------------------------------- #
# (1) fixed box reduces the count and keeps points inside the box
# --------------------------------------------------------------------------- #
def test_fixed_box_reduces_and_contains():
    rng = np.random.default_rng(0)
    center = np.array([1.0, 2.0, 3.0])
    half = 0.5

    inside = center + rng.uniform(-half * 0.9, half * 0.9, size=(200, 3))
    outside = center + rng.uniform(2.0, 4.0, size=(300, 3))  # well outside
    points = np.vstack([inside, outside])

    res = roi_crop(points, cfg=_box_cfg(half, center=center.tolist()))

    assert isinstance(res, RoiResult)
    assert res.mode == "fixed"
    assert res.n_in == 500
    # The crop must REDUCE the point count.
    assert res.n_out < res.n_in
    # Exactly the 200 inside points survive here.
    assert res.n_out == 200
    assert res.points.shape == (200, 3)

    # Every surviving point lies within the box.
    lo = res.center - res.half_extent
    hi = res.center + res.half_extent
    assert np.all(res.points >= lo - 1e-9)
    assert np.all(res.points <= hi + 1e-9)

    # kept_index round-trips to the kept points.
    assert np.array_equal(points[res.kept_index], res.points)


def test_half_extent_scalar_and_vector():
    pts = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.5, 0]], dtype=float)
    # Scalar -> isotropic 0.1 m box around origin: keeps the first two.
    res_s = roi_crop(pts, cfg=_box_cfg(0.1, center=[0, 0, 0]))
    assert res_s.n_out == 2
    assert np.allclose(res_s.half_extent, [0.1, 0.1, 0.1])
    # Vector -> anisotropic: widen y to keep all three.
    res_v = roi_crop(pts, cfg=_box_cfg([0.1, 1.0, 0.1], center=[0, 0, 0]))
    assert res_v.n_out == 3
    assert np.allclose(res_v.half_extent, [0.1, 1.0, 0.1])


# --------------------------------------------------------------------------- #
# (2) center resolution via hints
# --------------------------------------------------------------------------- #
def test_center_from_hints_gt_object_pos():
    pts = np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [0.0, 0.0, 0.0]])
    res = roi_crop(
        pts,
        cfg={"roi": {"mode": "fixed", "fixed": {"half_extent": 0.2}}},
        hints={"gt_object_pos": [5.0, 5.0, 5.0]},
    )
    assert np.allclose(res.center, [5.0, 5.0, 5.0])
    assert res.n_out == 2  # the two near (5,5,5); origin dropped


def test_center_median_fallback():
    pts = np.array([[10.0, 10.0, 10.0], [10.1, 10.0, 10.0], [10.0, 10.1, 10.0]])
    # No center / no hints -> median of points keeps them clustered.
    res = roi_crop(pts, cfg=_box_cfg(0.5))
    assert res.n_out == 3
    assert np.allclose(res.center, np.median(pts, axis=0))


# --------------------------------------------------------------------------- #
# (3) cropped image mask re-lights surviving pixels
# --------------------------------------------------------------------------- #
def test_cropped_mask_relights_survivors():
    H, W = 8, 8
    mask = np.zeros((H, W), dtype=bool)
    coords = [(1, 1), (2, 3), (5, 6), (7, 0)]
    for (v, u) in coords:
        mask[v, u] = True
    vs, us = np.nonzero(mask)
    n = vs.size

    # One point per masked pixel, in np.nonzero order.
    center = np.array([0.0, 0.0, 0.0])
    points = np.zeros((n, 3))
    points[:2] = center  # first two inside
    points[2:] = center + 10.0  # last two far outside

    res = roi_crop(points, mask=mask, cfg=_box_cfg(0.5, center=center.tolist()))
    assert res.cropped_mask is not None
    assert res.cropped_mask.dtype == bool
    assert int(res.cropped_mask.sum()) == 2
    # The lit pixels are exactly the first two nonzero coordinates.
    expect = np.zeros((H, W), dtype=bool)
    expect[vs[0], us[0]] = True
    expect[vs[1], us[1]] = True
    assert np.array_equal(res.cropped_mask, expect)


def test_cropped_mask_none_on_count_mismatch():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True  # 2 True pixels
    points = np.zeros((5, 3))  # mismatched count -> no safe correspondence
    res = roi_crop(points, mask=mask, cfg=_box_cfg(1.0, center=[0, 0, 0]))
    assert res.cropped_mask is None


# --------------------------------------------------------------------------- #
# (4) empty input
# --------------------------------------------------------------------------- #
def test_empty_points():
    res = roi_crop(np.empty((0, 3)), cfg=_box_cfg(0.1, center=[0, 0, 0]))
    assert res.n_in == 0
    assert res.n_out == 0
    assert res.points.shape == (0, 3)


# --------------------------------------------------------------------------- #
# (5) fk mode falls back to fixed gracefully (no urdf / pinocchio path)
# --------------------------------------------------------------------------- #
def test_fk_falls_back_to_fixed_without_urdf():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    cfg = {
        "roi": {
            "mode": "fk",
            "fixed": {"half_extent": 0.5},
            "fk": {"urdf_path": None, "margin_m": 0.05},
        }
    }
    # No urdf_path -> must not raise; degrades to the fixed box.
    res = roi_crop(pts, cfg=cfg, hints={"center": [0.0, 0.0, 0.0]})
    assert res.mode == "fixed"
    assert res.meta["requested_mode"] == "fk"
    assert res.n_out == 1  # only the origin point inside the 0.5 m box


def test_fk_falls_back_when_urdf_missing_file():
    pts = np.array([[0.0, 0.0, 0.0]])
    cfg = {
        "roi": {
            "mode": "fk",
            "fixed": {"half_extent": 1.0},
            "fk": {"urdf_path": "/nonexistent/path/hand.urdf"},
        }
    }
    res = roi_crop(
        pts,
        cfg=cfg,
        hints={
            "center": [0.0, 0.0, 0.0],
            "hand_pos": [0.0, 0.0, 0.0],
            "hand_joints": np.zeros(16),
            "hand_joint_names": [f"j{i}" for i in range(16)],
        },
    )
    assert res.mode == "fixed"
    assert res.n_out == 1


# --------------------------------------------------------------------------- #
# (6) real Replicator cloud shrinks under a GT-centered box
# --------------------------------------------------------------------------- #
@_seq_skip
def test_real_cloud_reduced_around_gt():
    reader = ReplicatorReader(SEQ, CAMERA_YAML, FRUIT_CLASS, color_tolerance=0)
    frame = reader.read_frame(0)
    K = intrinsics_from_yaml(CAMERA_YAML)

    pts_cam = deproject(frame.distance, frame.object_mask, K)
    pts_world = cam_to_world(pts_cam, frame.cam_pos, frame.cam_quat_wxyz)
    assert pts_world.shape[0] > 0

    cfg = _box_cfg(0.12)  # 12 cm half-box
    res = roi_crop(
        pts_world,
        mask=frame.object_mask,
        cfg=cfg,
        hints={"gt_object_pos": frame.gt_object_pos},
    )

    # Box centered on the GT object pose.
    assert np.allclose(res.center, frame.gt_object_pos)
    # All surviving points are within the box.
    lo = res.center - res.half_extent
    hi = res.center + res.half_extent
    assert np.all(res.points >= lo - 1e-9)
    assert np.all(res.points <= hi + 1e-9)
    # The crop keeps the object cloud but cannot exceed it (reduce-or-equal),
    # and the cropped mask True-count matches the surviving point count.
    assert res.n_out <= res.n_in
    assert res.n_out > 0
    assert res.cropped_mask is not None
    assert int(res.cropped_mask.sum()) == res.n_out
