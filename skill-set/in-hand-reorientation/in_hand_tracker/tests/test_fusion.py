"""C4 / doc 3.4: fused estimator output policy (position KF + orientation MEKF).

The :class:`StateEstimator` combines the continuous CV position track with the
intermittently-corrected orientation MEKF and turns them into a per-frame box +
optional pose. The contract under test mirrors doc 3.4:

  1. **spheroid path** -- every step emits a box, with NO dependence on
     orientation (a sphere has no pole); no pose is emitted.
  2. **elongated path** -- while a tag's ``q_obs`` keeps flowing the box is a
     TIGHT oriented box and the pose is ``confidence="high"``; after a tag
     dropout the attitude covariance grows past ``TAU`` and the output DEGRADES
     to a CONSERVATIVE axis-aligned box with ``confidence="bbox_only"``.
  3. **position continuity** -- the center is finite on every frame and never
     jumps across the dropout (the position track is independent of the tag).
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from in_hand_tracker.estimation.fusion import (
    DEFAULT_TAU,
    StateEstimator,
    StepOutput,
)
from in_hand_tracker.estimation import StateEstimator as StateEstimator_pkg
from in_hand_tracker.perception.bbox import Box3D


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cv_truth(t, p0, vel):
    return np.asarray(p0, dtype=np.float64) + np.asarray(vel, dtype=np.float64) * t


def _quat_wxyz(rot: Rotation) -> np.ndarray:
    q = rot.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_exported_from_package():
    assert StateEstimator_pkg is StateEstimator


# --------------------------------------------------------------------------- #
# (1) spheroid: box every step, no orientation dependence, no pose.
# --------------------------------------------------------------------------- #
def test_spheroid_emits_box_every_step_no_pose_no_orientation_dependence():
    rng = np.random.default_rng(0)
    p0 = np.array([0.30, -0.10, 0.85])
    vel = np.array([0.04, -0.01, 0.0])
    dt = 1.0 / 30.0
    a, c = 0.04, 0.042  # round (orange/tomato/peach)

    est = StateEstimator(kf_kwargs=dict(q=1e-2, r=0.003 ** 2))

    boxes = []
    for k in range(60):
        t = k * dt
        z = _cv_truth(t, p0, vel) + rng.normal(scale=0.003, size=3)
        # Deliberately feed a (wrong/random) q_obs to prove it is IGNORED for a
        # spheroid: the box must not depend on it.
        bogus_q = _quat_wxyz(Rotation.from_rotvec(rng.normal(scale=1.0, size=3)))
        out = est.step(dt, z, bogus_q, "spheroid", a, c)
        assert isinstance(out, StepOutput)
        assert isinstance(out.box, Box3D)
        assert out.pose is None
        assert out.confidence == "none"
        assert out.cov_trace_att is None  # MEKF untouched
        # Box is the isotropic bounding-sphere AABB: half-extent max(a, c).
        assert out.box.kind == "aabb_sphere"
        assert np.allclose(out.box.half_extent, max(a, c))
        assert np.all(np.isfinite(out.center))
        boxes.append(out.box)

    # A box on every single frame.
    assert len(boxes) == 60


def test_spheroid_identical_regardless_of_q_obs():
    """Same position stream, different orientation streams -> identical boxes."""
    dt = 1.0 / 30.0
    a, c = 0.05, 0.05
    z = np.array([0.1, 0.2, 0.6])

    est_a = StateEstimator()
    est_b = StateEstimator()
    rng = np.random.default_rng(7)
    for _ in range(20):
        q1 = _quat_wxyz(Rotation.from_rotvec(rng.normal(size=3)))
        q2 = _quat_wxyz(Rotation.from_rotvec(rng.normal(size=3)))
        out_a = est_a.step(dt, z, q1, "spheroid", a, c)
        out_b = est_b.step(dt, z, q2, "spheroid", a, c)
        assert np.allclose(out_a.box.half_extent, out_b.box.half_extent)
        assert np.allclose(out_a.box.center, out_b.box.center)
        assert np.allclose(out_a.box.R, out_b.box.R)


# --------------------------------------------------------------------------- #
# (2) + (3) elongated: tight/high while measured, degrade after dropout,
#            position continuous throughout.
# --------------------------------------------------------------------------- #
def test_elongated_tight_then_degrades_after_dropout_and_position_continuous():
    rng = np.random.default_rng(1)
    p0 = np.array([0.20, 0.05, 0.70])
    vel = np.array([0.06, -0.02, 0.01])     # the object is also translating
    dt = 1.0 / 30.0
    a, c = 0.03, 0.045                       # prolate lemon (c > a)
    obs_sigma = 0.01
    omega = np.array([0.0, 0.0, 0.8])        # spinning about body z

    est = StateEstimator(
        tau=DEFAULT_TAU,
        kf_kwargs=dict(q=0.5, r=obs_sigma ** 2),
        mekf_kwargs=dict(sigma_theta0=0.3, gyro_sigma=0.15),
    )

    r_true = Rotation.identity()
    centers = []
    cov_att_hist = []

    # ---- Phase A: tag VISIBLE -> tight box + high-confidence pose. ----
    n_warm = 40
    saw_high = False
    for k in range(n_warm):
        t = k * dt
        r_true = r_true * Rotation.from_rotvec(omega * dt)
        z = _cv_truth(t, p0, vel) + rng.normal(scale=0.003, size=3)
        q_obs = _quat_wxyz(
            Rotation.from_rotvec(rng.normal(scale=obs_sigma, size=3)) * r_true
        )
        out = est.step(dt, z, q_obs, "elongated", a, c,
                       q_R=np.eye(3) * obs_sigma ** 2, omega=omega)
        assert np.all(np.isfinite(out.center))
        centers.append(out.center)
        cov_att_hist.append(out.cov_trace_att)
        if k > 10:
            # After the MEKF settles, the box is tight and the pose is trusted.
            assert out.confidence == "high", out.confidence
            assert out.box.kind == "obb_tight"
            assert out.pose is not None
            c_out, q_out = out.pose
            assert np.allclose(c_out, out.center)
            assert q_out.shape == (4,)
            assert np.isclose(np.linalg.norm(q_out), 1.0, atol=1e-6)
            saw_high = True
    assert saw_high
    center_at_dropout = est.center().copy()
    last_high_center = centers[-1].copy()

    # ---- Phase B: tag DROPOUT -> orientation measurement OFF. ----
    # Position measurements ALSO stop (worst case for continuity); the position
    # filter predicts only. Orientation cov must grow past TAU and degrade.
    gap = 40
    degraded = False
    prev_center = last_high_center
    for g in range(gap):
        out = est.step(dt, None, None, "elongated", a, c, omega=omega)
        # (3) continuity: finite + small frame-to-frame step (no jump).
        assert np.all(np.isfinite(out.center))
        step_jump = np.linalg.norm(out.center - prev_center)
        # CV propagation over one 30 Hz frame at <=~0.1 m/s is mm-scale; assert
        # well under 1 cm to catch any discontinuity / reset to origin.
        assert step_jump < 0.01, f"position jumped {step_jump:.4f} m at gap {g}"
        prev_center = out.center
        centers.append(out.center)
        cov_att_hist.append(out.cov_trace_att)
        if out.confidence == "bbox_only":
            degraded = True
            assert out.box.kind == "aabb_conservative"
            # conservative half-extent is max(a, c) on every axis.
            assert np.allclose(out.box.half_extent, max(a, c))
            # The (stale) pose is retained for continuity but flagged.
            assert out.pose is not None

    assert degraded, "elongated output never degraded to 'bbox_only' after dropout"
    # The attitude covariance grew across the gap and crossed TAU.
    assert cov_att_hist[-1] > DEFAULT_TAU
    assert cov_att_hist[-1] > cov_att_hist[n_warm - 1]

    # (3) The dropout itself introduced no jump: first gap center ~ last measured.
    assert np.linalg.norm(centers[n_warm] - center_at_dropout) < 0.01

    # ---- Phase C: tag RETURNS -> orientation re-fixes, box tightens again. ----
    r_true = r_true * Rotation.from_rotvec(omega * dt)
    refixed = False
    for k in range(40):
        r_true = r_true * Rotation.from_rotvec(omega * dt)
        z = est.center() + rng.normal(scale=0.003, size=3)
        q_obs = _quat_wxyz(
            Rotation.from_rotvec(rng.normal(scale=obs_sigma, size=3)) * r_true
        )
        out = est.step(dt, z, q_obs, "elongated", a, c,
                       q_R=np.eye(3) * obs_sigma ** 2, omega=omega)
        assert np.all(np.isfinite(out.center))
        if out.confidence == "high":
            assert out.box.kind == "obb_tight"
            refixed = True
            break
    assert refixed, "orientation did not re-fix (tight box) after tags returned"


def test_elongated_no_orientation_ever_is_conservative_none():
    """An elongated object that NEVER sees a tag -> conservative box, no pose."""
    dt = 1.0 / 30.0
    a, c = 0.03, 0.05
    est = StateEstimator()
    z = np.array([0.0, 0.0, 0.5])
    for _ in range(20):
        out = est.step(dt, z, None, "elongated", a, c)
        assert out.box.kind == "aabb_conservative"
        assert np.allclose(out.box.half_extent, max(a, c))
        assert out.pose is None
        assert out.confidence == "none"
        assert np.all(np.isfinite(out.center))


def test_position_continuous_across_orientation_only_dropout():
    """Orientation drops out but POSITION measurements keep flowing.

    Position must track the moving center tightly (it is independent of the
    tag), while orientation degrades on its own schedule.
    """
    rng = np.random.default_rng(5)
    p0 = np.array([0.1, 0.1, 0.6])
    vel = np.array([0.08, 0.0, 0.0])
    dt = 1.0 / 30.0
    a, c = 0.03, 0.05
    omega = np.array([0.0, 0.5, 0.0])

    est = StateEstimator(kf_kwargs=dict(q=0.3, r=0.002 ** 2),
                         mekf_kwargs=dict(sigma_theta0=0.3, gyro_sigma=0.1))
    r_true = Rotation.identity()

    # Warm up orientation with tags.
    for k in range(30):
        t = k * dt
        r_true = r_true * Rotation.from_rotvec(omega * dt)
        z = _cv_truth(t, p0, vel) + rng.normal(scale=0.002, size=3)
        q_obs = _quat_wxyz(Rotation.from_rotvec(rng.normal(scale=0.01, size=3)) * r_true)
        est.step(dt, z, q_obs, "elongated", a, c, omega=omega)

    # Orientation gap, but position measurements continue.
    for k in range(30, 80):
        t = k * dt
        truth = _cv_truth(t, p0, vel)
        z = truth + rng.normal(scale=0.002, size=3)
        out = est.step(dt, z, None, "elongated", a, c, omega=omega)
        assert np.all(np.isfinite(out.center))
        # Position keeps tracking the (moving) truth despite no orientation.
        assert np.linalg.norm(out.center - truth) < 0.02, np.linalg.norm(out.center - truth)


def test_step_returns_finite_center_with_zero_dt():
    est = StateEstimator()
    est.step(0.0, [0.0, 0.0, 0.5], None, "spheroid", 0.04, 0.04)
    out = est.step(0.0, None, None, "spheroid", 0.04, 0.04)
    assert np.all(np.isfinite(out.center))


def test_as_dict_keys():
    est = StateEstimator()
    out = est.step(1.0 / 30.0, [0.0, 0.0, 0.5], None, "spheroid", 0.04, 0.04)
    d = out.as_dict()
    for key in ("center", "box", "pose", "confidence", "shape_class",
                "cov_trace_pos", "cov_trace_att"):
        assert key in d
