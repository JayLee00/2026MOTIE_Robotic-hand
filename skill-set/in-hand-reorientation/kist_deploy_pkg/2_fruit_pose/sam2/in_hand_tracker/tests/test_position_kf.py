"""C4 / doc 3.3: constant-velocity position Kalman filter.

The CV KF must (a) track a moving object center from noisy position
measurements, (b) keep PREDICTING through a measurement GAP with the covariance
GROWING (measurement OFF), and (c) stay CONTINUOUS -- the bbox-center it feeds
(:meth:`PositionKF.center`) is never NaN on any frame. These mirror the doc's
"tag 유무와 무관하게 항상 연속 -> bbox 끊김 없음" requirement.
"""

import numpy as np
import pytest

from in_hand_tracker.estimation.position_kf import PositionKF
from in_hand_tracker.estimation import PositionKF as PositionKF_pkg


def _cv_truth(t, p0, vel):
    """Ground-truth constant-velocity position at time ``t``."""
    return np.asarray(p0, dtype=np.float64) + np.asarray(vel, dtype=np.float64) * t


def test_exported_from_package():
    # AC: importable both directly and via the subpackage __init__.
    assert PositionKF_pkg is PositionKF


def test_tracks_cv_trajectory_within_tolerance():
    rng = np.random.default_rng(0)
    p0 = np.array([0.30, -0.10, 0.85])
    vel = np.array([0.05, -0.02, 0.01])   # m/s
    dt = 1.0 / 30.0                        # 30 Hz
    meas_std = 0.003                       # 3 mm position noise
    R = meas_std ** 2

    # Small spectral density: the motion is smooth/slow, so trust the CV model
    # -> a cleaner velocity estimate (and still tracks position tightly).
    kf = PositionKF(q=1e-2, r=R)

    n = 120
    errs = []
    for k in range(n):
        t = k * dt
        truth = _cv_truth(t, p0, vel)
        kf.predict(dt)
        z = truth + rng.normal(scale=meas_std, size=3)
        kf.update(z, R=R)

        # never NaN, always a (3,) center.
        c = kf.center()
        assert c.shape == (3,)
        assert np.all(np.isfinite(c))

        if k > 30:  # allow the filter to converge before grading
            errs.append(np.linalg.norm(c - truth))

    rms = float(np.sqrt(np.mean(np.square(errs))))
    # Tracked center should be well inside a few mm of truth (better than the
    # raw measurement noise thanks to temporal fusion).
    assert rms < 0.004, f"tracking RMS {rms:.4f} m too large"

    # Velocity should be recovered roughly (within 2 cm/s per axis).
    assert np.allclose(kf.velocity(), vel, atol=0.02), kf.velocity()


def test_measurement_gap_predicts_and_covariance_grows():
    rng = np.random.default_rng(1)
    p0 = np.array([0.0, 0.0, 0.5])
    vel = np.array([0.10, 0.0, 0.0])      # 10 cm/s along +x
    dt = 1.0 / 30.0
    meas_std = 0.002
    R = meas_std ** 2

    kf = PositionKF(q=0.5, r=R)

    # Warm-up with measurements so velocity is well estimated.
    for k in range(40):
        t = k * dt
        kf.predict(dt)
        kf.update(_cv_truth(t, p0, vel) + rng.normal(scale=meas_std, size=3), R=R)

    trace_before_gap = kf.cov_trace()
    center_before = kf.center()
    assert np.all(np.isfinite(center_before))

    # --- measurement GAP: predict only (measurement OFF) for 30 frames ---
    gap = 30
    last_trace = trace_before_gap
    last_center = center_before
    for g in range(gap):
        c = kf.predict(dt)
        # continuous & finite every frame.
        assert c.shape == (3,) and np.all(np.isfinite(c))
        # covariance strictly grows under predict-only.
        tr = kf.cov_trace()
        assert tr > last_trace, f"cov trace did not grow at gap step {g}: {tr} <= {last_trace}"
        last_trace = tr
        last_center = c

    # Covariance grew substantially over the gap.
    assert kf.cov_trace() > trace_before_gap

    # Despite no measurements, predict-only should still extrapolate toward the
    # true (constant-velocity) position -- it keeps MOVING, not frozen.
    t_end = (40 + gap) * dt
    truth_end = _cv_truth(t_end, p0, vel)
    # The propagated center tracks the constant-velocity truth to within a few cm
    # even with NO measurements across the whole gap.
    assert np.linalg.norm(last_center - truth_end) < 0.03
    # And it actually moved from where the gap started (not stuck).
    assert np.linalg.norm(last_center - center_before) > 0.05

    # A measurement after the gap shrinks the covariance again.
    kf.update(truth_end, R=R)
    assert kf.cov_trace() < last_trace


def test_center_never_nan_with_no_measurements_at_all():
    # Pure predict-only from construction: center stays finite & continuous.
    kf = PositionKF()
    prev = kf.center()
    for _ in range(50):
        c = kf.predict(1.0 / 30.0)
        assert c.shape == (3,)
        assert np.all(np.isfinite(c)), "bbox-center went NaN under predict-only"
        prev = c
    assert np.all(np.isfinite(prev))


def test_predict_zero_or_negative_dt_is_noop():
    kf = PositionKF()
    kf.initialize([1.0, 2.0, 3.0], vel=[0.1, 0.0, 0.0])
    c0 = kf.center()
    tr0 = kf.cov_trace()
    for bad_dt in (0.0, -0.01, float("nan")):
        c = kf.predict(bad_dt)
        assert np.allclose(c, c0)
        assert kf.cov_trace() == pytest.approx(tr0)


def test_update_accepts_scalar_vector_and_matrix_R():
    kf = PositionKF()
    kf.update([0.0, 0.0, 0.5])                       # default R
    kf.update([0.01, 0.0, 0.5], R=1e-4)              # scalar
    kf.update([0.0, 0.0, 0.5], R=[1e-4, 1e-4, 1e-3]) # per-axis vector
    kf.update([0.0, 0.0, 0.5], R=np.eye(3) * 1e-4)   # full matrix
    assert np.all(np.isfinite(kf.center()))


def test_update_rejects_bad_measurement():
    kf = PositionKF()
    with pytest.raises(ValueError):
        kf.update([np.nan, 0.0, 0.5])
    with pytest.raises(ValueError):
        kf.update([0.0, 0.0])  # wrong length


def test_first_update_snaps_to_measurement():
    # Without an explicit initialize, the first update should anchor the center
    # at the measurement instead of starting from the origin.
    kf = PositionKF()
    assert not kf.initialized
    z = np.array([0.42, -0.17, 0.91])
    c = kf.update(z)
    assert kf.initialized
    assert np.allclose(c, z)
