"""C4 / doc 3.3: quaternion MEKF (Multiplicative EKF) for object attitude.

Scenario: a quaternion rotates under a KNOWN constant body angular rate
``omega``. The MEKF is driven with that ``omega`` every step (predict), and fed
INTERMITTENT noisy attitude observations (update) -- measurements arrive only
inside "visible" windows and stop during "gaps" (tag occluded / lost). We
assert the contract from doc 3.3 / 3.4:

  * while measurements arrive, the tracked quaternion's angular error vs. the
    true attitude stays small (the multiplicative correction works);
  * during a measurement GAP the attitude covariance GROWS (predict-only
    propagation inflates uncertainty), so a caller can declare the pose
    unreliable once ``cov_trace()`` crosses a threshold;
  * the nominal quaternion stays unit-norm at every step (the manifold
    invariant the multiplicative formulation must preserve).
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from in_hand_tracker.estimation.orientation_mekf import (
    OrientationMEKF,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _angle_between_wxyz(q_a_wxyz, q_b_wxyz) -> float:
    """Geodesic angle (rad) between two WXYZ quaternions."""
    ra = Rotation.from_quat(quat_wxyz_to_xyzw(q_a_wxyz))
    rb = Rotation.from_quat(quat_wxyz_to_xyzw(q_b_wxyz))
    return float((ra.inv() * rb).magnitude())


def _noisy_obs_wxyz(q_true_wxyz, sigma_rad, rng):
    """Perturb a true WXYZ quaternion by a small random rotation (~sigma_rad)."""
    r_true = Rotation.from_quat(quat_wxyz_to_xyzw(q_true_wxyz))
    noise = Rotation.from_rotvec(rng.normal(scale=sigma_rad, size=3))
    return quat_xyzw_to_wxyz((noise * r_true).as_quat())


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_wxyz_xyzw_roundtrip():
    q_wxyz = np.array([0.5, 0.5, -0.5, 0.5])
    assert np.allclose(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(q_wxyz)), q_wxyz)
    # WXYZ identity -> XYZW identity.
    assert np.allclose(quat_wxyz_to_xyzw([1.0, 0.0, 0.0, 0.0]),
                       [0.0, 0.0, 0.0, 1.0])


def test_quat_stays_normalized_through_predicts_and_updates():
    rng = np.random.default_rng(0)
    mekf = OrientationMEKF(sigma_theta0=0.2, gyro_sigma=0.02)
    dt = 1.0 / 30.0
    omega = np.array([0.4, -0.7, 1.1])
    r_true = Rotation.identity()
    for k in range(120):
        r_true = r_true * Rotation.from_rotvec(omega * dt)
        mekf.predict(omega, dt)
        if k % 4 == 0:
            q_obs = _noisy_obs_wxyz(
                quat_xyzw_to_wxyz(r_true.as_quat()), 0.02, rng
            )
            mekf.update(q_obs, R=np.eye(3) * (0.02 ** 2))
        q = mekf.quat()
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9)


def test_tracks_known_constant_omega_while_measured():
    """With frequent measurements the tracked attitude error stays small."""
    rng = np.random.default_rng(1)
    omega = np.array([0.3, 0.5, -0.8])           # rad/s, constant body rate
    dt = 1.0 / 30.0
    obs_sigma = 0.015

    # Seed the filter slightly OFF the truth to prove convergence.
    q0_off = quat_xyzw_to_wxyz(
        (Rotation.from_rotvec([0.1, -0.1, 0.05]) * Rotation.identity()).as_quat()
    )
    mekf = OrientationMEKF(q0_off, sigma_theta0=0.3, gyro_sigma=0.02)

    r_true = Rotation.identity()
    errors = []
    for k in range(200):
        r_true = r_true * Rotation.from_rotvec(omega * dt)
        mekf.predict(omega, dt)
        q_true_wxyz = quat_xyzw_to_wxyz(r_true.as_quat())
        q_obs = _noisy_obs_wxyz(q_true_wxyz, obs_sigma, rng)
        mekf.update(q_obs, R=np.eye(3) * (obs_sigma ** 2))
        errors.append(_angle_between_wxyz(mekf.quat(), q_true_wxyz))

    # After settling, the tracked error must stay small (well under the ~0.16
    # rad initial offset and comparable to the measurement noise level).
    settled = np.array(errors[50:])
    assert settled.mean() < 0.05, f"mean tracked error {settled.mean():.4f} rad"
    assert settled.max() < 0.1, f"max tracked error {settled.max():.4f} rad"


def test_covariance_grows_during_measurement_gap():
    """No measurements -> predict-only -> covariance must GROW monotonically."""
    omega = np.array([0.2, 0.0, 0.4])
    dt = 1.0 / 30.0
    mekf = OrientationMEKF(sigma_theta0=0.05, gyro_sigma=0.05)

    # First, a short measured burst to drive the covariance DOWN.
    rng = np.random.default_rng(2)
    r_true = Rotation.identity()
    for _ in range(20):
        r_true = r_true * Rotation.from_rotvec(omega * dt)
        mekf.predict(omega, dt)
        q_obs = _noisy_obs_wxyz(quat_xyzw_to_wxyz(r_true.as_quat()), 0.01, rng)
        mekf.update(q_obs, R=np.eye(3) * (0.01 ** 2))

    trace_after_measure = mekf.cov_trace()

    # Now the gap: predict ONLY. Covariance must increase every step.
    traces = [mekf.cov_trace()]
    for _ in range(60):
        mekf.predict(omega, dt)
        traces.append(mekf.cov_trace())
    traces = np.array(traces)

    assert np.all(np.diff(traces) > 0.0), "cov_trace did not grow every gap step"
    assert traces[-1] > traces[0]
    # The gap inflates uncertainty well beyond the measured (settled) level,
    # so a TAU threshold can flip the pose to "unreliable".
    assert traces[-1] > 3.0 * trace_after_measure


def test_update_reduces_covariance_relative_to_gap():
    """An update should pull cov_trace back below where a gap would leave it."""
    omega = np.array([0.0, 0.3, 0.0])
    dt = 1.0 / 30.0
    rng = np.random.default_rng(3)

    mekf = OrientationMEKF(sigma_theta0=0.4, gyro_sigma=0.05)
    r_true = Rotation.identity()

    # Inflate via a gap.
    for _ in range(15):
        r_true = r_true * Rotation.from_rotvec(omega * dt)
        mekf.predict(omega, dt)
    trace_gap = mekf.cov_trace()

    # One measurement.
    r_true = r_true * Rotation.from_rotvec(omega * dt)
    mekf.predict(omega, dt)
    q_obs = _noisy_obs_wxyz(quat_xyzw_to_wxyz(r_true.as_quat()), 0.01, rng)
    mekf.update(q_obs, R=np.eye(3) * (0.01 ** 2))

    assert mekf.cov_trace() < trace_gap


def test_track_omega_mode_estimates_rate():
    """6-state mode: with no omega passed to predict, it tracks the rate."""
    rng = np.random.default_rng(4)
    omega = np.array([0.0, 0.0, 0.6])            # spin about body z
    dt = 1.0 / 30.0
    obs_sigma = 0.01

    mekf = OrientationMEKF(
        sigma_theta0=0.3, gyro_sigma=0.01, omega_sigma=0.3, track_omega=True
    )
    r_true = Rotation.identity()
    for k in range(300):
        r_true = r_true * Rotation.from_rotvec(omega * dt)
        # Feed omega only occasionally; rely on the estimate otherwise.
        passed = omega if (k < 30) else None
        mekf.predict(passed, dt)
        q_obs = _noisy_obs_wxyz(quat_xyzw_to_wxyz(r_true.as_quat()), obs_sigma, rng)
        mekf.update(q_obs, R=np.eye(3) * (obs_sigma ** 2))

    est = mekf.angular_rate()
    assert np.linalg.norm(est - omega) < 0.1, f"omega est {est} vs {omega}"
    q_err = _angle_between_wxyz(mekf.quat(), quat_xyzw_to_wxyz(r_true.as_quat()))
    assert q_err < 0.06, f"attitude error {q_err:.4f} rad"


def test_predict_with_zero_dt_is_noop():
    mekf = OrientationMEKF(sigma_theta0=0.1)
    q0 = mekf.quat().copy()
    t0 = mekf.cov_trace()
    mekf.predict(np.array([1.0, 2.0, 3.0]), dt=0.0)
    assert np.allclose(mekf.quat(), q0)
    assert mekf.cov_trace() == pytest.approx(t0)
