"""Quaternion Multiplicative EKF for object attitude (C4, doc 3.3).

A Multiplicative Extended Kalman Filter (MEKF) tracks orientation as a
*nominal* unit quaternion ``q`` plus a small-angle error state living in the
tangent space. The full quaternion is never integrated by the linear filter
(that would break normalization and the unit-norm manifold constraint);
instead the filter carries a minimal 3-vector attitude error ``dtheta`` whose
covariance is propagated, corrected, and then *reset* into the nominal
quaternion after every measurement. This keeps the nominal quaternion exactly
unit-norm at all times and the covariance well-conditioned.

State layout
------------
* Nominal attitude ``q`` -- unit quaternion. WXYZ at the public boundary;
  stored internally as XYZW (scipy ``Rotation`` convention).
* Error state ``dtheta`` (3,) -- a small rotation vector. By construction it is
  identically zero right after each reset, so only its covariance is carried.
* (Optional) angular rate ``omega`` (3,) -- when ``track_omega=True`` the rate
  is part of the state and its covariance is propagated alongside the attitude
  error (6-state filter). Otherwise ``omega`` is treated as a known input to
  :meth:`predict`.

The error quaternion uses the small-angle map ``dq ~= [1, 0.5*dtheta]`` (WXYZ),
and the attitude composition convention is LOCAL/BODY-frame error:

    q_true  =  q_nominal (x) dq      (error rotation applied on the right).

The body-frame error pairs naturally with a body-frame angular rate: the
attitude-error / rate cross term in the state transition is then simply
``I3 * dt`` (no rotation), which keeps the optional 6-state (omega-tracking)
filter consistent.

Conventions
-----------
* Quaternions are WXYZ at the data boundary (matching ``io.types``); converted
  to scipy XYZW internally.
* Angular rate ``omega`` is in the BODY frame (rad/s); :meth:`predict`
  integrates ``q_{k+1} = q_k (x) exp(0.5 * omega * dt)`` (right-multiply by the
  body-frame increment).

Public API
----------
* :meth:`predict(omega, dt)` -- propagate the nominal quaternion and GROW the
  covariance by the process noise. With no measurement, repeated predicts let
  the covariance grow without bound so a caller can declare the pose unreliable
  once :meth:`cov_trace` crosses a threshold.
* :meth:`update(q_obs, R)` -- multiplicative correction from an observed
  quaternion (WXYZ): form the error rotation, run the linear Kalman update on
  ``dtheta``, then reset the correction into the nominal quaternion.
* :meth:`quat()` -- the nominal quaternion as WXYZ.
* :meth:`cov_trace()` -- trace of the 3x3 attitude-error covariance block
  (the quantity the output policy in doc 3.4 thresholds against ``TAU``).

Pure numpy + scipy. No sim-side imports.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["OrientationMEKF", "quat_wxyz_to_xyzw", "quat_xyzw_to_wxyz"]


# --------------------------------------------------------------------------- #
# Quaternion convention helpers (WXYZ at the boundary <-> XYZW for scipy)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Reorder a WXYZ quaternion to scipy's XYZW layout."""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Reorder a scipy XYZW quaternion back to the WXYZ boundary layout."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _normalize_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    q = q / n
    # Canonical hemisphere (w >= 0) for deterministic comparisons.
    if q[3] < 0.0:
        q = -q
    return q


# --------------------------------------------------------------------------- #
# MEKF
# --------------------------------------------------------------------------- #
class OrientationMEKF:
    """Multiplicative EKF over a nominal quaternion + small-angle error state.

    Args:
        q0_wxyz: initial nominal quaternion (WXYZ). Defaults to identity.
        sigma_theta0: initial 1-sigma attitude uncertainty (rad) -> diagonal
            of the 3x3 error covariance.
        sigma_omega0: initial 1-sigma angular-rate uncertainty (rad/s); only
            used when ``track_omega=True``.
        gyro_sigma: process-noise spectral density on the attitude (rad/s/sqrt
            Hz); per-step the attitude covariance grows by ``gyro_sigma^2 * dt``
            (plus the omega contribution when tracking omega). This is what
            makes :meth:`cov_trace` grow during measurement gaps.
        omega_sigma: process-noise on the angular rate (rad/s^2/sqrt Hz); only
            used when ``track_omega=True`` (random-walk on omega).
        track_omega: if True the filter also estimates the body angular rate
            (6-state). :meth:`predict` then uses the estimated rate when no
            ``omega`` argument is supplied.
        omega0: initial angular-rate estimate (rad/s); only when tracking omega.
    """

    def __init__(
        self,
        q0_wxyz: Optional[np.ndarray] = None,
        *,
        sigma_theta0: float = 0.3,
        sigma_omega0: float = 1.0,
        gyro_sigma: float = 0.05,
        omega_sigma: float = 0.5,
        track_omega: bool = False,
        omega0: Optional[np.ndarray] = None,
    ) -> None:
        if q0_wxyz is None:
            self._q = np.array([0.0, 0.0, 0.0, 1.0])  # XYZW identity
        else:
            self._q = _normalize_xyzw(quat_wxyz_to_xyzw(q0_wxyz))

        self.track_omega = bool(track_omega)
        self._dim = 6 if self.track_omega else 3

        # Covariance: [0:3] attitude error (rad^2), [3:6] angular rate (rad/s)^2.
        P = np.zeros((self._dim, self._dim), dtype=np.float64)
        P[0:3, 0:3] = np.eye(3) * float(sigma_theta0) ** 2
        if self.track_omega:
            P[3:6, 3:6] = np.eye(3) * float(sigma_omega0) ** 2
        self.P = P

        self.gyro_sigma = float(gyro_sigma)
        self.omega_sigma = float(omega_sigma)

        if self.track_omega:
            self.omega = (
                np.zeros(3) if omega0 is None
                else np.asarray(omega0, dtype=np.float64).reshape(3)
            )
        else:
            self.omega = np.zeros(3)

    # ----- accessors ----- #
    def quat(self) -> np.ndarray:
        """Nominal quaternion as WXYZ (unit-norm, canonical w >= 0)."""
        return quat_xyzw_to_wxyz(self._q)

    def rotation(self) -> Rotation:
        """Nominal attitude as a scipy ``Rotation``."""
        return Rotation.from_quat(self._q)

    def cov_trace(self) -> float:
        """Trace of the 3x3 attitude-error covariance block (rad^2).

        This is the quantity the output policy (doc 3.4) compares against
        ``TAU``. It grows during measurement gaps and shrinks on update.
        """
        return float(np.trace(self.P[0:3, 0:3]))

    def angular_rate(self) -> np.ndarray:
        """Current body angular-rate estimate / last input (rad/s)."""
        return self.omega.copy()

    # ----- predict ----- #
    def predict(self, omega: Optional[np.ndarray] = None, dt: float = 0.0) -> None:
        """Propagate the nominal quaternion and GROW the covariance.

        Args:
            omega: body-frame angular rate (rad/s). If ``None`` and the filter
                tracks omega, the current estimate is used; if ``None`` and it
                does not, a zero rate is assumed (attitude held, covariance
                still grows from process noise).
            dt: time step (s). Non-positive ``dt`` is a no-op.
        """
        dt = float(dt)
        if dt <= 0.0:
            return

        if omega is not None:
            w = np.asarray(omega, dtype=np.float64).reshape(3)
            if self.track_omega:
                self.omega = w
        elif self.track_omega:
            w = self.omega
        else:
            w = np.zeros(3)

        # Nominal attitude propagation: right-multiply by the body increment
        # q_{k+1} = q_k (x) exp(0.5 * omega * dt).
        ang = w * dt
        dq_body = Rotation.from_rotvec(ang)
        self._q = _normalize_xyzw((self.rotation() * dq_body).as_quat())

        # Covariance propagation. With a body (right) attitude-error
        # convention and a body angular rate, the error-state transition over a
        # small step has F_theta ~= I (frame rotation is second order) and the
        # attitude error integrates the body rate error as F[theta, omega] = I*dt.
        # We add discrete process noise that GROWS the covariance.
        if self.track_omega:
            F = np.eye(6)
            # Attitude error integrates the (body) rate error over dt.
            F[0:3, 3:6] = np.eye(3) * dt
            self.P = F @ self.P @ F.T
            Q = np.zeros((6, 6))
            Q[0:3, 0:3] = np.eye(3) * (self.gyro_sigma ** 2) * dt
            Q[3:6, 3:6] = np.eye(3) * (self.omega_sigma ** 2) * dt
            self.P = self.P + Q
        else:
            # Pure attitude error: random-walk growth driven by gyro noise plus
            # the integrated effect of the (uncertain) commanded rate.
            Q = np.eye(3) * (self.gyro_sigma ** 2) * dt
            self.P = self.P + Q

        self.P = 0.5 * (self.P + self.P.T)  # keep symmetric

    # ----- update ----- #
    def update(self, q_obs_wxyz: np.ndarray, R: np.ndarray) -> None:
        """Multiplicative correction from an observed quaternion (WXYZ).

        Forms the error rotation between the observation and the nominal
        attitude, runs the linear Kalman update on the 3-vector attitude error,
        then RESETS the correction into the nominal quaternion (error -> 0).

        Args:
            q_obs_wxyz: observed quaternion (WXYZ).
            R: (3, 3) measurement covariance on the attitude residual (rad^2),
                or a scalar / (3,) treated as an isotropic / diagonal cov.
        """
        q_obs = _normalize_xyzw(quat_wxyz_to_xyzw(q_obs_wxyz))

        # Measurement covariance -> (3, 3).
        R = np.asarray(R, dtype=np.float64)
        if R.ndim == 0:
            R = np.eye(3) * float(R)
        elif R.ndim == 1:
            R = np.diag(R.reshape(3))
        else:
            R = R.reshape(3, 3)

        # Attitude residual as a 3-vector in the BODY/right-error convention:
        #   dq = q_nominal^{-1} (x) q_obs;  y = rotvec(dq).
        q_nom = Rotation.from_quat(self._q)
        q_meas = Rotation.from_quat(q_obs)
        dq = q_nom.inv() * q_meas
        y = dq.as_rotvec()  # (3,) small-angle innovation

        # Measurement model: z = dtheta + noise  -> H = [I3 | 0].
        H = np.zeros((3, self._dim))
        H[0:3, 0:3] = np.eye(3)

        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ S_inv             # (dim, 3)

        dx = K @ y                            # state correction
        dtheta = dx[0:3]

        # Reset: apply the attitude correction to the nominal quaternion
        # (RIGHT-multiply by the small body-frame rotation) and zero the error
        # state.
        dq_corr = Rotation.from_rotvec(dtheta)
        self._q = _normalize_xyzw((q_nom * dq_corr).as_quat())

        if self.track_omega:
            self.omega = self.omega + dx[3:6]

        # Joseph-form covariance update for numerical stability.
        I = np.eye(self._dim)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
