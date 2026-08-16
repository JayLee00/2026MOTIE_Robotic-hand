"""Constant-velocity position Kalman filter (C4, doc 3.3 "position CV KF").

Tracks the 3D object center under in-hand manipulation. The whole point (doc
3.3 / 3.4) is that the *position* track must stay CONTINUOUS regardless of which
measurement is available on a given frame:

  * The state is ``[x y z vx vy vz]`` and predicts with a constant-velocity
    model over a step ``dt``.
  * A measurement (sphere-center from the ellipsoid fit, AprilTag center, or a
    tactile-residual center) updates the state with a standard linear KF step;
    the observation model is "position only" (``H = [I3 | 0]``).
  * Measurement is ON/OFF per frame. With NO measurement we simply predict; the
    covariance then GROWS, which downstream (the orientation MEKF / output
    policy) reads as falling confidence -- but ``center()`` is ALWAYS defined
    (never NaN), so the bbox path never breaks (doc: "tag 유무와 무관하게 항상
    연속 -> bbox 끊김 없음").

Pure numpy. No scipy, no filterpy (implemented directly per the doc), and NO
sim-side imports.

State / matrix conventions
--------------------------
State vector ``x = [px, py, pz, vx, vy, vz]`` (length 6).

Transition (constant velocity, step ``dt``)::

    F = [[I3, dt*I3],
         [0 ,  I3  ]]

Process noise uses the standard piecewise-constant white-acceleration (a.k.a.
"discrete white noise") model with spectral density ``q`` (per axis, units
(m/s^2)^2)::

    Q = q * [[ dt^4/4 * I3, dt^3/2 * I3],
             [ dt^3/2 * I3, dt^2   * I3]]

Observation (position only)::

    z = H x + v,  H = [I3 | 0],  v ~ N(0, R)   (R is 3x3)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["PositionKF"]


def _as_pos(z) -> np.ndarray:
    """Coerce a position-like input to a finite (3,) float64 vector."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.shape[0] != 3:
        raise ValueError(f"position measurement must be length 3, got {z.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("position measurement contains non-finite values")
    return z


def _as_R(R, default: float) -> np.ndarray:
    """Coerce a measurement-noise argument into a (3, 3) covariance matrix.

    Accepts a scalar (isotropic variance), a length-3 vector (per-axis variance,
    placed on the diagonal), or a full (3, 3) matrix.
    """
    if R is None:
        return np.eye(3) * float(default)
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 0:
        return np.eye(3) * float(R)
    if R.ndim == 1:
        if R.shape[0] != 3:
            raise ValueError(f"R vector must be length 3, got {R.shape}")
        return np.diag(R)
    if R.shape != (3, 3):
        raise ValueError(f"R matrix must be (3, 3), got {R.shape}")
    return R


class PositionKF:
    """Constant-velocity (CV) Kalman filter for a 3D position track.

    State ``[x y z vx vy vz]``. Call :meth:`predict` once per frame with the
    elapsed ``dt`` (covariance grows), then :meth:`update` ONLY when a position
    measurement is available. With no measurement, predict alone keeps the
    center continuous while inflating uncertainty.

    Args:
        q: process spectral density (per axis, (m/s^2)^2). Larger -> the track
            trusts the model less and the measurement / new motion more, and the
            covariance grows faster during measurement gaps.
        r: default isotropic measurement variance (m^2) used by :meth:`update`
            when no per-call ``R`` is supplied.
        p0_pos: initial position variance (m^2).
        p0_vel: initial velocity variance ((m/s)^2).
    """

    def __init__(
        self,
        q: float = 1.0,
        r: float = 1e-4,
        p0_pos: float = 1.0,
        p0_vel: float = 1.0,
    ) -> None:
        self.q = float(q)
        self.r_default = float(r)

        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.diag(
            np.array([p0_pos, p0_pos, p0_pos, p0_vel, p0_vel, p0_vel], dtype=np.float64)
        )
        self._initialized = False  # True once a real measurement has anchored x.

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize(self, pos, vel=None, P: Optional[np.ndarray] = None) -> None:
        """Hard-set the state to a known position (and optional velocity).

        Useful to seed the filter from the first ellipsoid-fit center so the
        track starts on the object instead of the origin.
        """
        pos = _as_pos(pos)
        self.x[:3] = pos
        if vel is not None:
            self.x[3:] = _as_pos(vel)
        else:
            self.x[3:] = 0.0
        if P is not None:
            P = np.asarray(P, dtype=np.float64)
            if P.shape != (6, 6):
                raise ValueError(f"P must be (6, 6), got {P.shape}")
            self.P = P.copy()
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------ #
    # Predict (constant velocity)
    # ------------------------------------------------------------------ #
    def predict(self, dt: float) -> np.ndarray:
        """Constant-velocity prediction over ``dt`` seconds.

        Advances position by ``v * dt`` and inflates the covariance by the
        white-acceleration process noise. ``dt <= 0`` is a no-op (returns the
        current center) so a degenerate timestamp cannot corrupt the track.

        Returns:
            (3,) predicted center (a copy).
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            return self.center()

        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)
        # keep symmetric against round-off
        self.P = 0.5 * (self.P + self.P.T)
        return self.center()

    def _process_noise(self, dt: float) -> np.ndarray:
        """Discrete white-acceleration process-noise covariance (6, 6)."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        I3 = np.eye(3)
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[:3, :3] = (dt4 / 4.0) * I3
        Q[:3, 3:] = (dt3 / 2.0) * I3
        Q[3:, :3] = (dt3 / 2.0) * I3
        Q[3:, 3:] = dt2 * I3
        return self.q * Q

    # ------------------------------------------------------------------ #
    # Update (position measurement: sphere / tag / tactile center)
    # ------------------------------------------------------------------ #
    def update(self, z, R=None) -> np.ndarray:
        """Fuse a 3D position measurement (the measurement-ON path).

        The observation model is position-only (``H = [I3 | 0]``); any of the
        sphere-center / tag-center / tactile-residual-center observations plug in
        here as a (3,) ``z``. On the FIRST measurement (before any
        :meth:`initialize`) the position is snapped to ``z`` so the track does
        not have to "fly in" from the origin; the velocity is still learned by
        the normal KF math thereafter.

        Args:
            z: (3,) measured center.
            R: measurement covariance -- scalar / (3,) / (3, 3) / ``None``
                (``None`` -> the constructor's default isotropic variance).

        Returns:
            (3,) posterior center (a copy).
        """
        z = _as_pos(z)
        Rm = _as_R(R, self.r_default)

        if not self._initialized:
            # First real anchor: snap position, zero velocity, set position
            # covariance to the measurement noise (velocity stays uncertain).
            self.x[:3] = z
            self.x[3:] = 0.0
            self.P[:3, :3] = Rm
            self._initialized = True
            return self.center()

        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        y = z - H @ self.x                       # innovation (3,)
        S = H @ self.P @ H.T + Rm                 # innovation covariance (3,3)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)   # Kalman gain (6,3)
        except np.linalg.LinAlgError:
            return self.center()

        self.x = self.x + K @ y
        I6 = np.eye(6)
        # Joseph-form covariance update (numerically stable, stays PSD).
        ImKH = I6 - K @ H
        self.P = ImKH @ self.P @ ImKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.center()

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def center(self) -> np.ndarray:
        """(3,) current position estimate (a copy). NEVER NaN."""
        return self.x[:3].copy()

    def velocity(self) -> np.ndarray:
        """(3,) current velocity estimate (a copy)."""
        return self.x[3:].copy()

    def cov_trace(self) -> float:
        """Trace of the 3x3 POSITION covariance block.

        Grows under predict-only (measurement OFF), shrinks on update. The
        output policy reads this as a position-confidence proxy.
        """
        return float(np.trace(self.P[:3, :3]))

    def position_cov(self) -> np.ndarray:
        """(3, 3) position covariance block (a copy)."""
        return self.P[:3, :3].copy()
