"""Fused state estimator + output policy (C4, doc 3.3 + 3.4).

This module combines the two separate estimators -- the constant-velocity
:class:`~in_hand_tracker.estimation.position_kf.PositionKF` and the quaternion
:class:`~in_hand_tracker.estimation.orientation_mekf.OrientationMEKF` -- behind a
single :class:`StateEstimator` and applies the doc 3.4 OUTPUT POLICY on top.

Design (doc 3.3 "위치/자세 분리" + 3.4 "출력 정책")
---------------------------------------------------
* **Position is always tracked and always continuous.** Every ``step`` predicts
  the CV filter over ``dt`` and, if a position measurement is available
  (ellipsoid-fit center / AprilTag center / tactile-residual center), fuses it.
  With NO measurement the filter predicts only -> the center stays finite and
  moves smoothly through tag dropouts (no jumps, no NaN). This is the
  "tag 유무와 무관하게 항상 연속 -> bbox 끊김 없음" requirement.

* **A bbox is emitted on EVERY frame** from the position center + the calibrated
  axes ``(a, c)`` via :mod:`in_hand_tracker.perception.bbox`.

* **Orientation drives the box shape ONLY for elongated objects.** Measurement
  ON/OFF is per-frame; there is NO hard switching between a "tag mode" and a
  "no-tag mode". Instead the orientation MEKF's attitude-error covariance trace
  is compared to a threshold ``TAU`` (doc 3.4):

    - ``shape_class == "spheroid"`` (orange / tomato / peach): orientation is
      irrelevant to the box (a sphere has no meaningful pole), so we emit the
      isotropic :func:`aabb_from_sphere` every frame and never touch the MEKF.
      No ``pose`` is emitted.

    - ``shape_class == "elongated"`` (lemon): if a tag has been seen recently so
      ``orientation_mekf.cov_trace() < TAU`` -> emit a TIGHT oriented box
      (:func:`tight_obb`) plus ``pose=(C, q)`` with ``confidence="high"``. Once
      tags drop out the MEKF only predicts and its covariance grows; when it
      crosses ``TAU`` we DEGRADE smoothly to a CONSERVATIVE enclosing AABB
      (:func:`conservative_aabb`, half-extent ``max(a, c)``) and flag the pose
      ``confidence="bbox_only"`` (the orientation is no longer trustworthy, but
      the position-only box is still safe). When tags return the MEKF re-fixes
      and the box tightens again -- no discontinuity in the position track.

Boundary conventions
---------------------
* Quaternions are WXYZ at the public boundary (matching ``io.types`` and the
  AprilTag detector); the MEKF converts to scipy XYZW internally.
* Pure numpy/scipy. No sim-side imports, no filterpy.

Public API
----------
* :class:`StateEstimator` with :meth:`StateEstimator.step`.
* :class:`StepOutput` (the per-frame result; also returned as a plain ``dict``
  via :meth:`StepOutput.as_dict` for callers that prefer mapping access).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..perception.bbox import (
    Box3D,
    aabb_from_sphere,
    conservative_aabb,
    tight_obb,
)
from .orientation_mekf import OrientationMEKF
from .position_kf import PositionKF

__all__ = ["StateEstimator", "StepOutput", "DEFAULT_TAU"]


# Default attitude-error covariance-trace threshold (rad^2) below which the
# orientation MEKF is trusted enough to emit a tight box + high-confidence pose.
# A fresh tag update drives the trace well below this; a multi-frame gap inflates
# it past this, flipping the output to the conservative "bbox_only" mode.
DEFAULT_TAU = 0.05


@dataclass
class StepOutput:
    """One frame of fused estimator output (doc 3.4).

    Attributes:
        center: (3,) position-KF center estimate (ALWAYS finite, every frame).
        box: :class:`~in_hand_tracker.perception.bbox.Box3D` emitted this frame.
        pose: ``(center, quat_wxyz)`` when an orientation is available, else
            ``None``. For ``confidence == "bbox_only"`` the quaternion is the
            MEKF's last (no-longer-trusted) propagated attitude, retained for
            continuity / debugging.
        confidence: ``"high"`` (tight box + trusted pose), ``"bbox_only"``
            (orientation lost, conservative box), or ``"none"`` (spheroid path /
            orientation never observed -> position-only box, no pose).
        shape_class: the class used for the branch this frame.
        cov_trace_pos: position covariance trace (confidence proxy, grows on a
            measurement gap, shrinks on update).
        cov_trace_att: attitude-error covariance trace (``None`` for spheroids).
    """

    center: np.ndarray
    box: Box3D
    pose: Optional[tuple] = None
    confidence: str = "none"
    shape_class: str = "spheroid"
    cov_trace_pos: float = 0.0
    cov_trace_att: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """Mapping view of the output (keys: center/box/pose/confidence/...)."""
        return {
            "center": self.center,
            "box": self.box,
            "pose": self.pose,
            "confidence": self.confidence,
            "shape_class": self.shape_class,
            "cov_trace_pos": self.cov_trace_pos,
            "cov_trace_att": self.cov_trace_att,
        }


def _is_spheroid(shape_class: str) -> bool:
    return str(shape_class).lower() in ("spheroid", "sphere", "round")


class StateEstimator:
    """Single fused estimator: CV position KF + quaternion MEKF + output policy.

    The two filters are intentionally SEPARATE (doc 3.3) -- position is tracked
    continuously regardless of tag availability, while orientation is only
    corrected when a tag is seen and otherwise propagated with growing
    uncertainty. :meth:`step` runs one frame and returns a :class:`StepOutput`.

    Args:
        tau: attitude-error covariance-trace threshold (rad^2). Below it the
            orientation is trusted (tight box + high-confidence pose); above it
            the output degrades to a conservative box + ``"bbox_only"``.
        position_kf: optional preconstructed :class:`PositionKF` (else a default
            is built from ``kf_kwargs``).
        orientation_mekf: optional preconstructed :class:`OrientationMEKF` (else
            a default is built from ``mekf_kwargs``).
        kf_kwargs: kwargs forwarded to :class:`PositionKF` when one is not given.
        mekf_kwargs: kwargs forwarded to :class:`OrientationMEKF` when one is not
            given.
        att_R: default measurement covariance for an orientation update -- scalar
            / (3,) / (3, 3) (rad^2). Used when :meth:`step` is given ``q_obs``
            without a per-call ``q_R``.
    """

    def __init__(
        self,
        tau: float = DEFAULT_TAU,
        *,
        position_kf: Optional[PositionKF] = None,
        orientation_mekf: Optional[OrientationMEKF] = None,
        kf_kwargs: Optional[Dict[str, Any]] = None,
        mekf_kwargs: Optional[Dict[str, Any]] = None,
        att_R: Any = 0.02 ** 2,
    ) -> None:
        self.tau = float(tau)
        self.position_kf = position_kf or PositionKF(**(kf_kwargs or {}))
        self.orientation_mekf = orientation_mekf or OrientationMEKF(
            **(mekf_kwargs or {})
        )
        self._att_R_default = att_R
        # True once an orientation measurement has ever been fused; before that a
        # spheroid never engages the MEKF and an elongated object has no pose.
        self._att_seen = False

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def center(self) -> np.ndarray:
        """(3,) current position estimate (always finite)."""
        return self.position_kf.center()

    def quat_wxyz(self) -> np.ndarray:
        """(4,) current nominal orientation (WXYZ)."""
        return self.orientation_mekf.quat()

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(
        self,
        dt: float,
        position_obs: Optional[np.ndarray],
        q_obs: Optional[np.ndarray],
        shape_class: str,
        a: float,
        c: float,
        *,
        position_R: Any = None,
        q_R: Any = None,
        omega: Optional[np.ndarray] = None,
    ) -> StepOutput:
        """Advance one frame and apply the output policy (doc 3.4).

        Args:
            dt: elapsed time since the previous step (s). ``dt <= 0`` propagates
                nothing (both filters treat it as a no-op) but a measurement, if
                present, is still fused.
            position_obs: (3,) measured center (ellipsoid-fit / tag / tactile),
                or ``None`` for a position-measurement gap. ``None`` -> predict
                only; the center stays continuous.
            q_obs: (4,) observed orientation (WXYZ), or ``None`` for an
                orientation gap. Ignored entirely for spheroids.
            shape_class: ``"spheroid"`` (position-only box) or ``"elongated"``
                (orientation-gated tight/conservative box).
            a: equatorial semi-axis (m).
            c: polar semi-axis (m).
            position_R: optional measurement covariance for the position fuse
                (scalar / (3,) / (3, 3)); ``None`` -> the KF default.
            q_R: optional measurement covariance for the orientation fuse
                (scalar / (3,) / (3, 3)); ``None`` -> the estimator default.
            omega: optional body angular rate (rad/s) used to PROPAGATE the
                orientation across the step / fill a gap; ``None`` -> the MEKF's
                own assumption (zero or its tracked rate).

        Returns:
            :class:`StepOutput` for this frame.
        """
        spheroid = _is_spheroid(shape_class)

        # ---- 1) Position: predict (CV) then fuse if a measurement exists. ----
        # predict is a no-op for dt <= 0; the center is never corrupted.
        self.position_kf.predict(dt)
        if position_obs is not None:
            self.position_kf.update(position_obs, R=position_R)
        center = self.position_kf.center()

        # ---- 2) Orientation: only relevant for elongated objects. ----
        # For a spheroid the orientation is meaningless to the box, so we skip
        # the MEKF entirely (cheapest, most robust path -- doc 3.4).
        if not spheroid:
            # Propagate the nominal attitude + grow covariance (no-op for dt<=0).
            self.orientation_mekf.predict(omega, dt)
            if q_obs is not None:
                R = q_R if q_R is not None else self._att_R_default
                self.orientation_mekf.update(q_obs, R)
                self._att_seen = True

        # ---- 3) Output policy (doc 3.4). ----
        return self._emit(center, spheroid, float(a), float(c))

    # ------------------------------------------------------------------ #
    # Output policy
    # ------------------------------------------------------------------ #
    def _emit(
        self, center: np.ndarray, spheroid: bool, a: float, c: float
    ) -> StepOutput:
        cov_pos = self.position_kf.cov_trace()

        if spheroid:
            # Orientation-independent: isotropic bounding-sphere AABB, no pose.
            box = aabb_from_sphere(center, max(abs(a), abs(c)))
            return StepOutput(
                center=center,
                box=box,
                pose=None,
                confidence="none",
                shape_class="spheroid",
                cov_trace_pos=cov_pos,
                cov_trace_att=None,
            )

        # Elongated: gate on the attitude-error covariance trace.
        cov_att = self.orientation_mekf.cov_trace()
        q_wxyz = self.orientation_mekf.quat()

        if self._att_seen and cov_att < self.tau:
            # Orientation trusted (recent tag) -> tight oriented box + pose.
            major_axis = self.orientation_mekf.rotation().apply([0.0, 0.0, 1.0])
            box = tight_obb(center, a, c, major_axis)
            return StepOutput(
                center=center,
                box=box,
                pose=(center, q_wxyz),
                confidence="high",
                shape_class="elongated",
                cov_trace_pos=cov_pos,
                cov_trace_att=cov_att,
            )

        # Orientation never seen, or lost (gap inflated cov past TAU) -> safe
        # conservative enclosing AABB (half-extent max(a, c)), pose flagged.
        box = conservative_aabb(center, a, c)
        if self._att_seen:
            # We DID have an orientation; keep the (stale) quaternion for
            # continuity but flag it as no longer trustworthy.
            pose = (center, q_wxyz)
            confidence = "bbox_only"
        else:
            pose = None
            confidence = "none"
        return StepOutput(
            center=center,
            box=box,
            pose=pose,
            confidence=confidence,
            shape_class="elongated",
            cov_trace_pos=cov_pos,
            cov_trace_att=cov_att,
        )
