"""Core dataclasses shared across the pipeline.

All arrays are numpy. Conventions (verified against the recorded Replicator
sequences):
  * rgb / seg_color are BGR uint8 (as loaded by cv2.imread).
  * distance is float32 (H, W), the EUCLIDEAN ray length in meters (NOT Z).
  * object_mask is bool (H, W).
  * ALL quaternions are WXYZ (Isaac/USD convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Frame:
    """One time step of a recorded (or live) sequence.

    Geometry note: ``distance`` is the per-pixel EUCLIDEAN ray length (meters),
    so ``deproject`` must scale the *unit* ray direction by it, not treat it as
    a Z value. Camera->world uses the per-sequence extrinsic (cam_pos,
    cam_quat_wxyz) from camera_pose.json.
    """

    idx: int
    rgb: np.ndarray                     # (H, W, 3) BGR uint8
    distance: np.ndarray                # (H, W)    float32, euclidean ray length (m)
    seg_color: np.ndarray               # (H, W, 3) BGR uint8 (colorized semantic seg)
    object_mask: np.ndarray             # (H, W)    bool

    # Camera extrinsic (per sequence; world frame).
    cam_pos: np.ndarray                 # (3,)  float
    cam_quat_wxyz: np.ndarray           # (4,)  float, WXYZ

    # Ground-truth object pose (world frame).
    gt_object_pos: np.ndarray           # (3,)  float
    gt_object_quat_wxyz: np.ndarray     # (4,)  float, WXYZ

    # Hand root pose (world frame) + joint state.
    hand_pos: np.ndarray                # (3,)  float
    hand_quat_wxyz: np.ndarray          # (4,)  float, WXYZ
    hand_joints: np.ndarray             # (16,) float
    hand_joint_names: List[str] = field(default_factory=list)


@dataclass
class GtShape:
    """Ground-truth object shape (calibration target / oracle).

    ``a`` is the equatorial semi-axis, ``c`` the polar semi-axis, BOTH in
    ABSOLUTE meters (already the final radii -- not unit values to be scaled).
    ``scale`` is the generator's prim xform scale, recorded as metadata only
    (a scalar or an ``[sx, sy, sz]`` list); it MUST NOT be multiplied into
    ``a``/``c``. ``shape_class`` is a free-form label ("spheroid"|"elongated").
    """

    a: float
    c: float
    scale: object = 1.0
    shape_class: str = "spheroid"


@dataclass
class FitResult:
    """Placeholder for the M2 axis-fixed ellipsoid fit output (import-safe).

    ``ellipsoid_fit`` is the RANSAC fit center that AC-10 grades. It is distinct
    from the diagnostic ``surface_centroid`` (AC-3b), which carries a
    one-radius self-occlusion bias and must NEVER feed the fit-center path.
    """

    center: Optional[np.ndarray] = None        # (3,) ellipsoid_fit center (world)
    R_wo: Optional[np.ndarray] = None          # (3, 3) object orientation (world)
    a: Optional[float] = None
    c: Optional[float] = None
    inlier_ratio: Optional[float] = None
    n_points: Optional[int] = None
    ok: bool = False
