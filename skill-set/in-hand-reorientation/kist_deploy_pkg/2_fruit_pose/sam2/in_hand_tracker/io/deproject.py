"""Deprojection + camera->world geometry (ported from the sim perception stack).

Clean-room reimplementation in numpy + scipy. NO sim-side (recorder / Isaac)
imports.

Critical conventions (verified on Replicator_02000):
  * ``distance`` is the EUCLIDEAN ray length, so:
        point_cam = dir_unit * distance,  dir_unit = normalize(Kinv @ [u, v, 1])
    Do NOT treat the distance value as Z.
  * Camera->world uses the TRANSPOSED world rotation relative to the brownfield
    helper:
        points_world = (R_wc.T @ (diag(1,-1,-1) @ points_cam.T)).T + cam_pos
    where R_wc is the rotation matrix from the camera's WXYZ quaternion and
    diag(1,-1,-1) is the GL->CV (OpenGL/USD -> OpenCV) flip. The LITERAL
    (non-transposed) ``R_wc @ flip`` is WRONG for this data (~0.69 m vs ~0.06 m
    to the GT centroid on frame 0).
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

# OpenGL/USD camera frame (+X right, +Y up, -Z fwd) -> OpenCV (+X right, +Y
# down, +Z fwd). Involutive, det = +1 (180 deg about camera X).
_GL2CV = np.diag([1.0, -1.0, -1.0])


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Assemble a 3x3 pinhole intrinsics matrix K."""
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def intrinsics_from_yaml(camera_yaml: Union[str, dict]) -> np.ndarray:
    """Build K from a camera.yaml path (or an already-loaded dict)."""
    if isinstance(camera_yaml, dict):
        cfg = camera_yaml
    else:
        import yaml

        with open(camera_yaml) as f:
            cfg = yaml.safe_load(f)
    return intrinsics_matrix(
        float(cfg["fx"]), float(cfg["fy"]), float(cfg["cx"]), float(cfg["cy"])
    )


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """WXYZ quaternion -> 3x3 rotation matrix (via scipy, xyzw at the boundary)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    # scipy expects xyzw; convert wxyz -> xyzw at the boundary.
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def deproject(distance: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Deproject masked pixels to the CAMERA frame using EUCLIDEAN ray length.

    Args:
        distance: (H, W) euclidean ray length per pixel (meters).
        mask: (H, W) bool; only True pixels are deprojected.
        K: 3x3 intrinsics.

    Returns:
        (N, 3) camera-frame points (OpenCV convention, +Z forward), where N is
        the number of True mask pixels. ``point_cam = dir_unit * distance``.
    """
    distance = np.asarray(distance)
    mask = np.asarray(mask, dtype=bool)
    if distance.shape != mask.shape:
        raise ValueError(
            f"distance {distance.shape} and mask {mask.shape} must match"
        )

    vs, us = np.nonzero(mask)
    if vs.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Ray directions in the OpenCV camera frame: Kinv @ [u, v, 1].
    x = (us - cx) / fx
    y = (vs - cy) / fy
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=1).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    d = distance[vs, us].astype(np.float64)
    return dirs * d[:, None]


def cam_to_world(
    points_cam: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat_wxyz: np.ndarray,
    convention: str = "transposed",
) -> np.ndarray:
    """Camera-frame points (N, 3) -> world frame.

    The recorded datasets disagree on the camera-pose convention (verified
    independently, frame 0 masked-cloud centroid vs GT object pose):
      * ``"transposed"`` -> ``(R_wc.T @ flip)`` : correct for the ORIGINAL
        soldering dataset (~0.06 m to GT; literal ~0.69 m).
      * ``"literal"``    -> ``(R_wc @ flip)``   : correct for the REGENERATED
        fruit datasets (~0.02 m to GT; transposed ~0.86 m).
    Use :func:`detect_convention` (with the GT object pose) to pick per
    sequence; ``"transposed"`` stays the default for backward compatibility.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    R_wc = _quat_wxyz_to_matrix(cam_quat_wxyz)

    if points_cam.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    flipped = _GL2CV @ points_cam.T          # (3, N)
    if convention == "transposed":
        world = (R_wc.T @ flipped).T + cam_pos    # (N, 3)
    elif convention == "literal":
        world = (R_wc @ flipped).T + cam_pos      # (N, 3)
    else:
        raise ValueError(
            f"convention must be 'transposed' or 'literal', got {convention!r}"
        )
    return world


def detect_convention(
    points_cam: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat_wxyz: np.ndarray,
    gt_obj_pos: np.ndarray,
) -> tuple[str, float]:
    """Pick the cam->world convention that lands the masked-cloud centroid
    nearest the GT object position.

    The two conventions differ by ~0.8 m on real data, so the choice is
    unambiguous. Returns ``(convention, distance_to_gt)``. Requires a GT
    object position (sim data); for GT-less data fall back to a configured
    default. Falls back to ``"transposed"`` when ``points_cam`` is empty.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    if points_cam.shape[0] == 0:
        return "transposed", float("inf")
    gt = np.asarray(gt_obj_pos, dtype=np.float64).reshape(3)
    best, best_d = "transposed", float("inf")
    for conv in ("transposed", "literal"):
        centroid = cam_to_world(
            points_cam, cam_pos, cam_quat_wxyz, convention=conv
        ).mean(axis=0)
        d = float(np.linalg.norm(centroid - gt))
        if d < best_d:
            best_d, best = d, conv
    return best, best_d


def cam_to_world_literal(
    points_cam: np.ndarray, cam_pos: np.ndarray, cam_quat_wxyz: np.ndarray
) -> np.ndarray:
    """The LITERAL (non-transposed) brownfield variant: ``R_wc @ flip``.

    Exposed ONLY so tests can pin that the transpose is what is required (this
    variant is WRONG for the recorded data, ~0.69 m off). Do not use it.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    R_wc = _quat_wxyz_to_matrix(cam_quat_wxyz)
    if points_cam.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    flipped = _GL2CV @ points_cam.T
    return (R_wc @ flipped).T + cam_pos


def surface_centroid(points_world: np.ndarray) -> np.ndarray:
    """DIAGNOSTIC ONLY: mean of the visible world-frame surface points (3,).

    This is the AC-3b sanity check. It carries a ~one-radius self-occlusion
    bias (only the camera-facing surface is seen) and is intentionally a
    SEPARATE, clearly-named field. NEVER feed this into the ellipsoid
    fit-center (AC-10 / ``ellipsoid_fit``) path.
    """
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if points_world.shape[0] == 0:
        return np.full(3, np.nan)
    return points_world.mean(axis=0)
