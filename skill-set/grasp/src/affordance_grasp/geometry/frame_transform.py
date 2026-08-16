from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_VITRA_CALIBRATION_RESULT_PATH = Path(
    "/home/kist/HARILAB/Grasp_fruit/configs/calibration/extrinsic_20260612_170053.json"
)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ translation
    return out


def load_base_to_camera_transform(calibration_result_path=None) -> np.ndarray:
    """Load T_{base←camera}: transforms points from camera frame to robot base (fr3_link0)."""
    path = (
        DEFAULT_VITRA_CALIBRATION_RESULT_PATH
        if calibration_result_path is None
        else Path(calibration_result_path)
    )
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    transform = np.asarray(data["T_base_camera"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"T_base_camera must be 4x4, got {transform.shape}")
    return transform


def load_world_to_base_transform(calibration_result_path=None) -> np.ndarray | None:
    """Load T_{world←base} from calibration JSON (key: 'T_world_base').

    Returns None if the key is absent — caller should fall back to identity or
    use xyzrpy_to_transform() with known launch parameters.
    """
    if calibration_result_path is None:
        return None
    path = Path(calibration_result_path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "T_world_base" not in data:
        return None
    transform = np.asarray(data["T_world_base"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"T_world_base must be 4x4, got {transform.shape}")
    return transform


def xyzrpy_to_transform(x: float, y: float, z: float,
                         roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from XYZ (metres) and RPY (radians, extrinsic XYZ)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T[:3, 3]  = [x, y, z]
    return T


def camera_to_base_transform(T_camera_target: np.ndarray, T_base_camera: np.ndarray) -> np.ndarray:
    T_camera_target = np.asarray(T_camera_target, dtype=np.float64).reshape(4, 4)
    T_base_camera = np.asarray(T_base_camera, dtype=np.float64).reshape(4, 4)
    return T_base_camera @ T_camera_target


def transform_to_xyz_quat_xyzw(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    xyz = transform[:3, 3].copy()
    quat_xyzw = R.from_matrix(transform[:3, :3]).as_quat()
    return xyz, quat_xyzw


def pose_payload(transform: np.ndarray) -> dict:
    xyz, quat_xyzw = transform_to_xyz_quat_xyzw(transform)
    return {
        "xyz": xyz.tolist(),
        "quat_xyzw": quat_xyzw.tolist(),
        "transform_matrix": np.asarray(transform, dtype=np.float64).reshape(4, 4).tolist(),
    }
