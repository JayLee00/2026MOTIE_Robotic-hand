#!/usr/bin/env python3
"""
Top-down grasp stage for Grasp_fruit pipeline.

From a segmentation mask + depth image, computes a top-down EE pose
(world -Z approach) at the object centroid with a fixed KISTAR hand pose
(thumb j1=j2=90°, rest=0).

Usage:
    python scripts/run_topdown_grasp.py \
        --input  data/raw/scene.npz \
        --mask   data/interim/scene_mask.png \
        --output data/outputs/

Output files:
    {output_dir}/{stem}_topdown_summary.json   — grasp summary (로봇 전송용)
    {output_dir}/{stem}_topdown_overlay.png    — visualisation
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from affordance_grasp.geometry.frame_transform import (
    DEFAULT_VITRA_CALIBRATION_RESULT_PATH,
    invert_transform,
    load_base_to_camera_transform,
    load_world_to_base_transform,
    xyzrpy_to_transform,
    pose_payload,
)
from affordance_grasp.io.dataset_io import ensure_dir, load_rgbd_bundle, load_rgb_image, save_json

# 손 자세 / encoder 변환 — configs/hand.yaml 에서 관리 (utils/hand.py 로드)
from utils.hand import (
    FIXED_JOINT_ANGLES_DRO,
    HAND_INIT_ENC, HAND_RELEASE_ENC,
    DRO_TO_HW as _DRO_TO_HW,
    RAD_TO_ENC,
)

# 팔 설정 — configs/arm.yaml 에서 관리 (utils/arm.py 로드)
from utils.arm import (
    APPROACH_OFFSET_M, GRASP_Z_OFFSET_M,
    EE_YAW_DEG, EE_X_OFFSET_M, EE_Y_OFFSET_M,
    TOP_Z_PCT, Z_TOP_PCT,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def backproject_mask(depth: np.ndarray, K: np.ndarray, mask: np.ndarray,
                     depth_scale: float) -> np.ndarray:
    """Back-project masked depth pixels → 3-D points in camera frame (metres)."""
    K = np.asarray(K, dtype=np.float64)
    depth_m = depth.astype(np.float64) / depth_scale
    ys, xs = np.where(mask & (depth_m > 0))
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    zs = depth_m[ys, xs]
    pts = np.stack([
        (xs - K[0, 2]) * zs / K[0, 0],
        (ys - K[1, 2]) * zs / K[1, 1],
        zs,
    ], axis=1).astype(np.float32)
    return pts


def statistical_outlier_removal(pts: np.ndarray, k: int = 20,
                                std_ratio: float = 2.0) -> np.ndarray:
    """SOR: k-NN 평균 거리가 mean + std_ratio*std 초과인 점 제거."""
    if len(pts) <= k:
        return pts
    tree       = cKDTree(pts)
    dists, _   = tree.query(pts, k=k + 1)   # 첫 번째는 자기 자신(거리=0)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold  = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists <= threshold]


def build_topdown_pose(obj_pts_world: np.ndarray, z_offset: float,
                       ee_yaw_deg: float = EE_YAW_DEG,
                       ee_x_offset_m: float = EE_X_OFFSET_M,
                       ee_y_offset_m: float = EE_Y_OFFSET_M) -> tuple:
    """Return (T_4x4, info_dict) for a top-down grasp in world frame.

    Improvements applied:
      1. z_top = Z_TOP_PCT percentile (not raw max) — noise robust
      2. centroid (cx, cy) from top TOP_Z_PCT% Z points — heaviest/thickest region
      3. yaw from PCA of ALL points' XY projection — full shape orientation
         + EE_YAW_DEG mount offset added on top
    """
    z_vals = obj_pts_world[:, 2]

    # 1. z_top: percentile (노이즈에 강함)
    z_top = float(np.percentile(z_vals, Z_TOP_PCT))

    # 2. Top cluster centroid: z 상위 TOP_Z_PCT% 점들의 XY 평균
    z_thresh = float(np.percentile(z_vals, 100.0 - TOP_Z_PCT))
    top_pts  = obj_pts_world[z_vals >= z_thresh]
    cx = float(top_pts[:, 0].mean())
    cy = float(top_pts[:, 1].mean())

    # 3. PCA yaw: 물체 전체 XY 투영의 장축 방향 (물체 모양 기반)
    xy   = obj_pts_world[:, :2].astype(np.float64)
    xy_c = xy - xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy_c, full_matrices=False)
    major = Vt[0]                                      # 분산 최대 방향
    alpha = math.atan2(float(major[1]), float(major[0]))
    projected = xy_c @ major
    half_len = float(np.percentile(np.abs(projected), 95.0))

    # 중지 방향 ⊥ PCA 장축 → +90° 보정
    alpha_perp = alpha + math.pi / 2

    # 원기둥 180° 대칭: 두 후보 중 기본 yaw(EE_YAW_DEG)로부터 편차 최소 방향 선택
    def _wrap_pi(a: float) -> float:
        while a > math.pi:  a -= 2.0 * math.pi
        while a < -math.pi: a += 2.0 * math.pi
        return a

    c1 = _wrap_pi(alpha_perp)
    c2 = _wrap_pi(alpha_perp + math.pi)
    alpha_final = c1 if abs(c1) <= abs(c2) else c2

    # ee_yaw_deg(-45°) = 핸드 장착 오프셋; alpha_final = PCA 수직 보정
    yaw = math.radians(ee_yaw_deg) + alpha_final

    c, s = math.cos(yaw), math.sin(yaw)

    # Rx(π): EE Z → world -Z (straight down against gravity)
    R_down = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ], dtype=np.float64)

    # Rz(yaw): 핸드 장착 오프셋 + PCA 물체 장축 보정
    R_yaw = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # offset은 ee_yaw_deg(-45°)만 적용 시 캘리브레이션된 값이므로,
    # PCA 추가 회전(alpha_final)만큼 2D 회전해서 world XY로 변환
    cf, sf = math.cos(alpha_final), math.sin(alpha_final)
    ox = ee_x_offset_m * cf - ee_y_offset_m * sf
    oy = ee_x_offset_m * sf + ee_y_offset_m * cf

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_yaw @ R_down
    T[:3, 3]  = [cx + ox, cy + oy, z_top + z_offset]

    info = {
        "cx": cx, "cy": cy, "z_top": z_top,
        "pca_alpha_deg":      math.degrees(alpha),
        "pca_alpha_perp_deg": math.degrees(alpha_perp),
        "pca_alpha_final_deg": math.degrees(alpha_final),
        "pca_center_xy": xy.mean(axis=0).tolist(),
        "pca_major_xy": major.tolist(),
        "pca_half_length_m": half_len,
        "total_yaw_deg": math.degrees(yaw),
        "top_cluster_pts": len(top_pts),
    }
    return T, info


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def project_world_to_pixel(point_world: np.ndarray, K: np.ndarray,
                           T_world_camera: np.ndarray) -> tuple[int, int] | None:
    T_camera_world = invert_transform(T_world_camera)
    point_h = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    point_cam = T_camera_world @ point_h
    if point_cam[2] <= 1e-6:
        return None

    u = K[0, 0] * point_cam[0] / point_cam[2] + K[0, 2]
    v = K[1, 1] * point_cam[1] / point_cam[2] + K[1, 2]
    return int(round(u)), int(round(v))


def draw_pca_overlay(canvas: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray,
                     pose_info: dict) -> None:
    center_xy = np.asarray(pose_info["pca_center_xy"], dtype=np.float64)
    major_xy = np.asarray(pose_info["pca_major_xy"], dtype=np.float64)
    half_len = float(pose_info["pca_half_length_m"])
    z = float(pose_info["z_top"])

    p0_world = np.array([center_xy[0], center_xy[1], z], dtype=np.float64)
    p1_world = np.array([
        center_xy[0] + major_xy[0] * half_len,
        center_xy[1] + major_xy[1] * half_len,
        z,
    ], dtype=np.float64)
    p2_world = np.array([
        center_xy[0] - major_xy[0] * half_len,
        center_xy[1] - major_xy[1] * half_len,
        z,
    ], dtype=np.float64)

    p0 = project_world_to_pixel(p0_world, K, T_world_camera)
    p1 = project_world_to_pixel(p1_world, K, T_world_camera)
    p2 = project_world_to_pixel(p2_world, K, T_world_camera)
    if p0 is None or p1 is None or p2 is None:
        return

    h, w = canvas.shape[:2]
    if not (0 <= p0[0] < w and 0 <= p0[1] < h):
        return

    cv2.line(canvas, p2, p1, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.arrowedLine(canvas, p0, p1, (0, 200, 255), 3, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(canvas, p0, 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "PCA major", (p0[0] + 8, p0[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


def build_overlay_canvas(rgb_bgr: np.ndarray, mask: np.ndarray,
                         xyz_base: list, K: np.ndarray,
                         T_world_camera: np.ndarray, pose_info: dict) -> np.ndarray:
    canvas = rgb_bgr.copy()
    green_overlay = canvas.copy()
    green_overlay[mask] = (0, 180, 60)
    canvas = cv2.addWeighted(green_overlay, 0.45, canvas, 0.55, 0.0)
    draw_pca_overlay(canvas, K, T_world_camera, pose_info)

    lines = [
        "Grasp: TOP-DOWN",
        f"x={xyz_base[0]:.3f}  y={xyz_base[1]:.3f}",
        f"z={xyz_base[2]:.3f}",
        f"pca={pose_info['pca_alpha_deg']:.1f} deg",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2, cv2.LINE_AA)
    return canvas


def save_overlay(canvas: np.ndarray, stem: str, output_dir: Path) -> Path:
    out_path = output_dir / f"{stem}_topdown_overlay.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved overlay: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", help="NPZ bundle (keys: rgb, depth, K)")
    src.add_argument("--rgb",   help="RGB image path (requires --depth)")
    p.add_argument("--depth",   help="Depth NPZ when --rgb is used")

    p.add_argument("--mask",     required=True, help="Binary mask PNG (255=object)")
    p.add_argument("--output",   default=None,  help="Output directory (default: data/outputs/)")
    p.add_argument("--query",    default=None,
                   help="물체 이름 — fruits.yaml 오프셋 조회용 (optional)")
    p.add_argument("--calibration", default=None,
                   help="Calibration JSON with T_base_camera (4x4). "
                        f"Default: {DEFAULT_VITRA_CALIBRATION_RESULT_PATH}")
    p.add_argument("--z_offset", type=float, default=GRASP_Z_OFFSET_M,
                   help=f"Z above object top in world frame, metres (default: arm.yaml grasp_z_offset_m={GRASP_Z_OFFSET_M})")
    p.add_argument("--x_offset", type=float, default=None,  #추가
                   help="EE X offset (m). CLI 우선 — 주면 arm.yaml/fruits.yaml 무시")  #추가
    p.add_argument("--y_offset", type=float, default=None,  #추가
                   help="EE Y offset (m). CLI 우선 — 주면 arm.yaml/fruits.yaml 무시")  #추가
    p.add_argument("--depth_scale", type=float, default=1.0,
                   help="Depth divisor to convert raw values to metres (default: 1.0)")
    p.add_argument("--hand_pose", default=None,
                   help="JSON file with 'joint_angles' list (16 floats, DRO order). "
                        "Overrides built-in fixed pose.")

    # world → base transform (로봇 베이스가 world에 대해 기울어진 경우 필수)
    # 캘리브레이션 JSON에 'T_world_base' 키가 있으면 자동으로 읽힘.
    # 없을 경우 아래 xyzrpy 인자로 직접 지정 (fr3_kistar.launch.py 기본값과 동일).
    p.add_argument("--robot_base_x",     type=float, default=None,
                   help="world→base translation X (m). 캘리브 JSON에 T_world_base 없을 때 사용.")
    p.add_argument("--robot_base_y",     type=float, default=None)
    p.add_argument("--robot_base_z",     type=float, default=None)
    p.add_argument("--robot_base_roll",  type=float, default=None,
                   help="world→base roll  (rad)")
    p.add_argument("--robot_base_pitch", type=float, default=None)
    p.add_argument("--robot_base_yaw",   type=float, default=None)

    p.add_argument("--preview", action="store_true",
                   help="overlay 창을 띄워 확인 후 아무 키를 누르면 계속 진행.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output) if args.output else ROOT / "data" / "outputs"
    ensure_dir(output_dir)

    # --- Load inputs ---
    if args.input:
        bundle = load_rgbd_bundle(args.input)
        rgb_bgr = bundle["rgb"]
        depth   = bundle["depth"]
        K       = bundle["K"]
        stem    = Path(args.input).stem
    else:
        if args.depth is None:
            print("[ERROR] --depth required when --rgb is used.")
            sys.exit(1)
        depth_bundle = load_rgbd_bundle(args.depth)
        rgb_bgr = load_rgb_image(args.rgb)
        depth   = depth_bundle["depth"]
        K       = depth_bundle["K"]
        stem    = Path(args.rgb).stem

    mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"[ERROR] Cannot read mask: {args.mask}")
        sys.exit(1)
    mask = mask_img > 127

    print(f"RGB shape:   {rgb_bgr.shape}")
    print(f"Depth shape: {depth.shape}  dtype={depth.dtype}")
    print(f"Mask active: {mask.sum()} px")

    if mask.sum() == 0:
        print("[ERROR] Mask is empty — no object to grasp.")
        sys.exit(1)

    # --- Load hand pose ---
    if args.hand_pose:
        with open(args.hand_pose) as f:
            hp = json.load(f)
        joint_angles_dro = list(hp["joint_angles"])
        if len(joint_angles_dro) != 16:
            print(f"[ERROR] hand_pose JSON must have 16 joint_angles, got {len(joint_angles_dro)}.")
            sys.exit(1)
        print(f"Loaded hand pose: {args.hand_pose}")
    else:
        joint_angles_dro = list(FIXED_JOINT_ANGLES_DRO)
        print("Hand pose: built-in (thumb j1=j2=90°, rest=0)")

    # --- Back-project mask → camera-frame point cloud ---
    obj_pts_cam = backproject_mask(depth, K, mask, args.depth_scale)
    print(f"Object points (camera frame): {len(obj_pts_cam)}")

    if len(obj_pts_cam) == 0:
        print("[ERROR] No valid depth under mask. Check depth image and --depth_scale.")
        sys.exit(1)

    # --- Statistical Outlier Removal ---
    obj_pts_cam = statistical_outlier_removal(obj_pts_cam, k=20, std_ratio=2.0)
    print(f"Object points after SOR     : {len(obj_pts_cam)}")

    if len(obj_pts_cam) == 0:
        print("[ERROR] SOR 후 포인트가 없습니다. std_ratio를 높여보세요.")
        sys.exit(1)

    # --- Load calibration: T_{base←camera} ---
    calib_path = args.calibration or str(DEFAULT_VITRA_CALIBRATION_RESULT_PATH)
    try:
        T_base_camera = load_base_to_camera_transform(calib_path)
        print(f"Calibration (T_base_camera): {calib_path}")
    except FileNotFoundError:
        if args.calibration:
            raise
        print("[ERROR] Default calibration file not found. Use --calibration.")
        sys.exit(1)

    # --- Determine T_{world←base} ---
    # 우선순위: 1) 캘리브 JSON의 'T_world_base' 키
    #           2) --robot_base_* CLI 인자
    #           3) 없으면 identity (base = world 로 간주, 경고 출력)
    T_world_base = load_world_to_base_transform(calib_path)

    if T_world_base is None:
        xyzrpy_args = [args.robot_base_x, args.robot_base_y, args.robot_base_z,
                       args.robot_base_roll, args.robot_base_pitch, args.robot_base_yaw]
        if all(v is not None for v in xyzrpy_args):
            T_world_base = xyzrpy_to_transform(
                args.robot_base_x, args.robot_base_y, args.robot_base_z,
                args.robot_base_roll, args.robot_base_pitch, args.robot_base_yaw,
            )
            print(f"T_world_base: from CLI args  "
                  f"xyz=({args.robot_base_x:.3f},{args.robot_base_y:.3f},{args.robot_base_z:.3f})  "
                  f"rpy=({args.robot_base_roll:.3f},{args.robot_base_pitch:.3f},{args.robot_base_yaw:.3f})")
        else:
            T_world_base = np.eye(4, dtype=np.float64)
            print("[WARNING] T_world_base not provided — assuming base frame == world frame.")
            print("          로봇 베이스가 기울어진 경우 --robot_base_* 인자 또는")
            print("          캘리브 JSON에 'T_world_base' 키를 추가하세요.")

    # T_{world←camera} = T_{world←base} @ T_{base←camera}
    T_world_camera = T_world_base @ T_base_camera
    print(f"T_world_base  R:\n{T_world_base[:3,:3]}")
    print(f"T_world_camera t: {T_world_camera[:3,3].tolist()}")

    # Camera frame → world frame
    ones = np.ones((len(obj_pts_cam), 1), dtype=np.float64)
    pts_h = np.hstack([obj_pts_cam.astype(np.float64), ones])
    obj_pts_world = (T_world_camera @ pts_h.T).T[:, :3]

    # --- Per-fruit offset override (configs/fruits.yaml) ---
    ee_yaw_deg    = EE_YAW_DEG
    ee_x_offset_m = EE_X_OFFSET_M
    ee_y_offset_m = EE_Y_OFFSET_M
    z_offset      = args.z_offset

    fruits_path = ROOT / "configs" / "fruits.yaml"
    if args.query and fruits_path.exists():
        fruits_cfg = yaml.safe_load(fruits_path.read_text()) or {}
        key        = args.query.strip().lower()
        override   = fruits_cfg.get(key)
        if override:
            ee_yaw_deg    = float(override.get('yaw_deg',          ee_yaw_deg))
            ee_x_offset_m = float(override.get('x_offset_m',       ee_x_offset_m))
            ee_y_offset_m = float(override.get('y_offset_m',       ee_y_offset_m))
            z_offset      = float(override.get('grasp_z_offset_m', z_offset))
            print(f"[fruits.yaml] {key!r} → "
                  f"yaw={ee_yaw_deg:.1f}°  x={ee_x_offset_m:.3f}m  "
                  f"y={ee_y_offset_m:.3f}m  z_offset={z_offset:.3f}m")
        else:
            print(f"[fruits.yaml] {key!r} 항목 없음 → arm.yaml 기본값 사용")

    # CLI --x_offset/--y_offset 는 fruits.yaml/arm.yaml 보다 우선 적용  #추가
    if args.x_offset is not None:                                       #추가
        ee_x_offset_m = args.x_offset                                   #추가
    if args.y_offset is not None:                                       #추가
        ee_y_offset_m = args.y_offset                                   #추가
    if args.x_offset is not None or args.y_offset is not None:          #추가
        print(f"[CLI] x_offset={ee_x_offset_m:.3f}m  y_offset={ee_y_offset_m:.3f}m (CLI 우선)")  #추가

    # --- Build top-down EE pose in world frame ---
    T_world_ee, pose_info = build_topdown_pose(
        obj_pts_world, z_offset, ee_yaw_deg, ee_x_offset_m, ee_y_offset_m)

    print(f"Object centroid (world): x={pose_info['cx']:.3f}  y={pose_info['cy']:.3f}  "
          f"z_top({Z_TOP_PCT:.0f}pct)={pose_info['z_top']:.3f}")
    print(f"PCA major axis: alpha={pose_info['pca_alpha_deg']:.1f}°  "
          f"alpha_perp={pose_info['pca_alpha_perp_deg']:.1f}°  "
          f"alpha_final={pose_info['pca_alpha_final_deg']:.1f}°  "
          f"total_yaw={pose_info['total_yaw_deg']:.1f}°  "
          f"top_cluster_pts={pose_info['top_cluster_pts']}")

    xyz       = T_world_ee[:3, 3].tolist()
    quat_xyzw = Rot.from_matrix(T_world_ee[:3, :3]).as_quat().tolist()
    print(f"EE pose (world): xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})  "
          f"quat={[round(v, 3) for v in quat_xyzw]}")

    # --- Compose joint angles in HW order + encoder units ---
    joint_angles_hw  = [joint_angles_dro[i] for i in _DRO_TO_HW]
    joint_angles_enc = [int(round(v * RAD_TO_ENC)) for v in joint_angles_hw]
    print(f"Hand enc (HW): {joint_angles_enc}")

    # --- Hand init / release pose in encoder units (configs/hand.yaml 기준) ---
    hand_init_enc    = list(HAND_INIT_ENC)
    hand_release_enc = list(HAND_RELEASE_ENC)
    print(f"Hand init enc   : {hand_init_enc}")
    print(f"Hand release enc: {hand_release_enc}")

    # --- Build summary JSON ---
    grasp_entry = {
        "index":            0,
        "score":            1.0,
        "base_pose":        pose_payload(T_world_ee),   # world 프레임 기준 EE 자세
        "joint_angles":     [round(v, 6) for v in joint_angles_dro],  # DRO order
        "joint_angles_rad": [round(v, 6) for v in joint_angles_hw],   # HW order
        "joint_angles_enc": joint_angles_enc,
    }

    summary = {
        "robot_name":       "kistar",
        "grasp_type":       "topdown",
        "num_grasps":       1,
        "z_offset":         args.z_offset,
        "approach_offset":  APPROACH_OFFSET_M,
        "hand_init_enc":    hand_init_enc,
        "hand_release_enc": hand_release_enc,
        "calibration_path": str(Path(calib_path).resolve()),
        "T_base_camera":    T_base_camera.tolist(),
        "T_world_base":     T_world_base.tolist(),
        "coordinate_frame": "world",
        "grasps":           [grasp_entry],
    }

    summary_path = output_dir / f"{stem}_topdown_summary.json"
    save_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")

    # --- Build overlay canvas ---
    canvas = build_overlay_canvas(rgb_bgr, mask, xyz, K, T_world_camera, pose_info)

    # --- Preview: 창으로 띄워 확인 ---
    if args.preview:
        print("\n[PREVIEW] overlay 창을 확인하고 아무 키를 누르면 계속 진행합니다...")
        cv2.imshow("Grasp Preview (press any key to continue)", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- Save overlay image ---
    save_overlay(canvas, stem, output_dir)

    print(f"\n[OK] Top-down grasp complete.")


if __name__ == "__main__":
    main()
