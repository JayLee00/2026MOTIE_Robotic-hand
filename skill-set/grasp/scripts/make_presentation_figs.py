#!/usr/bin/env python3
"""
발표용 이미지 2장 생성.

  *_fig_pca.png      : 마스크 오버레이 + PCA 장축 화살표
  *_fig_position.png : 마스크 오버레이 + EE 파지 위치 마커

Usage:
    conda activate pipeline_all
    cd HARILAB/Grasp_fruit
    python scripts/make_presentation_figs.py \\
        data/outputs/interactive_010_000_topdown_summary.json
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ROOT    = Path(__file__).resolve().parents[1]
SRC     = ROOT / "src"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from affordance_grasp.geometry.frame_transform import invert_transform
from affordance_grasp.io.dataset_io import load_rgbd_bundle


# ── 재사용 헬퍼 ──────────────────────────────────────────────────────────────

def backproject_mask(depth, K, mask, depth_scale=1.0):
    K = np.asarray(K, dtype=np.float64)
    depth_m = depth.astype(np.float64) / depth_scale
    ys, xs  = np.where(mask & (depth_m > 0))
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    zs  = depth_m[ys, xs]
    return np.stack([
        (xs - K[0, 2]) * zs / K[0, 0],
        (ys - K[1, 2]) * zs / K[1, 1],
        zs,
    ], axis=1).astype(np.float32)


def sor(pts, k=20, std_ratio=2.0):
    if len(pts) <= k:
        return pts
    tree       = cKDTree(pts)
    dists, _   = tree.query(pts, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    thr        = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists <= thr]


def project_to_pixel(point_world, K, T_world_camera):
    T_cw    = invert_transform(T_world_camera)
    ph      = np.array([*point_world, 1.0], dtype=np.float64)
    pc      = T_cw @ ph
    if pc[2] <= 1e-6:
        return None
    u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
    v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
    return int(round(u)), int(round(v))


def base_canvas(rgb_bgr, mask):
    """반투명 녹색 마스크 오버레이 base 이미지."""
    canvas = rgb_bgr.copy()
    ov     = canvas.copy()
    ov[mask] = (30, 200, 60)
    return cv2.addWeighted(ov, 0.40, canvas, 0.60, 0.0)


# ── 이미지 1: PCA 장축 ────────────────────────────────────────────────────────

def draw_pca(canvas, K, T_world_camera, pose_info, arm_px=110):
    center_xy = np.array(pose_info["pca_center_xy"], dtype=np.float64)
    major_xy  = np.array(pose_info["pca_major_xy"],  dtype=np.float64)
    z         = float(pose_info["z_top"])

    # 중심 투영
    px0 = project_to_pixel([center_xy[0], center_xy[1], z], K, T_world_camera)
    if px0 is None:
        return

    # 5 cm step으로 픽셀 방향 추정 (half_len 전체를 투영하면 카메라 뒤로 날아갈 수 있음)
    DELTA = 0.05
    px_d = project_to_pixel([center_xy[0] + major_xy[0] * DELTA,
                              center_xy[1] + major_xy[1] * DELTA, z],
                             K, T_world_camera)
    if px_d is None:
        return

    dx = px_d[0] - px0[0];  dy = px_d[1] - px0[1]
    norm = math.sqrt(dx**2 + dy**2)
    if norm < 0.5:
        return
    dx /= norm;  dy /= norm

    px1 = (int(px0[0] + dx * arm_px), int(px0[1] + dy * arm_px))
    px2 = (int(px0[0] - dx * arm_px), int(px0[1] - dy * arm_px))

    # 원본 overlay와 동일한 스타일
    cv2.line(canvas, px2, px1, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.arrowedLine(canvas, px0, px1, (0, 200, 255), 3, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(canvas, px0, 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "PCA major", (px0[0] + 8, px0[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


# ── 이미지 2: EE 파지 위치 ────────────────────────────────────────────────────

def draw_position(canvas, K, T_world_camera, xyz_world):
    px = project_to_pixel(xyz_world, K, T_world_camera)
    if px is None:
        return

    h, w = canvas.shape[:2]
    RED   = (50,  50, 255)
    WHITE = (255, 255, 255)
    R_OUT = 40
    R_IN  = 10
    ARM   = 60    # crosshair 길이

    # 바깥 원 (목표 링)
    cv2.circle(canvas, px, R_OUT, RED,   3, cv2.LINE_AA)
    cv2.circle(canvas, px, R_OUT, WHITE, 1, cv2.LINE_AA)

    # crosshair
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        p_near = (px[0] + dx * R_IN,  px[1] + dy * R_IN)
        p_far  = (px[0] + dx * ARM,   px[1] + dy * ARM)
        cv2.line(canvas, p_near, p_far, RED,   3, cv2.LINE_AA)
        cv2.line(canvas, p_near, p_far, WHITE, 1, cv2.LINE_AA)

    # 중심 점
    cv2.circle(canvas, px, 5, WHITE, -1, cv2.LINE_AA)
    cv2.circle(canvas, px, 5, RED,    1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/make_presentation_figs.py <topdown_summary.json>")
        sys.exit(1)

    summary_path = Path(sys.argv[1]).resolve()
    if not summary_path.exists():
        print(f"[ERROR] not found: {summary_path}")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    stem      = summary_path.stem.replace("_topdown_summary", "")
    out_dir   = summary_path.parent
    raw_dir   = ROOT / "data" / "raw"
    inter_dir = ROOT / "data" / "interim"

    # ── 소스 파일 로드 ────────────────────────────────────────────────────────
    npz_path  = raw_dir   / f"{stem}.npz"
    mask_path = inter_dir / f"{stem}_mask.png"

    if not npz_path.exists():
        print(f"[ERROR] NPZ not found: {npz_path}")
        sys.exit(1)
    if not mask_path.exists():
        print(f"[ERROR] mask not found: {mask_path}")
        sys.exit(1)

    bundle  = load_rgbd_bundle(str(npz_path))
    rgb_bgr = bundle["rgb"]
    depth   = bundle["depth"]
    K       = np.array(bundle["K"], dtype=np.float64)

    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask     = mask_img > 127

    T_base_camera = np.array(summary["T_base_camera"], dtype=np.float64)
    T_world_base  = np.array(summary["T_world_base"],  dtype=np.float64)
    T_world_camera = T_world_base @ T_base_camera
    z_offset       = float(summary.get("z_offset", 0.13))

    # ── 포인트클라우드 처리 ───────────────────────────────────────────────────
    pts_cam   = backproject_mask(depth, K, mask)
    pts_cam   = sor(pts_cam)
    ones      = np.ones((len(pts_cam), 1), dtype=np.float64)
    pts_world = (T_world_camera @ np.hstack([pts_cam.astype(np.float64), ones]).T).T[:, :3]

    # ── Grasp pose 재계산 ─────────────────────────────────────────────────────
    from utils.arm import EE_YAW_DEG, EE_X_OFFSET_M, EE_Y_OFFSET_M, TOP_Z_PCT, Z_TOP_PCT

    z_vals  = pts_world[:, 2]
    z_top   = float(np.percentile(z_vals, Z_TOP_PCT))
    z_thr   = float(np.percentile(z_vals, 100.0 - TOP_Z_PCT))
    top_pts = pts_world[z_vals >= z_thr]
    cx      = float(top_pts[:, 0].mean())
    cy      = float(top_pts[:, 1].mean())

    xy    = pts_world[:, :2].astype(np.float64)
    xy_c  = xy - xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy_c, full_matrices=False)
    major    = Vt[0]
    alpha    = math.atan2(float(major[1]), float(major[0]))
    proj     = xy_c @ major
    half_len = float(np.percentile(np.abs(proj), 95.0))

    def _wrap_pi(a):
        while a > math.pi:  a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

    alpha_perp  = alpha + math.pi / 2
    c1 = _wrap_pi(alpha_perp)
    c2 = _wrap_pi(alpha_perp + math.pi)
    alpha_final = c1 if abs(c1) <= abs(c2) else c2

    cf = math.cos(alpha_final); sf = math.sin(alpha_final)
    ox = EE_X_OFFSET_M * cf - EE_Y_OFFSET_M * sf
    oy = EE_X_OFFSET_M * sf + EE_Y_OFFSET_M * cf

    xyz_world = [cx + ox, cy + oy, z_top + z_offset]

    pose_info = {
        "pca_center_xy":  xy.mean(axis=0).tolist(),
        "pca_major_xy":   major.tolist(),
        "pca_half_length_m": half_len,
        "z_top": z_top,
    }

    # ── 이미지 생성 ───────────────────────────────────────────────────────────
    out_pca = out_dir / f"{stem}_fig_pca.png"
    out_pos = out_dir / f"{stem}_fig_position.png"

    canvas_pca = base_canvas(rgb_bgr, mask)
    draw_pca(canvas_pca, K, T_world_camera, pose_info)
    cv2.imwrite(str(out_pca), canvas_pca)
    print(f"[OK] {out_pca}")

    canvas_pos = base_canvas(rgb_bgr, mask)

    # SAM3 bbox — 빨간 박스
    sam3_path = inter_dir / f"{stem}_sam3.json"
    if sam3_path.exists():
        with open(sam3_path) as f:
            sam3 = json.load(f)
        x1, y1, x2, y2 = [int(v) for v in sam3["sam3"]["box_xyxy"]]
        cv2.rectangle(canvas_pos, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    # 좌상단 x, y, z 텍스트
    lines = [
        f"x={xyz_world[0]:.3f}  y={xyz_world[1]:.3f}",
        f"z={xyz_world[2]:.3f}",
    ]
    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.70
    THICK = 2
    for i, line in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(canvas_pos, line, (10, y), FONT, SCALE, (0,   0,   0), THICK + 2, cv2.LINE_AA)
        cv2.putText(canvas_pos, line, (10, y), FONT, SCALE, (80, 255,  80), THICK,     cv2.LINE_AA)

    cv2.imwrite(str(out_pos), canvas_pos)
    print(f"[OK] {out_pos}")


if __name__ == "__main__":
    main()
