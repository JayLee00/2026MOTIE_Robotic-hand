#!/usr/bin/env python3
"""통합 실시간 시각화 — 과일 6DoF + Paxini 촉각 3D + FT 스트립차트.

두 프로그램을 합친 것:
  1) 과일 포즈  : record/fruit_overlay.py 의 Overlay 클래스를 그대로 임포트해서 사용
                 (RGB 위에 3D OBB 와이어프레임 + XYZ 축 + 수치). ROS 비의존이라 재사용 가능.
  2) 센서 표현  : kist-vtdp-wrapper/tools/viz_demo.py 의 Scene3D(Open3D) 를 이식
                 (지문 CAD 4개에 127 탁셀을 힘 크기로 색칠 + 상위 2개 힘 화살표).
                 → fruit_overlay 자체 FT 패널은 끄고(show_ft=False) 이걸로 대체한다.

원본 viz_demo 는 HDF5→MP4 오프라인 렌더러였다. 실측 병목은 Open3D 가 아니라
matplotlib 전체 재그리기(31~51ms)였으므로, 합성을 cv2 로 바꿔 실시간화했다.
    Open3D render 5.6ms + cv2 합성 ≈ 8~12ms  → 30Hz 여유 있음

레이아웃
    ┌────────────────────────┬──────────────────┐
    │ 카메라 + 과일 6DoF     │ 촉각 3D (4파트)  │
    ├────────────────────────┴──────────────────┤
    │ FT 스트립차트 Fz/Tx/Ty × 4 (링버퍼)       │
    └───────────────────────────────────────────┘

실행:
    source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
    export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
    python3 Visualization/live_viz.py
    python3 Visualization/live_viz.py --selftest    # ROS 없이 렌더 검증 (가짜 데이터)
    python3 Visualization/live_viz.py --no-tactile  # 과일만
키: q/ESC 종료, s 스냅샷
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "record"))          # fruit_overlay 재사용

from tactile_render import (BG, KIN_FINGER_LABEL, N_PART,  # noqa: E402
                            N_TAXEL, PART_COLORS, VMAX_DEFAULT, Scene3D)

OUT_DIR = HERE / "snapshots"
TRAIL_SEC = 6.0                                   # 스트립차트 표시 구간

# 하단 FT 패널: 손가락 4칸 × (Tx, Ty, Fz×1000)
# kin 원본 축 순서는 (Fz, Tx, Ty) 이므로 표시 순서에 맞춰 채널을 재배열한다.
FT_CHAN = ((1, "Tx", 1.0),
           (2, "Ty", 1.0),
           (0, "Fz", 100.0))                      # Fz 만 스케일이 달라 x100 해야 같이 보인다
                                                  #   (원본 viz_demo 는 x1000 을 썼다. 값이
                                                  #    안 보이거나 너무 크면 이 숫자만 조절)
FT_CHAN_COLORS = ("#5ee0c8", "#f4a24c", "#c86bd8")   # Tx, Ty, Fz
FINGER_NAME = ("Thumb", "Index", "Middle", "Ring")
TOP_FRAC = 0.74                                   # 4칸으로 늘리며 세로도 줄여 비율 유지


def _bgr(h: str):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def draw_strip(canvas, buf_t, buf_v, x, y, w, h, title, vmin, vmax, colors):
    """cv2 폴리라인 스트립차트 — matplotlib 대신 (재그리기 비용 제거)."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), _bgr("#1a1e27"), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), _bgr("#2a2f3a"), 1)
    cv2.putText(canvas, title, (x + 6, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, _bgr("#e8eaf0"), 1, cv2.LINE_AA)
    if len(buf_t) < 2:
        return
    t = np.asarray(buf_t)
    t0, t1 = t[0], max(t[-1], t[0] + 1e-3)
    span = max(vmax - vmin, 1e-6)
    zero_y = int(y + h - (0.0 - vmin) / span * h)
    if y < zero_y < y + h:
        cv2.line(canvas, (x, zero_y), (x + w, zero_y), _bgr("#2a2f3a"), 1)
    for p in range(buf_v.shape[1] if buf_v.ndim > 1 else 1):
        v = buf_v[:, p]
        xs = (x + (t - t0) / (t1 - t0) * w).astype(np.int32)
        ys = (y + h - np.clip((v - vmin) / span, 0, 1) * h).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, colors[p % len(colors)], 1, cv2.LINE_AA)


class Viz:
    """ROS 무관 렌더러 — 최신 값만 들고 있다가 compose() 로 한 장 만든다."""

    def __init__(self, use_tactile=True, vmax=VMAX_DEFAULT, w=1600, h=900):
        self.W, self.H = w, h
        self.use_tactile = use_tactile
        self.vmax = vmax
        self.rgb = None                            # BGR 카메라 프레임
        self.tac = np.zeros((N_PART, N_TAXEL, 3))  # paxini raw
        self.kin_t, self.kin_v = deque(), deque()  # FT 링버퍼
        self.scene = Scene3D() if use_tactile else None
        self.t_rgb = self.t_tac = self.t_kin = 0.0

        from fruit_overlay import Overlay          # 과일 포즈 렌더러 재사용
        self.ov = Overlay(show_ft=False)           # 자체 FT 패널은 끔 → Scene3D 로 대체

    def push_kin(self, v):
        now = time.time()
        self.kin_t.append(now)
        self.kin_v.append(np.asarray(v, dtype=np.float64).reshape(N_PART, 3))
        while self.kin_t and now - self.kin_t[0] > TRAIL_SEC:
            self.kin_t.popleft()
            self.kin_v.popleft()
        self.t_kin = now

    def compose(self) -> np.ndarray:
        cv = np.full((self.H, self.W, 3), _bgr(BG), np.uint8)
        top_h = int(self.H * TOP_FRAC)
        left_w = int(self.W * 0.56)

        # ── 좌상: 카메라 + 과일 6DoF (fruit_overlay.Overlay 그대로) ──
        if self.rgb is not None:
            img = self.ov.draw(self.rgb.copy())
            s = min(left_w / img.shape[1], top_h / img.shape[0])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
            cv[:img.shape[0], :img.shape[1]] = img
        else:
            cv2.putText(cv, "waiting camera...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, _bgr("#f4a24c"), 2, cv2.LINE_AA)

        # ── 우상: 촉각 3D ──
        if self.use_tactile and self.scene is not None:
            tac = self.scene.render(self.tac, self.vmax)          # RGB
            tac = cv2.cvtColor(tac, cv2.COLOR_RGB2BGR)
            tw = self.W - left_w
            s = min(tw / tac.shape[1], top_h / tac.shape[0])
            tac = cv2.resize(tac, (int(tac.shape[1] * s), int(tac.shape[0] * s)))
            cv[:tac.shape[0], left_w:left_w + tac.shape[1]] = tac
            stale = time.time() - self.t_tac > 0.5
            cv2.putText(cv, f"Paxini tactile  vmax={self.vmax:.2f}"
                            + ("  [STALE]" if stale else ""),
                        (left_w + 8, top_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, _bgr("#f4a24c" if stale else "#e8eaf0"), 1, cv2.LINE_AA)

        # ── 하단: FT 스트립차트 (Fz, Tx, Ty) × 4 파트 ──
        if self.kin_v:
            V = np.stack(self.kin_v)                               # (T, 4, 3) 원본 (Fz,Tx,Ty)
            colors = [_bgr(c) for c in FT_CHAN_COLORS]
            pad, y = 10, top_h + 8
            sw = (self.W - pad * (N_PART + 1)) // N_PART           # 손가락 4칸
            sh = self.H - y - 10
            for p in range(N_PART):
                # 이 손가락의 (Tx, Ty, Fz*1000) 3줄
                v = np.stack([V[:, p, ch] * sc for ch, _, sc in FT_CHAN], axis=1)
                m = max(float(np.abs(v).max()), 1e-3) * 1.15       # 칸마다 자동 스케일
                label = "  ".join(f"{nm}{'x%.0f' % sc if sc != 1.0 else ''}"
                                  for _, nm, sc in FT_CHAN)
                draw_strip(cv, self.kin_t, v, pad + p * (sw + pad), y, sw, sh,
                           f"{FINGER_NAME[p]} ({KIN_FINGER_LABEL[p]})   {label}",
                           -m, m, colors)
        return cv


def selftest():
    """ROS 없이 렌더 경로만 검증."""
    v = Viz(use_tactile=True)
    v.rgb = np.full((480, 640, 3), 40, np.uint8)
    cv2.putText(v.rgb, "SELFTEST", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (90, 90, 90), 2)
    v.ov.K = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1.0]])
    ang = np.radians(30)
    v.ov.set_pose((0.02, -0.01, 0.30), (0.0, np.sin(ang / 2), 0.0, np.cos(ang / 2)),
                  "camera_color_optical_frame", time.time())
    v.ov.size = [0.070, 0.055, 0.055]
    rng = np.random.default_rng(0)
    v.tac = np.abs(rng.normal(0, 0.25, (N_PART, N_TAXEL, 3)))
    v.tac[1, 40:60, 2] += 1.2                                      # 접촉 흉내
    v.t_tac = time.time()
    for i in range(120):                                           # 스트립차트 채우기
        t = i / 30.0
        # 원본 축 순서 (Fz, Tx, Ty). 실제 Fz 는 아주 작아서(x1000 하는 이유) 그렇게 흉내낸다.
        v.push_kin(np.stack([[0.0015 * np.sin(t * 1.3 + p),        # Fz  (x1000 하면 ~1.5)
                              0.6 * np.cos(t + p * 0.7),           # Tx
                              0.4 * np.sin(t * 2 + p)]             # Ty
                             for p in range(N_PART)]))
    out = v.compose()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "selftest_live_viz.png"
    cv2.imwrite(str(p), out)
    print(f"selftest OK — {out.shape[1]}x{out.shape[0]} 저장: {p}")


def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32MultiArray

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    rclpy.init()
    node = Node("live_viz")
    v = Viz(use_tactile=not args.no_tactile, vmax=args.vmax)

    def on_comp(m):
        img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            v.rgb, v.t_rgb = img, time.time()

    def on_raw(m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        v.rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else img
        v.t_rgb = time.time()

    if "compressed" in args.color_topic:
        node.create_subscription(CompressedImage, args.color_topic, on_comp, qos)
    else:
        node.create_subscription(Image, args.color_topic, on_raw, qos)
    node.create_subscription(
        CameraInfo, args.info_topic,
        lambda m: setattr(v.ov, "K", np.asarray(m.k, np.float64).reshape(3, 3)), qos)
    node.create_subscription(
        PoseStamped, args.pose_topic,
        lambda m: v.ov.set_pose(
            (m.pose.position.x, m.pose.position.y, m.pose.position.z),
            (m.pose.orientation.x, m.pose.orientation.y,
             m.pose.orientation.z, m.pose.orientation.w),
            m.header.frame_id, time.time()), qos)
    node.create_subscription(Float32MultiArray, args.size_topic,
                             lambda m: setattr(v.ov, "size", list(m.data)), qos)

    def on_tac(m):
        d = np.asarray(m.data, np.float64)
        if d.size >= N_PART * N_TAXEL * 3:
            v.tac = d[:N_PART * N_TAXEL * 3].reshape(N_PART, N_TAXEL, 3)
            v.t_tac = time.time()

    if not args.no_tactile:
        node.create_subscription(Float32MultiArray, args.tactile_topic, on_tac, qos)
    node.create_subscription(Float32MultiArray, args.ft_topic,
                             lambda m: v.push_kin(list(m.data)[:12]), qos)

    print(f"구독:\n  color   {args.color_topic}\n  info    {args.info_topic}\n"
          f"  pose    {args.pose_topic}\n  size    {args.size_topic}\n"
          f"  tactile {args.tactile_topic}\n  ft      {args.ft_topic}\n"
          "키: q=종료  s=스냅샷")
    win = "KIST live viz  (q=quit, s=snapshot)"
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            out = v.compose()
            cv2.imshow(win, out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                p = OUT_DIR / f"viz_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(p), out)
                print(f"스냅샷 저장: {p}")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description="과일 6DoF + Paxini 촉각 통합 실시간 시각화")
    ap.add_argument("--selftest", action="store_true", help="ROS 없이 렌더 검증")
    ap.add_argument("--no-tactile", action="store_true", help="촉각 3D 끄기(과일만)")
    ap.add_argument("--vmax", type=float, default=VMAX_DEFAULT,
                    help=f"촉각 색상 정규화 상한 [0.1N] (기본 {VMAX_DEFAULT}; 실측 p99.9=1.33)")
    ap.add_argument("--color-topic", default="/front_cam/front/color/image_raw/compressed")
    ap.add_argument("--info-topic", default="/front_cam/front/color/camera_info")
    ap.add_argument("--pose-topic", default="/fruit/pose")
    ap.add_argument("--size-topic", default="/fruit/size")
    ap.add_argument("--tactile-topic", default="/paxini/right/raw",
                    help="로봇 핸드 촉각 1524 (글러브면 /glove/paxini/right/raw)")
    ap.add_argument("--ft-topic", default="/hand/right/kin", help="FT 12 (Fz,Tx,Ty)x4")
    args = ap.parse_args()
    selftest() if args.selftest else run_ros(args)


if __name__ == "__main__":
    main()
