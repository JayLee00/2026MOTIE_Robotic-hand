#!/usr/bin/env python3
"""RealSense 2D 영상 + 과일 6DoF 실시간 오버레이 (이 PC에서 LAN으로 보기, GPU 불필요).

제어 PC의 컬러 영상과 과일 pose를 구독해, 영상 위에 6DoF를 직접 그린다:
    - 3D 박스 와이어프레임 (pose + size 로 만든 OBB 를 K 로 투영)
    - 좌표축 3개 (X=빨강, Y=초록, Z=파랑) → 오리엔테이션을 눈으로 확인
    - 중심점 + 텍스트 (위치[m], RPY[deg], 크기 a/b/c[m], 수신율)

화면 하단에는 손가락 F/T 를 좌우 두 패널로 동시에 띄운다
(각각 폭은 영상의 1/4, 높이는 1/5 — 가로로 넓은 띠. 좌우 합쳐 너비의 절반):
    좌측 하단  paxini ft  /paxini/<side>/ft   12ch = 손가락당 [Fz, Fx, Fy]
    우측 하단  hand ft    /hand/<side>/kin    12ch = 손가락당 [Ty, Tx, Fz]
    손가락 순서는 둘 다 엄지(T) → 검지(I) → 중지(M) → 약지(R).

    읽는 법 (두 패널 같은 문법):
      원의 색  = 법선력 Fz.  빨강=+, 파랑=-, 회색=0.  (크기는 색 진하기)
      화살표   = 면내 2채널. paxini 는 전단력 (Fx,Fy), hand 는 모멘트 (Tx,Ty).
      원 아래  = 손가락 기호 + Fz 수치.  패널 맨 아래 = 현재 자동 스케일.
    단위가 문서화돼 있지 않아 스케일은 자동(최근 최대값 추적)이다.
    고정하려면 --ft-scale / --hand-ft-scale. 끄려면 --no-ft.

구독:
    <color>/image_raw (또는 /compressed)      영상          [자동 탐지]
    <color>/camera_info                       K (투영에 필수) [자동 탐지]
    /fruit/pose   geometry_msgs/PoseStamped    위치+방향(카메라 광학 프레임)
    /fruit/size   std_msgs/Float32MultiArray   [a, b, c] 전체 축 길이 [m]
    /paxini/<side>/ft, /hand/<side>/kin        손가락 F/T (위 참조)

※ pose/size 는 fruit_pose_bridge.py 가 /inhand/bbox_corners 로부터 만든다.
  (live_bbox_gui.py 자체도 자기 화면에 박스를 그리지만, 그건 GPU 인식 PC 화면.
   이 뷰어는 인식 결과만 받아서 이 PC에서 가볍게 본다.)

키: q/ESC=종료, s=스냅샷 PNG 저장

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 record/fruit_overlay.py
  python3 record/fruit_overlay.py --raw                 # compressed 없을 때
  python3 record/fruit_overlay.py --selftest            # ROS 없이 투영 렌더 확인
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "snapshots"
STALE_SEC = 0.5

# OBB 8코너 부호조합 및 12엣지(한 비트만 다른 코너쌍)
SIGNS = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=np.float64)  # (8,3)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.sum(np.abs(SIGNS[i] - SIGNS[j]) > 0) == 1]


def quat_to_R(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def R_to_rpy_deg(R: np.ndarray):
    """ZYX(roll-pitch-yaw) [deg]."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 0.9999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    d = 180.0 / math.pi
    return roll * d, pitch * d, yaw * d


def project(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    """카메라 광학 프레임 3D점(N,3) -> 픽셀(N,2). (x right, y down, z forward)"""
    Z = np.clip(P[:, 2], 1e-6, None)
    u = K[0, 0] * P[:, 0] / Z + K[0, 2]
    v = K[1, 1] * P[:, 1] / Z + K[1, 2]
    return np.stack([u, v], axis=1)


# ─────────────────────── 손가락 F/T 패널 ───────────────────────
FINGER_TAGS = ("T", "I", "M", "R")            # 엄지 index 중지 약지
FINGER_NAMES = ("thumb", "index", "middle", "ring")


def _compact(v: float) -> str:
    """좁은 셀에 들어가게 짧게: 999 넘으면 k 단위."""
    a = abs(v)
    if a >= 9950:
        return f"{v/1000:+.0f}k"
    if a >= 995:
        return f"{v/1000:+.1f}k"
    return f"{v:+.0f}" if a >= 10 else f"{v:+.1f}"


def _put_center(img, text, cx, y, fs, color):
    """텍스트를 cx 기준 가로 중앙에 놓는다 (셀이 좁아 왼쪽 정렬은 겹친다)."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.putText(img, text, (int(cx - tw / 2), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                fs, color, 1, cv2.LINE_AA)


def heat_bgr(v: float, vmax: float):
    """발산 색상표: 음수=파랑, 0=회색, 양수=빨강 (BGR)."""
    t = float(np.clip(v / max(vmax, 1e-9), -1.0, 1.0))
    a = abs(t)
    base = np.array([70.0, 70.0, 70.0])
    tgt = np.array([235.0, 130.0, 45.0]) if t < 0 else np.array([45.0, 60.0, 235.0])
    return tuple(int(x) for x in base * (1.0 - a) + tgt * a)


class FTPanel:
    """4손가락 x 3채널 F/T 를 한 패널로 그린다.

    3채널 중 2개는 **화살표**(면내 벡터), 남은 1개(법선력 Fz)는 **원의 색**으로 보여
    한 눈에 "어느 손가락이 얼마나 세게 눌리고 어느 방향으로 밀리나"를 읽게 한다.

    단위가 문서화되어 있지 않아 스케일은 자동이다: 최근 최대값을 천천히 감쇠시키며
    추적하고 현재 스케일을 제목에 적는다(--ft-scale / --hand-ft-scale 로 고정 가능).
    """

    def __init__(self, title: str, ch_names, vec_idx, col_idx,
                 fixed_scale: float = 0.0, tau: float = 3.0):
        self.title = title
        self.ch = tuple(ch_names)          # 채널 3개 이름 (데이터 순서 그대로)
        self.vx, self.vy = vec_idx         # 화살표에 쓸 채널 인덱스 (x, y)
        self.ci = col_idx                  # 색에 쓸 채널 인덱스
        self.fixed = float(fixed_scale)
        self.tau = float(tau)              # 스케일 감쇠 시간상수 [s]
        self.data = None                   # (4,3)
        self.last_rx = 0.0
        self.rate = 0.0
        self._prev_t = None
        self._vmax = 1e-6
        self._cmax = 1e-6

    def set(self, values, t: float):
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size < 12:
            v = np.pad(v, (0, 12 - v.size))
        self.data = v[:12].reshape(4, 3)
        self.last_rx = t
        if self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 1e-6:
                inst = 1.0 / dt
                self.rate = inst if self.rate == 0 else 0.9 * self.rate + 0.1 * inst
        # 자동 스케일: 순간 최대는 즉시 반영, 줄어들 때는 시간상수 tau 로 감쇠.
        # 감쇠를 '샘플당'으로 걸면 토픽 주기에 따라 속도가 달라진다(paxini 90Hz vs
        # hand kin 200Hz). 시간 기반이라야 두 패널이 같은 속도로 회복한다.
        dt = max(0.0, t - self._prev_t) if self._prev_t is not None else 0.0
        self._prev_t = t
        k = math.exp(-dt / self.tau) if dt > 0 else 1.0
        vmag = float(np.max(np.hypot(self.data[:, self.vx], self.data[:, self.vy])))
        cmag = float(np.max(np.abs(self.data[:, self.ci])))
        self._vmax = max(vmag, self._vmax * k)
        self._cmax = max(cmag, self._cmax * k)

    def scales(self):
        if self.fixed > 0:
            return self.fixed, self.fixed
        return max(self._vmax, 1e-6), max(self._cmax, 1e-6)

    def draw_into(self, out: np.ndarray, x0: int, y0: int, pw: int, ph: int):
        h, w = out.shape[:2]
        x0 = max(0, min(x0, w - 1)); y0 = max(0, min(y0, h - 1))
        pw = min(pw, w - x0); ph = min(ph, h - y0)
        if pw < 40 or ph < 30:
            return

        roi = out[y0:y0 + ph, x0:x0 + pw]
        cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)
        cv2.rectangle(out, (x0, y0), (x0 + pw - 1, y0 + ph - 1), (90, 90, 90), 1)

        fresh = self.data is not None and (time.time() - self.last_rx) < STALE_SEC
        allzero = self.data is not None and not np.any(self.data)
        vmax, cmax = self.scales()

        # 레이아웃은 높이 기준으로 결정론적으로 잡는다. 행 4개(제목·손가락기호·수치·범례)가
        # 항상 들어가고 남은 높이를 원에 준다 → 폭만 늘려도 아무것도 잘리지 않는다.
        th = max(9, int(ph * 0.145))
        fs = max(0.26, min(0.44, th / 40.0))
        cw = pw // 4
        circle_h = ph - 4 * th
        show_num = circle_h >= 14
        if not show_num:                       # 아주 낮은 패널: 수치 행 포기
            circle_h = ph - 3 * th
        r = max(5, min(cw // 2 - 3, circle_h // 2))
        cy = y0 + th + r + 1
        tag_y = y0 + ph - (2 * th if show_num else th) - 3
        num_y = y0 + ph - th - 3
        body_y = y0 + th

        cv2.putText(out, self.title, (x0 + 3, y0 + th - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255) if fresh else (150, 150, 160), 1, cv2.LINE_AA)
        for i in range(4):
            cx = x0 + cw * i + cw // 2
            if i:                                   # 셀 구분선
                cv2.line(out, (x0 + cw * i, body_y), (x0 + cw * i, y0 + ph - th),
                         (70, 70, 70), 1)
            if self.data is None:
                cv2.circle(out, (cx, cy), r, (60, 60, 60), 1, cv2.LINE_AA)
                continue
            fz = float(self.data[i, self.ci])
            vx = float(self.data[i, self.vx])
            vy = float(self.data[i, self.vy])

            fill = heat_bgr(fz, cmax) if fresh else (60, 60, 60)
            cv2.circle(out, (cx, cy), r, fill, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), r, (210, 210, 210) if fresh else (90, 90, 90),
                       1, cv2.LINE_AA)

            # 면내 벡터 화살표 (이미지 y 는 아래로 증가 → 부호 반전).
            # 채워진 원 위에서도 보이게 검은 테두리를 먼저 굵게 깔고 흰 선을 덮는다.
            mag = math.hypot(vx, vy)
            if fresh and mag > 1e-9:
                s = min(1.0, mag / vmax) * (r * 0.95)   # 원 안에 머물게 → 겹침 없음
                ex = int(round(cx + vx / mag * s))
                ey = int(round(cy - vy / mag * s))
                cv2.arrowedLine(out, (cx, cy), (ex, ey), (0, 0, 0), 3,
                                cv2.LINE_AA, tipLength=0.35)
                cv2.arrowedLine(out, (cx, cy), (ex, ey), (255, 255, 255), 1,
                                cv2.LINE_AA, tipLength=0.35)
            cv2.circle(out, (cx, cy), 1, (20, 20, 20), -1)

            _put_center(out, FINGER_TAGS[i], cx, tag_y, fs,
                        (255, 255, 255) if fresh else (140, 140, 140))
            if show_num:
                _put_center(out, _compact(fz), cx, num_y, fs * 0.95,
                            (235, 235, 235) if fresh else (130, 130, 130))

        # 상태 / 범례 (패널이 좁아도 잘리지 않게 짧게)
        if self.data is None:
            msg, col = "no data", (150, 150, 255)
        elif not fresh:
            msg, col = "STALE", (150, 150, 255)
        elif allzero:
            msg, col = "ALL ZERO - sensor?", (90, 200, 255)
        else:
            msg = (f"{self.ch[self.vx]}{self.ch[self.vy]}<{_compact(vmax)} "
                   f"{self.ch[self.ci]}<{_compact(cmax)}")
            col = (200, 200, 200)
        cv2.putText(out, msg, (x0 + 3, y0 + ph - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.8, col, 1, cv2.LINE_AA)


class Overlay:
    """영상 위에 6DoF 를 그리는 순수 렌더러 (ROS 무관 → selftest 가능)."""

    def __init__(self, panel_w_scale: float = 0.25, panel_h_scale: float = 0.20,
                 ft_scale: float = 0.0, hand_ft_scale: float = 0.0,
                 ft_tau: float = 3.0, show_ft: bool = True):
        self.K = None
        self.pos = None        # (3,)
        self.R = None          # (3,3)
        self.size = None       # [a,b,c] 전체 길이
        self.frame_id = "-"
        self.last_rx = 0.0
        self.rate = 0.0
        self._prev_t = None

        # 좌우로 넓게: 폭은 영상의 1/4, 높이는 1/5 → 가로로 늘어난 띠 모양.
        # 두 패널이 각각 1/4 이라 좌우 합쳐 화면 너비의 절반을 쓴다.
        self.panel_w_scale = panel_w_scale
        self.panel_h_scale = panel_h_scale
        self.show_ft = show_ft
        # paxini: 손가락당 [Fz, Fx, Fy] → 화살표 = (Fx, Fy) 전단력, 색 = Fz 법선력
        self.paxini = FTPanel("paxini ft  [Fz,Fx,Fy]", ("Fz", "Fx", "Fy"),
                              vec_idx=(1, 2), col_idx=0, fixed_scale=ft_scale, tau=ft_tau)
        # hand kin(=hand_ft): 손가락당 [Ty, Tx, Fz] → 화살표 = (Tx, Ty) 모멘트, 색 = Fz
        self.hand_ft = FTPanel("hand ft  [Ty,Tx,Fz]", ("Ty", "Tx", "Fz"),
                               vec_idx=(1, 0), col_idx=2, fixed_scale=hand_ft_scale, tau=ft_tau)

    def set_pose(self, p, q, frame_id, t):
        self.pos = np.asarray(p, dtype=np.float64)
        self.R = quat_to_R(*q)
        self.frame_id = frame_id or "-"
        self.last_rx = t
        if self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 1e-6:
                inst = 1.0 / dt
                self.rate = inst if self.rate == 0 else 0.9 * self.rate + 0.1 * inst
        self._prev_t = t

    def draw(self, img: np.ndarray) -> np.ndarray:
        out = img
        h, w = out.shape[:2]
        fresh = self.pos is not None and (time.time() - self.last_rx) < STALE_SEC
        has_geom = self.K is not None and self.pos is not None and self.R is not None

        if has_geom:
            half = (np.asarray(self.size[:3], dtype=np.float64) / 2.0
                    if self.size and len(self.size) >= 3 else np.full(3, 0.035))
            corners = self.pos[None, :] + (SIGNS * half[None, :]) @ self.R.T   # (8,3)
            axis_len = float(max(half)) * 1.6
            pts3 = np.vstack([corners, self.pos[None, :],
                              self.pos + self.R[:, 0] * axis_len,
                              self.pos + self.R[:, 1] * axis_len,
                              self.pos + self.R[:, 2] * axis_len])
            if np.all(pts3[:, 2] > 0.02):                 # 전부 카메라 앞
                px = project(pts3, self.K).astype(int)
                box, ctr, ax = px[:8], px[8], px[9:12]
                box_col = (0, 200, 255) if fresh else (110, 110, 110)
                for i, j in EDGES:
                    cv2.line(out, tuple(box[i]), tuple(box[j]), box_col, 1, cv2.LINE_AA)
                # 좌표축: X=빨강 Y=초록 Z=파랑 (BGR)
                for k, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    cv2.arrowedLine(out, tuple(ctr), tuple(ax[k]),
                                    col if fresh else (110, 110, 110), 2,
                                    cv2.LINE_AA, tipLength=0.25)
                cv2.circle(out, tuple(ctr), 4, (0, 0, 255) if fresh else (110, 110, 110), -1)

        # 텍스트 패널
        lines = []
        if self.K is None:
            lines.append("waiting camera_info (K required)")
        if fresh:
            x, y, z = self.pos
            r, p, yw = R_to_rpy_deg(self.R)
            sz = "/".join(f"{v:.3f}" for v in self.size[:3]) if self.size else "-"
            lines += [
                f"pos [m]  x={x:+.3f} y={y:+.3f} z={z:+.3f}",
                f"RPY[deg] r={r:+6.1f} p={p:+6.1f} y={yw:+6.1f}",
                f"size a/b/c [m] {sz}",
                f"{self.rate:4.1f} Hz   frame={self.frame_id}",
            ]
        else:
            lines.append("NO FRUIT POSE" if self.pos is None else "POSE STALE")
            lines.append("check live_bbox_gui / fruit_pose_bridge")

        pad, lh = 8, 20
        box_h = lh * len(lines) + pad
        panel = out[0:box_h, 0:min(w, 430)]
        cv2.rectangle(panel, (0, 0), (panel.shape[1], panel.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.45, out[0:box_h, 0:panel.shape[1]], 0.55, 0,
                        out[0:box_h, 0:panel.shape[1]])
        for i, ln in enumerate(lines):
            cv2.putText(out, ln, (pad, pad + lh * i + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255) if fresh else (150, 150, 255), 1, cv2.LINE_AA)

        # 손가락 F/T 패널: 좌측 하단 = paxini, 우측 하단 = hand ft (영상의 panel_scale 배)
        if self.show_ft:
            pw = max(120, int(w * self.panel_w_scale))
            ph = max(64, int(h * self.panel_h_scale))
            self.paxini.draw_into(out, 0, h - ph, pw, ph)
            self.hand_ft.draw_into(out, w - pw, h - ph, pw, ph)
        return out


# ─────────────────────────── ROS 실행 경로 ───────────────────────────
def discover(node, want_raw: bool):
    """컬러 이미지/camera_info 토픽 자동 탐지."""
    names = dict(node.get_topic_names_and_types())
    color = info = None
    for name, types in sorted(names.items()):
        low = name.lower()
        if "color" not in low or "depth" in low:
            continue
        if info is None and "sensor_msgs/msg/CameraInfo" in types:
            info = name
        if color is None:
            if want_raw and "sensor_msgs/msg/Image" in types and low.endswith("image_raw"):
                color = name
            elif (not want_raw) and "sensor_msgs/msg/CompressedImage" in types \
                    and low.endswith("compressed"):
                color = name
    if color is None:   # fallback: 아무 컬러 Image
        for name, types in sorted(names.items()):
            low = name.lower()
            if "color" in low and "depth" not in low and "sensor_msgs/msg/Image" in types \
                    and low.endswith("image_raw"):
                color = name
                break
    return color, info


def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, CompressedImage, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32MultiArray

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    rclpy.init()
    node = Node("fruit_overlay")
    ov = Overlay(panel_w_scale=args.panel_w_scale, panel_h_scale=args.panel_h_scale,
                 ft_scale=args.ft_scale, hand_ft_scale=args.hand_ft_scale,
                 ft_tau=args.ft_tau, show_ft=not args.no_ft)
    latest = {"img": None, "new": False}

    t0 = time.perf_counter()          # 디스커버리 대기
    while time.perf_counter() - t0 < 1.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    color = args.color_topic or discover(node, args.raw)[0]
    info = args.info_topic or discover(node, args.raw)[1]
    if not color:
        print("✗ 컬러 이미지 토픽을 못 찾음 — 제어 PC realsense 켜졌는지 확인\n"
              "  (또는 --color-topic /front_cam/front/color/image_raw --raw)")
        node.destroy_node(); rclpy.shutdown(); return
    print(f"구독: color={color}\n      info ={info or '없음(K 없으면 박스 안 그려짐)'}\n"
          f"      pose =/fruit/pose  size=/fruit/size\n키: q=종료  s=스냅샷")

    def on_raw(m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        latest["img"] = img.copy()
        latest["new"] = True

    def on_comp(m):
        img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            latest["img"] = img
            latest["new"] = True

    if "compressed" in color.lower():
        node.create_subscription(CompressedImage, color, on_comp, qos)
    else:
        node.create_subscription(Image, color, on_raw, qos)
    if info:
        node.create_subscription(CameraInfo, info,
                                 lambda m: setattr(ov, "K", np.asarray(m.k, dtype=np.float64).reshape(3, 3)),
                                 qos)
    node.create_subscription(PoseStamped, "/fruit/pose",
                             lambda m: ov.set_pose(
                                 (m.pose.position.x, m.pose.position.y, m.pose.position.z),
                                 (m.pose.orientation.x, m.pose.orientation.y,
                                  m.pose.orientation.z, m.pose.orientation.w),
                                 m.header.frame_id, time.time()), qos)
    node.create_subscription(Float32MultiArray, "/fruit/size",
                             lambda m: setattr(ov, "size", list(m.data)), qos)
    if not args.no_ft:
        pax = f"/paxini/{args.side}/ft"
        kin = f"/hand/{args.side}/kin"
        node.create_subscription(Float32MultiArray, pax,
                                 lambda m: ov.paxini.set(m.data, time.time()), qos)
        node.create_subscription(Float32MultiArray, kin,
                                 lambda m: ov.hand_ft.set(m.data, time.time()), qos)
        print(f"      ft   ={pax} (Fz,Fx,Fy)  kin={kin} (Ty,Tx,Fz)")

    def snapshot(disp):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / f"fruit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(p), disp)
        print(f"스냅샷 저장: {p}")

    win = "Fruit 6DoF overlay (q=quit, s=snapshot)"
    t_start = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            if latest["img"] is None:
                continue
            # 새 프레임이 없으면 다시 그리지 않는다 (같은 프레임 수천번 렌더 = CPU 폭주 방지).
            # 단 창 응답성을 위해 waitKey 는 계속 돌린다.
            if not latest["new"] and not args.save_after:
                if not args.headless:
                    k = cv2.waitKey(5) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k == ord("s"):
                        snapshot(ov.draw(latest["img"].copy()))
                else:
                    time.sleep(0.005)
                continue
            latest["new"] = False
            disp = ov.draw(latest["img"].copy())
            # --save-after: 지정 초 뒤 자동 스냅샷 후 종료 (창 없이 원격/백그라운드 확인용)
            if args.save_after and time.time() - t_start >= args.save_after:
                snapshot(disp)
                break
            if args.headless:
                continue
            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                snapshot(disp)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def selftest(pw_scale: float = 0.25, ph_scale: float = 0.20):
    """ROS 없이 가짜 pose + 가짜 F/T 로 투영·렌더 검증."""
    img = np.full((480, 640, 3), 40, np.uint8)
    cv2.putText(img, "SELFTEST (fake pose)", (170, 460), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (80, 80, 80), 1)
    ov = Overlay(panel_w_scale=pw_scale, panel_h_scale=ph_scale)
    ov.K = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1.0]])
    ang = math.radians(30)
    q = (0.0, math.sin(ang / 2), 0.0, math.cos(ang / 2))     # y축 30deg
    ov.set_pose((0.02, -0.01, 0.30), q, "camera_color_optical_frame", time.time())
    ov.size = [0.078, 0.072, 0.068]
    # 가짜 F/T: 손가락마다 다른 법선력·전단 방향 (부호·크기 조합을 모두 포함)
    now = time.time()
    ov.paxini.set([  3.2, +1.1, +0.4,      # 엄지  Fz,Fx,Fy
                     1.4, -0.8, +1.2,      # 검지
                     0.0,  0.0,  0.0,      # 중지 (접촉 없음)
                    -2.1, +0.3, -1.5], now)  # 약지 (Fz 음수)
    ov.hand_ft.set([  8.0,  74.0, -281.0,  # 엄지  Ty,Tx,Fz
                      2.0,  40.0,   89.0,
                      6.0, 617.0,  200.0,
                     10.0, -98.0,   53.0], now)
    out = ov.draw(img)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "selftest_overlay.png"
    cv2.imwrite(str(p), out)
    print(f"selftest OK — 박스/축/텍스트 + F/T 패널 2개 렌더 정상, 저장: {p}")


def main():
    ap = argparse.ArgumentParser(description="RealSense 2D 영상 + 과일 6DoF 오버레이")
    ap.add_argument("--raw", action="store_true", help="raw Image 사용(기본 compressed 우선)")
    ap.add_argument("--color-topic", default=None, help="컬러 토픽 직접 지정")
    ap.add_argument("--info-topic", default=None, help="camera_info 토픽 직접 지정")
    ap.add_argument("--selftest", action="store_true", help="ROS 없이 렌더 검증")
    ap.add_argument("--headless", action="store_true", help="창 없이 실행(--save-after 와 함께)")
    ap.add_argument("--save-after", type=float, default=0.0,
                    help="N초 뒤 스냅샷 자동 저장 후 종료 (0=끔)")
    ap.add_argument("--side", default="right", help="paxini/hand 쪽 (right|left)")
    ap.add_argument("--panel-w-scale", type=float, default=0.25,
                    help="F/T 패널 폭 = 영상 너비의 이 배수 (기본 0.25 = 1/4)")
    ap.add_argument("--panel-h-scale", type=float, default=0.20,
                    help="F/T 패널 높이 = 영상 높이의 이 배수 (기본 0.20 = 1/5)")
    ap.add_argument("--ft-scale", type=float, default=0.0,
                    help="paxini 화살표/색 스케일 고정 (0=자동)")
    ap.add_argument("--hand-ft-scale", type=float, default=0.0,
                    help="hand ft 화살표/색 스케일 고정 (0=자동)")
    ap.add_argument("--ft-tau", type=float, default=3.0,
                    help="자동 스케일 감쇠 시간상수[s] — 작으면 민감, 크면 안정 (기본 3)")
    ap.add_argument("--no-ft", action="store_true", help="F/T 패널 끄기")
    args = ap.parse_args()
    (selftest(args.panel_w_scale, args.panel_h_scale)
     if args.selftest else run_ros(args))


if __name__ == "__main__":
    main()
