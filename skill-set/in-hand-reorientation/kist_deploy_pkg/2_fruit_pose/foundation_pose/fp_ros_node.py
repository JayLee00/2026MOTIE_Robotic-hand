#!/usr/bin/env python3
"""FoundationPose ROS2 브리지 — **호스트에서** 실행된다.

    카메라 토픽 ─┬─ (첫 프레임) SAM2 로 과일 마스크
                 └─ RGB-D + K ──TCP──> 컨테이너의 fp_server ──> 4x4 pose
                                                                   │
                          /<ns>/pose (PoseStamped), /<ns>/size ◄───┘

논문대로 **마스크는 첫 프레임에만** 필요하다("the object is detected using an
off-the-shelf method such as Mask R-CNN or CNOS"). 이후에는 직전 자세를 조건으로
크롭해 추적하므로 매 프레임 세그멘테이션이 필요 없다. 그래서 기존 SAM2 를 그대로
초기화용으로만 쓴다.

기본 발행 네임스페이스는 기존 파이프라인(/fruit/*)과 **겹치지 않게** /fruit_fp/* 다.
`--ns /fruit` 로 바꾸면 기존 fruit_overlay.py 가 그대로 받아 그린다(드롭인 비교용).

실행: run_foundation_pose.sh 가 알아서 띄운다. 수동으로는
    python3 fp_ros_node.py --server 127.0.0.1:5577 --diameter 0.07
"""
from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String

import message_filters

HDR = struct.Struct("<Q")
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1)

FM_ROOT = "/home/js/Desktop/vive_franka_teleop/fruit-manipulation"
SAM2_CKPT = os.path.join(FM_ROOT, "sam2.1_hiera_tiny.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# ── fp_server 클라이언트 ────────────────────────────────────────────────────
class FPClient:
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = None

    def connect(self, timeout: float = 5.0):
        s = socket.create_connection(self.addr, timeout=timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(30.0)
        self.sock = s

    def call(self, req: dict) -> dict:
        body = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(HDR.pack(len(body)) + body)
        head = self._recv(HDR.size)
        (n,) = HDR.unpack(head)
        return pickle.loads(self._recv(n))

    def _recv(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise ConnectionError("fp_server 연결 끊김")
            buf.extend(c)
        return bytes(buf)

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None


def mat_to_quat(R: np.ndarray):
    """3x3 회전행렬 → 쿼터니언 [x,y,z,w]."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
                (R[2, 1] - R[1, 2]) / s)
    if i == 1:
        s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
                (R[0, 2] - R[2, 0]) / s)
    s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
            (R[1, 0] - R[0, 1]) / s)


class FoundationPoseNode(Node):
    def __init__(self, a):
        super().__init__("foundation_pose")
        self.a = a
        self.bridge = CvBridge()
        self.K = None
        host, port = a.server.rsplit(":", 1)
        self.client = FPClient(host, int(port))
        self.sam = None
        self.seeded = False      # 스트림 트래커에 씨앗을 심었는지
        self.registered = False
        self.ever_registered = False   # 한 번이라도 등록 성공 → 이후엔 자동 복구 허용
        # 세그는 자세 루프와 분리해 돈다 (같이 돌리면 28Hz→14Hz 로 떨어지고
        # 프레임 간 이동량이 두 배가 되어 회전 정합이 무너진다 — 실측)
        self._sam_lock = threading.Lock()     # self.sam 은 두 스레드가 만진다
        self._seg_lock = threading.Lock()
        self._seg_frame = None                # 세그 스레드에 넘길 최신 (rgb, depth)
        self._seg_since_seed = 0              # 마지막 시드 이후 추적한 프레임 수
        self._seg_hz = 0.0
        self._t_rtt = 0.0        # fp_server 왕복 [ms]
        self._t_srv = 0.0        # 그중 FoundationPose 추론 [ms]
        self._q_prev = None      # 직전 발행 쿼터니언 (부호 연속성용)
        self._sym = None         # 대칭 회전 집합 (메시 크기를 알아야 만들 수 있다)
        self.mesh_extents = None # 메시 축별 크기 (정렬 안 함 — 장축이 어느 축인지 알아야)
        self.n = 0
        self.n_ok = 0
        self.t_last = time.perf_counter()
        self.last_pose = None

        self.lost = 0            # 연속 추적실패 프레임 수
        self.n_reg = 0           # register 횟수
        self.click_pt = None     # 사용자가 찍은 시드 픽셀 (클릭 모드)
        self.last_rgb = None     # GUI 표시용
        self.last_mask = None    # 확인용 마스크 오버레이
        self.size = None         # 비전으로 실측한 크기 [m] (없으면 CAD 공칭)
        self.win = "FoundationPose select  (클릭=물체선택  r=재선택  q=종료)"

        ns = a.ns.rstrip("/")
        self.pub_pose = self.create_publisher(PoseStamped, f"{ns}/pose", QOS)
        self.pub_size = self.create_publisher(Float32MultiArray, f"{ns}/size", QOS)
        # 물체가 바뀌었을 때: 빈 문자열=SAM2 재세그+재등록, 경로=메시 교체 후 재등록
        self.create_subscription(String, f"{ns}/reset", self._on_reset, 10)
        self.get_logger().info(f"발행: {ns}/pose, {ns}/size   재등록: {ns}/reset")

        self.create_subscription(CameraInfo, a.info_topic, self._on_info, QOS)
        sub_c = message_filters.Subscriber(self, Image, a.color_topic, qos_profile=QOS)
        sub_d = message_filters.Subscriber(self, Image, a.depth_topic, qos_profile=QOS)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_c, sub_d], queue_size=5, slop=0.05)
        self.sync.registerCallback(self._on_rgbd)

        self._stop = threading.Event()
        self._seg_thread = threading.Thread(target=self._seg_loop, daemon=True)
        self._seg_thread.start()

        self.create_timer(2.0, self._status)

        if a.click:
            # cv2 GUI 는 메인 스레드에서 돌아야 한다 → 타이머(=spin 스레드)에서 처리
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.win, self._on_mouse)
            self.create_timer(1.0 / 30.0, self._gui)
            self.get_logger().info("클릭 모드: 창에서 물체를 클릭하면 그 지점으로 SAM2 를 겁니다")

    # ── 클릭 GUI ──────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_pt = (float(x), float(y))
            self.registered = False        # 새로 고른 물체로 다시 등록
            self.lost = 0
            self.get_logger().info(f"클릭 ({x},{y}) — SAM2 재세그멘테이션")

    def _gui(self):
        # 화면 그리기 실패로 자세추정이 죽으면 안 된다. rclpy 는 타이머 콜백의 예외를
        # 잡지 않아 실행기째 내려간다 — 표시용 코드는 반드시 스스로 삼켜야 한다.
        try:
            self._gui_draw()
        except KeyboardInterrupt:
            raise
        except Exception as e:                                   # noqa: BLE001
            self.get_logger().warn(f"GUI 갱신 실패(무시): {e}", throttle_duration_sec=5.0)

    def _gui_draw(self):
        if self.last_rgb is None:
            return
        img = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR).copy()
        # 세그 스레드가 last_mask 를 갈아치우므로 반드시 한 번만 집어 로컬에 둔다.
        # 한 줄에서 두 번 읽으면 그 사이에 바뀌어 shape 가 어긋나고, 타이머 콜백에서
        # 예외가 나면 rclpy 실행기가 통째로 죽는다(실측: ValueError 로 노드 사망).
        mk = self.last_mask
        if mk is not None and mk.shape == img.shape[:2]:
            img[mk] = (0.45 * np.array([0, 255, 0]) + 0.55 * img[mk]).astype(np.uint8)
        if self.click_pt is not None:
            p = (int(self.click_pt[0]), int(self.click_pt[1]))
            cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        msg = ("추적 중 — 다른 물체를 클릭하면 그쪽으로 옮겨갑니다" if self.registered
               else "물체를 클릭하세요")
        cv2.putText(img, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if self.registered else (0, 200, 255), 2)
        cv2.imshow(self.win, img)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            raise KeyboardInterrupt
        if k == ord("r"):
            self.click_pt = None
            self.registered = False
            self.get_logger().info("재선택 — 자동 시드로 되돌림")

    # ── SAM2 (fruit-manipulation 의 스트림 트래커 그대로) ────────────────────
    def _load_sam2(self):
        """live_bbox_gui.py 와 같은 Sam2StreamTracker 를 쓴다.

        image predictor 로 매번 새로 프롬프트하면 프레임마다 마스크 입도가 흔들린다.
        video predictor 는 메모리뱅크로 **처음 고른 그 물체**를 이어서 추적하므로
        경계가 부드럽고 튀지 않는다 — 기존 파이프라인에서 검증된 방식이다.
        """
        sys.path.insert(0, FM_ROOT)                                   # in_hand_tracker
        sys.path.insert(0, os.path.join(FM_ROOT, "third_party", "sam2"))
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"SAM2 로드 중 ({dev}) — {SAM2_CKPT}")
        from in_hand_tracker.perception.sam2_stream import Sam2StreamTracker
        self.sam = Sam2StreamTracker(SAM2_CKPT, SAM2_CFG, device=dev)
        self.seeded = False
        self.get_logger().info("SAM2 스트림 트래커 준비 완료 (video predictor + 메모리)")

    def _trim_by_depth(self, m: np.ndarray, depth: np.ndarray):
        """마스크가 물체 경계를 넘어 문 배경 화소를 깊이로 잘라낸다.

        안 자르면 점군이 시선 방향으로 늘어나 크기 실측이 무너진다(실측 826mm).
        """
        m = m & (depth > self.a.depth_band[0]) & (depth < self.a.depth_band[1])
        dm = depth[m]
        dm = dm[dm > 0]
        if dm.size < 20:
            return m
        med = float(np.median(dm))
        slab = 1.5 * float(max(self.a.abc))
        return m & (np.abs(depth - med) <= slab)

    def _seed_point(self, depth: np.ndarray):
        """ROI + 깊이대역 안에서 가장 가까운 덩어리의 무게중심 → SAM2 점 프롬프트.

        기존 live_bbox_gui.py 의 --roi-frac / --depth-band 와 같은 발상이다.
        """
        h, w = depth.shape
        cx, cy, fw, fh = self.a.roi_frac
        x0, x1 = int((cx - fw / 2) * w), int((cx + fw / 2) * w)
        y0, y1 = int((cy - fh / 2) * h), int((cy + fh / 2) * h)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        roi = depth[y0:y1, x0:x1]
        lo, hi = self.a.depth_band
        valid = (roi > lo) & (roi < hi)
        if valid.sum() < 200:
            return None
        # 가장 가까운 5% 깊이만 남겨 (배경 테이블 제거) 무게중심을 잡는다
        thr = np.percentile(roi[valid], 5) + self.a.near_slab
        near = valid & (roi <= thr)
        if near.sum() < 100:
            near = valid
        ys, xs = np.nonzero(near)
        return (float(xs.mean()) + x0, float(ys.mean()) + y0)

    def _measure_size(self, mask: np.ndarray, depth: np.ndarray):
        """마스크+깊이로 실제 과일 크기를 잰다 → /fruit/size.

        CAD 는 종류당 대표 1개뿐이라 개체 크기는 CAD 로 알 수 없다. 그래서 크기는
        비전으로 잰다.

        한계를 분명히 해두면: 카메라는 **앞면만** 본다. 그래서 점군 PCA 의 세 축 중
        시선 방향 축(가장 짧게 나오는 축)은 실제의 절반 수준으로 과소평가된다.
        과일은 장축 둘레로 대체로 회전대칭이므로, 그 축은 중간축으로 대체한다.
        결과는 (장축, 중간축, 중간축) 이며 앞 두 개는 실측에 가깝다.
        """
        vs, us = np.nonzero(mask)
        z = depth[vs, us]
        ok = z > 0
        if ok.sum() < 100:
            return None
        vs, us, z = vs[ok], us[ok], z[ok]
        K = self.K
        pts = np.stack([(us - K[0, 2]) * z / K[0, 0],
                        (vs - K[1, 2]) * z / K[1, 1], z], axis=1)
        c = pts.mean(axis=0)
        ev = np.linalg.eigh(np.cov((pts - c).T))[1][:, ::-1]      # 고유값 큰 순
        proj = (pts - c) @ ev
        ext = np.sort(proj.max(axis=0) - proj.min(axis=0))[::-1]
        a, b = float(ext[0]), float(ext[1])
        return [a, b, b]                                          # 세 번째는 회전대칭 가정

    def _update_size(self, m: np.ndarray, depth: np.ndarray, quiet: bool = False):
        """마스크로 크기를 재서 /fruit/size 값을 갱신. 말이 안 되면 버린다."""
        meas = self._measure_size(m, depth)
        nom = float(max(self.a.abc))
        if meas and not (nom / 3.0 <= meas[0] <= nom * 3.0):
            # 마스크가 물체를 벗어나면 여기서 걸린다. 엉터리 값이 HDF5 에 들어가는
            # 것보다 CAD 공칭치를 쓰는 게 낫다.
            self.get_logger().warn(
                f"비전 실측 {[round(x*1000, 1) for x in meas]} mm 가 CAD 공칭 "
                f"{nom*1000:.0f} mm 와 3배 이상 차이 → 버림", throttle_duration_sec=5.0)
            return
        if not meas:
            return
        # 프레임마다 조금씩 흔들리므로 지수평균으로 눕힌다 (크기는 원래 상수여야 한다)
        self.size = meas if self.size is None else [
            0.85 * s + 0.15 * v for s, v in zip(self.size, meas)]
        if not quiet:
            self.get_logger().info(
                f"비전 실측 크기: {[round(x*1000, 1) for x in meas]} mm "
                f"(CAD 공칭 {[round(x*1000, 1) for x in self.a.abc]} mm)")

    def _depth_at(self, depth: np.ndarray, pt, r: int = 5):
        """시드 픽셀 주변의 유효 깊이 중앙값 [m]. 없으면 None."""
        h, w = depth.shape
        u, v = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= u < w and 0 <= v < h):
            return None
        patch = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        valid = patch[patch > 0]
        return float(np.median(valid)) if valid.size >= 5 else None

    def _make_mask(self, rgb: np.ndarray, depth: np.ndarray):
        # 시드 우선순위:
        #   1) 사용자가 방금 클릭한 지점
        #   2) 마지막으로 성공한 자세의 화면 투영 — 발산 복구 시 물체가 실제로 있는 곳
        #   3) 자동 ROI+깊이 시드 (원래 파이프라인이 쓰던 방식)
        # 첫 등록 전에는 클릭을 기다린다. 자동 시드가 테이블을 잡으면 엉터리 자세가
        # 그대로 발행되기 때문이다. 하지만 한 번 등록에 성공한 뒤로는 자동 복구가
        # 되어야 한다 — 안 그러면 놓친 순간 영원히 못 돌아온다.
        pt = self.click_pt
        how = "클릭"
        if pt is None:
            if not self.ever_registered and self.a.click:
                return None, "클릭 대기 중 — 창에서 과일을 클릭하세요"
            pt, how = self._pose_px(), "자세투영"
            if pt is None or not self._depth_at(depth, pt):
                pt, how = self._seed_point(depth), "자동ROI"
            if pt is None:
                return None, "ROI/깊이대역 안에 물체 없음"
        if self.sam is None:
            self._load_sam2()

        # 스트림 트래커를 그 점으로 다시 씨앗 심는다. 이후 프레임은 _track_mask 가
        # 메모리로 이어받으므로 여기서 다시 프롬프트할 일이 없다.
        # 점 하나만 주면 SAM2 가 과일의 '일부'를 잡는 일이 잦다(실측 4897px, 기대 14000px).
        # 그 조각으로 register 하면 자세 중심이 어긋나 이탈 판정 → 재등록 루프에 빠진다.
        # 메시 크기와 깊이를 알고 있으니 물체가 차지할 박스를 계산해 같이 준다 —
        # live_bbox_gui.py 주석대로 "box prompt yields a much cleaner initial mask".
        box = None
        zc = self._depth_at(depth, pt)
        if zc:
            r = 0.5 * self.K[0, 0] * float(max(self.a.abc)) / zc * 1.15   # 약간 여유
            h, w = depth.shape
            box = [max(0.0, pt[0] - r), max(0.0, pt[1] - r),
                   min(w - 1.0, pt[0] + r), min(h - 1.0, pt[1] + r)]
        with self._sam_lock:
            self.sam.reset()
            self.sam.load_first_frame(rgb)
            raw = self.sam.add_prompt(points=[(float(pt[0]), float(pt[1]))], labels=[1],
                                      box=box)
        self._seg_since_seed = 0
        self.seeded = True

        m = self._trim_by_depth(np.asarray(raw).astype(bool), depth)
        if int(m.sum()) < self.a.min_mask_px:
            self.seeded = False
            return None, f"마스크가 너무 작음 ({int(m.sum())}px)"
        # 씨앗을 심었으면 클릭 지점은 소임을 다했다. 남겨두면 나중에 재등록이 걸릴 때
        # 물체가 이미 옮겨갔는데도 옛 좌표로 다시 씨앗을 심어 엉뚱한 걸 잡는다.
        self.click_pt = None
        self.last_mask = m
        if self.a.size_source == "vision":
            self._update_size(m, depth)
        self.get_logger().info(
            f"SAM2 시드: {int(m.sum())}px, seed=({pt[0]:.0f},{pt[1]:.0f}) [{how}]")
        return m, None

    def _sym_rots(self):
        """과일의 근사 대칭 = **장축 둘레 회전**. 메시에서 가장 긴 축을 기준으로 만든다.

        레몬처럼 장축 둘레로 거의 회전대칭인 물체는 서로 다른 회전이 관측상 똑같이
        그럴듯하다. FoundationPose 는 그중 하나를 고를 뿐이라 재등록 때마다 다른
        대표값이 나올 수 있고, 그게 "축이 휙 바뀌는" 정체다.
        """
        if self._sym is not None:
            return self._sym
        ext = self.mesh_extents
        if ext is None:
            return None
        ax = int(np.argmax(ext))                       # 메시 좌표계의 장축
        a = np.zeros(3)
        a[ax] = 1.0
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        self._sym = []
        for th in np.linspace(0, 2 * np.pi, self.a.sym_steps, endpoint=False):
            # 로드리게스: 축 a 둘레 th 회전
            self._sym.append(np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K))
        self.get_logger().info(
            f"대칭 집합: 메시 축 {ax}(길이 {ext[ax]:.3f}m) 둘레 {self.a.sym_steps}단계")
        return self._sym

    def _snap_to_prev(self, T: np.ndarray) -> np.ndarray:
        """재등록 결과를 직전 자세에 가장 가까운 대칭 동등물로 바꾼다.

        ★ 매 프레임이 아니라 **재등록 직후에만** 부른다. 매번 걸면 물체가 실제로
        장축 둘레로 도는 회전까지 눌러버린다. 불연속은 재등록에서만 생긴다.
        """
        sym = self._sym_rots()
        if sym is None or self.last_pose is None:
            return T
        Rp = self.last_pose[:3, :3]
        R0 = T[:3, :3]
        best, best_d = None, None
        for S in sym:
            R = R0 @ S
            c = (np.trace(Rp.T @ R) - 1.0) / 2.0
            d = float(np.arccos(np.clip(c, -1.0, 1.0)))          # 측지 각도
            if best_d is None or d < best_d:
                best_d, best = d, R
        out = T.copy()
        out[:3, :3] = best
        c0 = (np.trace(Rp.T @ R0) - 1.0) / 2.0
        d0 = float(np.arccos(np.clip(c0, -1.0, 1.0)))
        if d0 - best_d > np.deg2rad(5):
            self.get_logger().info(
                f"대칭 스냅: 직전 자세와 {np.degrees(d0):.0f}° → {np.degrees(best_d):.0f}°")
        return out

    def _pose_px(self):
        """현재 자세를 화면 좌표로 투영 — 재시드 지점으로 쓴다.

        물체가 움직인 뒤에는 처음 클릭한 좌표가 이미 낡았다. FoundationPose 가
        물체 위치를 알고 있으니 그걸 재시드 지점으로 쓰는 게 정확하다.
        """
        T = self.last_pose
        if T is None or self.K is None:
            return None
        z = float(T[2, 3])
        if z <= 0:
            return None
        return (self.K[0, 0] * T[0, 3] / z + self.K[0, 2],
                self.K[1, 1] * T[1, 3] / z + self.K[1, 2])

    def _seg_loop(self):
        """세그 전용 스레드: 메모리로 추적하되 주기적으로/무너지면 다시 씨앗을 심는다.

        live_bbox_gui.py 의 `reseg_every_n` + in-hand gate 와 같은 발상이다.
        메모리만으로 오래 끌면 마스크가 물체 일부로 쪼그라들거나 배경으로 샌다
        (실측: 10600px → 4800px). 주기적으로 다시 프롬프트해 붙잡아 준다.
        """
        t0 = time.perf_counter()
        n = 0
        if self.a.seg_hz <= 0:
            # 세그 끔: 등록 마스크만 쓰고 이후 SAM2 를 아예 안 돌린다.
            # 마스크는 자세추정에 들어가지 않으므로(첫 프레임 등록에만 쓰임) 자세만
            # 볼 거면 이게 가장 빠르고 정확하다.
            self.get_logger().info("연속 세그 끔 (--seg-hz 0) — 자세 주기 우선")
            return
        t_last = 0.0
        min_dt = 1.0 / self.a.seg_hz
        while not self._stop.is_set():
            # 세그는 오버레이·크기용이라 카메라 속도가 필요 없다. 제한을 안 걸면
            # SAM2 가 GPU 를 계속 물고 있어 FoundationPose 가 28Hz→11Hz 로 떨어지고,
            # 프레임 간 이동량이 커져 회전이 튄다(실측).
            now = time.perf_counter()
            if min_dt and now - t_last < min_dt:
                time.sleep(0.005)
                continue
            with self._seg_lock:
                fr = self._seg_frame
                self._seg_frame = None
            if fr is None or not self.seeded:
                time.sleep(0.005)
                continue
            t_last = now
            rgb, depth = fr

            need_reseed = (self.a.reseg_every > 0
                           and self._seg_since_seed >= self.a.reseg_every)
            try:
                with self._sam_lock:
                    if need_reseed:
                        pt = self._pose_px()
                        if pt is None:
                            need_reseed = False
                        else:
                            self.sam.reset()
                            self.sam.load_first_frame(rgb)
                            m = np.asarray(self.sam.add_prompt(
                                points=[(float(pt[0]), float(pt[1]))], labels=[1]))
                            self._seg_since_seed = 0
                    if not need_reseed:
                        m = np.asarray(self.sam.track(rgb))
                        self._seg_since_seed += 1
            except Exception as e:                               # noqa: BLE001
                self.get_logger().warn(f"SAM2 세그 실패: {e}", throttle_duration_sec=5.0)
                time.sleep(0.05)
                continue

            m = self._trim_by_depth(m.astype(bool), depth)
            if int(m.sum()) < self.a.min_mask_px:
                # 마스크가 무너졌다 = 트래커가 물체를 놓쳤다 → 다음 턴에 강제 재시드
                self._seg_since_seed = self.a.reseg_every
                self.get_logger().warn(
                    f"마스크 붕괴 ({int(m.sum())}px) — 자세 위치로 재시드",
                    throttle_duration_sec=3.0)
                continue
            self.last_mask = m
            if self.a.size_source == "vision":
                self._update_size(m, depth, quiet=True)
            n += 1
            dt = time.perf_counter() - t0
            if dt >= 2.0:
                self._seg_hz, n, t0 = n / dt, 0, time.perf_counter()

    # ── 물체 교체 / 재등록 ─────────────────────────────────────────────────
    def _on_reset(self, msg: String):
        """물체가 바뀌었을 때 호출. data 가 비면 재세그만, 경로면 메시까지 교체.

            ros2 topic pub --once /fruit_fp/reset std_msgs/String '{data: ""}'
            ros2 topic pub --once /fruit_fp/reset std_msgs/String '{data: "/…/apple.obj"}'
        """
        path = msg.data.strip()
        if path and self.client.sock is not None:
            try:
                rep = self.client.call({"cmd": "set_mesh", "mesh": path})
                if not rep.get("ok"):
                    self.get_logger().error(f"메시 교체 실패: {rep.get('err')}")
                    return
                ext = rep.get("extents")
                if ext:
                    self.mesh_extents = np.asarray(ext, dtype=float)
                    self._sym = None    # 물체가 바뀌면 대칭 집합도 다시
                    self.a.abc = sorted((float(x) for x in ext), reverse=True)
                self.size = None        # 새 물체 → 이전 실측치는 버린다
                self._q_prev = None
                self.get_logger().info(
                    f"메시 교체됨: {path}  공칭 {[round(x, 4) for x in self.a.abc]} m")
            except Exception as e:                                # noqa: BLE001
                self.get_logger().error(f"메시 교체 호출 실패: {e}")
                self.client.close()
                return
        self.registered = False        # 다음 프레임에서 SAM2 로 다시 마스크
        self.lost = 0
        self.get_logger().info("재등록 요청 — 다음 프레임에서 SAM2 재세그멘테이션")

    def _tracking_ok(self, T: np.ndarray, depth: np.ndarray) -> bool:
        """추정된 자세가 아직 실제 물체 위에 있는지 값싸게 검사.

        자세의 원점은 메시 **중심**이지만 깊이센서가 보는 건 **앞면**이다. 볼록한
        물체라면 관측 깊이는 [z-반지름, z+반지름] 안에 있어야 하므로, 여기에
        여유(check_tol)를 더해 판정한다. 반지름을 안 빼면 늘 이탈로 오판해
        재등록 루프에 빠진다(실측으로 확인).
        """
        z = float(T[2, 3])
        if z <= 0:
            return False
        u = self.K[0, 0] * T[0, 3] / z + self.K[0, 2]
        v = self.K[1, 1] * T[1, 3] / z + self.K[1, 2]
        h, w = depth.shape
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            return False
        r = self.a.check_win
        patch = depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
        valid = patch[patch > 0]
        if valid.size < 10:
            # 깊이 구멍(반사·어두운 표면·손 가림)일 뿐 물체가 사라진 게 아닐 수 있다.
            # 여기서 실패로 치면 멀쩡한 추적을 계속 재등록시킨다 — 모르면 통과시킨다.
            return True
        radius = 0.5 * max(self.a.abc)
        return abs(float(np.median(valid)) - z) <= radius + self.a.check_tol

    # ── 콜백 ──────────────────────────────────────────────────────────────
    def _on_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"K 수신: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} "
                                   f"cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}")

    def _on_rgbd(self, c_msg: Image, d_msg: Image):
        if self.K is None:
            return
        if self.client.sock is None:
            try:
                self.client.connect()
                # 서버가 들고 있는 메시의 실제 크기를 받아 /size 로 쓴다.
                # --diameter 추정값을 쓰면 오버레이 박스가 실물과 어긋난다.
                rep = self.client.call({"cmd": "ping"})
                ext = rep.get("extents")
                if ext:
                    self.mesh_extents = np.asarray(ext, dtype=float)   # 축 순서 유지
                    self._sym = None
                    self.a.abc = sorted((float(x) for x in ext), reverse=True)
                    self.get_logger().info(
                        f"메시 크기 수신: {[round(x, 4) for x in self.a.abc]} m")
                self.get_logger().info(f"fp_server 접속: {self.a.server}")
            except Exception as e:                                # noqa: BLE001
                self.get_logger().warn(f"fp_server 접속 실패: {e}", throttle_duration_sec=5.0)
                return

        rgb = self.bridge.imgmsg_to_cv2(c_msg, "rgb8")
        self.last_rgb = rgb
        d = self.bridge.imgmsg_to_cv2(d_msg, "passthrough")
        depth = (d.astype(np.float32) / 1000.0) if d.dtype == np.uint16 else d.astype(np.float32)
        depth[(depth < self.a.depth_band[0]) | (depth > self.a.depth_band[1])] = 0.0

        req = {"cmd": "track", "rgb": rgb, "depth": depth, "K": self.K}
        if not self.registered:
            mask, err = self._make_mask(rgb, depth)
            if mask is None:
                self.get_logger().warn(f"초기 마스크 실패: {err}", throttle_duration_sec=3.0)
                return
            req = {"cmd": "register", "rgb": rgb, "depth": depth, "K": self.K, "mask": mask}
        else:
            # 세그는 별도 스레드가 돈다. 여기서 track() 을 부르면 자세 루프가 그만큼
            # 느려지고(28Hz→14Hz), 프레임 간 이동량이 두 배가 되어 회전 정합이
            # 무너진다(실측). 최신 프레임만 넘겨주고 즉시 자세로 넘어간다.
            with self._seg_lock:
                self._seg_frame = (rgb, depth)

        try:
            _t = time.perf_counter()
            rep = self.client.call(req)
            # 왕복 - 서버 추론시간 = 직렬화·TCP·전처리에 쓴 시간. 둘을 갈라 봐야
            # 병목이 FoundationPose 인지 파이썬 오버헤드인지 알 수 있다.
            self._t_rtt = (time.perf_counter() - _t) * 1e3
            self._t_srv = float(rep.get("ms") or 0.0)
        except Exception as e:                                    # noqa: BLE001
            self.get_logger().error(f"fp_server 호출 실패: {e} — 재접속")
            self.client.close()
            self.registered = False
            return

        self.n += 1
        if not rep.get("ok"):
            self.get_logger().warn(f"추정 실패: {rep.get('err')}", throttle_duration_sec=3.0)
            return
        just_registered = not self.registered
        if just_registered:
            self.registered = True
            self.ever_registered = True
            self.n_reg += 1
            self.get_logger().info(f"등록(register) 완료 #{self.n_reg} — 이후 트래킹")

        T = np.asarray(rep["pose"], dtype=np.float64)
        if just_registered:
            # register() 는 회전 후보 252개를 직전 자세와 무관하게 새로 뿌린다.
            # 물체가 가만히 있어도 대칭적으로 동등한 다른 대표값이 나올 수 있으므로,
            # 직전 자세에 가장 가까운 동등물로 맞춰 궤적을 잇는다.
            T = self._snap_to_prev(T)

        # 물체가 바뀌거나 트래커가 흘러가면 자동으로 다시 등록한다
        if self.a.auto_reset and self._tracking_ok(T, depth):
            self.lost = 0
        elif self.a.auto_reset:
            self.lost += 1
            if self.lost >= self.a.lost_patience:
                self.get_logger().warn(
                    f"추적 이탈 {self.lost}프레임 — SAM2 로 재등록합니다")
                self.registered = False
                self.lost = 0
                return          # 이 프레임은 버리고 다음 프레임에서 재세그
            return              # 의심스러운 자세는 발행하지 않는다

        self.n_ok += 1
        self.last_pose = T
        self._publish(T, c_msg.header)

    def _publish(self, T: np.ndarray, header):
        q = np.asarray(mat_to_quat(T[:3, :3]), dtype=float)
        # q 와 -q 는 같은 회전이지만 숫자로는 큰 점프로 보인다. 부호를 직전 값에
        # 맞춰 두지 않으면 소비측(HDF5·학습)이 실제보다 훨씬 튀는 신호로 읽는다.
        if self._q_prev is not None and float(q @ self._q_prev) < 0.0:
            q = -q
        self._q_prev = q
        p = PoseStamped()
        p.header.stamp = header.stamp
        p.header.frame_id = header.frame_id or self.a.frame_id
        p.pose.position.x, p.pose.position.y, p.pose.position.z = (float(v) for v in T[:3, 3])
        p.pose.orientation.x, p.pose.orientation.y = float(q[0]), float(q[1])
        p.pose.orientation.z, p.pose.orientation.w = float(q[2]), float(q[3])
        self.pub_pose.publish(p)

        s = Float32MultiArray()
        s.data = [float(x) for x in (self.size or self.a.abc)]
        self.pub_size.publish(s)

    def _status(self):
        now = time.perf_counter()
        hz = self.n / max(1e-9, now - self.t_last)
        self.n, self.t_last = 0, now
        if self.last_pose is None:
            self.get_logger().warn(f"아직 자세 없음 (K={'O' if self.K is not None else 'X'}, "
                                   f"등록={'O' if self.registered else 'X'})")
            return
        t = self.last_pose[:3, 3]
        q = mat_to_quat(self.last_pose[:3, :3])
        # 마스크 중심을 같이 찍는다 — 이게 물체를 따라 움직여야 세그가 추적 중인 것이다
        seg = ""
        mk = self.last_mask                       # 위와 같은 이유로 스냅샷
        if mk is not None and mk.any():
            ys, xs = np.nonzero(mk)
            seg = (f"  seg=({xs.mean():.0f},{ys.mean():.0f}) "
                   f"{int(mk.sum())}px @{self._seg_hz:.0f}Hz")
        self.get_logger().info(
            f"{hz:4.1f}Hz  pos=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]m  "
            f"quat=[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}]{seg}  "
            f"| 추론 {self._t_srv:.0f}ms / 왕복 {self._t_rtt:.0f}ms "
            f"(전송·직렬화 {self._t_rtt - self._t_srv:.0f}ms)")


def main():
    ap = argparse.ArgumentParser(description="FoundationPose ROS2 브리지 (호스트)")
    ap.add_argument("--server", default="127.0.0.1:5577")
    ap.add_argument("--ns", default="/fruit_fp",
                    help="발행 네임스페이스. '/fruit' 로 주면 기존 오버레이가 그대로 받는다")
    ap.add_argument("--color-topic", default="/front_cam/front/color/image_fast")
    ap.add_argument("--depth-topic", default="/front_cam/front/aligned_depth_to_color/image_raw")
    ap.add_argument("--info-topic", default="/front_cam/front/color/camera_info")
    ap.add_argument("--frame-id", default="camera_color_optical_frame")
    ap.add_argument("--roi-frac", default="0.5,0.5,0.9,0.9",
                    help="초기 마스크를 찾을 ROI (cx,cy,w,h 비율)")
    ap.add_argument("--depth-band", default="0.15,1.2", help="유효 깊이 [m] 'min,max'")
    ap.add_argument("--near-slab", type=float, default=0.05,
                    help="가장 가까운 깊이로부터 이 두께[m] 안쪽만 시드로 사용")
    ap.add_argument("--min-mask-px", type=int, default=400)
    ap.add_argument("--seg-hz", type=float, default=5.0,
                    help="등록 이후 연속 세그 주기 [Hz]. **0 = 끔**(자세 주기 최대). "
                         "마스크는 자세추정에 안 쓰이고 오버레이·크기 측정용이라, "
                         "돌아가는 물체의 자세가 중요하면 낮추거나 끄는 게 맞다. "
                         "무제한으로 돌리면 GIL 경합으로 자세가 28Hz→7Hz 로 떨어진다")
    ap.add_argument("--reseg-every", dest="reseg_every", type=int, default=30,
                    help="세그 스레드가 이 프레임 수마다 자세 위치로 재시드 (0=끔). "
                         "메모리만으로 오래 끌면 마스크가 쪼그라들거나 배경으로 샌다 "
                         "— live_bbox_gui.py 의 reseg_every_n 과 같은 값")
    ap.add_argument("--size-source", choices=["vision", "cad"], default="vision",
                    help="/fruit/size 를 무엇으로 낼지. vision=마스크+깊이 실측(기본, "
                         "개체마다 크기가 달라서), cad=대표 CAD 공칭치수")
    ap.add_argument("--no-click", dest="click", action="store_false", default=True,
                    help="클릭 선택 창을 끄고 ROI+깊이 자동 시드만 사용")
    ap.add_argument("--no-auto-reset", dest="auto_reset", action="store_false",
                    default=True,
                    help="추적 이탈 시 자동 재등록 끄기. 켜두는 게 기본이다 — 끄면 "
                         "한 번 발산했을 때(실측: z 0.31→0.11m) 영원히 못 돌아온다")
    ap.add_argument("--sym-steps", type=int, default=36,
                    help="장축 둘레 대칭을 몇 단계로 볼지. 재등록 직후 직전 자세에 가장 "
                         "가까운 동등 회전을 고르는 데만 쓴다(매 프레임 아님)")
    ap.add_argument("--check-tol", type=float, default=0.15,
                    help="자세 z 와 관측 깊이 중앙값의 허용 차 [m] (물체 반지름에 더해짐). "
                         "이건 '발산 감지기'이지 정밀도 검사가 아니다. 빡빡하게 잡으면 "
                         "멀쩡한 추적을 1~2초마다 재등록시켜 오히려 자세가 망가진다")
    ap.add_argument("--check-win", type=int, default=6,
                    help="깊이 비교 창 반경 [px]")
    ap.add_argument("--lost-patience", type=int, default=30,
                    help="이 프레임 수만큼 연속 이탈하면 재등록 (register 가 ~1.3s 라 넉넉히)")
    ap.add_argument("--diameter", type=float, default=0.070, help="과일 지름 [m]")
    ap.add_argument("--abc", default=None, help="축별 지름 'a,b,c' [m]")
    a = ap.parse_args()

    a.roi_frac = tuple(float(x) for x in a.roi_frac.split(","))
    a.depth_band = tuple(float(x) for x in a.depth_band.split(","))
    a.abc = ([float(x) for x in a.abc.split(",")] if a.abc
             else [a.diameter, a.diameter, a.diameter])

    rclpy.init()
    node = FoundationPoseNode(a)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._stop.set()
        node._seg_thread.join(timeout=2.0)
        node.client.close()
        if a.click:
            cv2.destroyAllWindows()
        node.destroy_node()


if __name__ == "__main__":
    main()
