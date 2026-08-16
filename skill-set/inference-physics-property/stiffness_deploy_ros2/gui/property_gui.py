#!/usr/bin/env python3
"""property_gui.py — 물성(크기/강성/무게) 추정 결과 GUI (별도 프로세스).

레이아웃 (스케치 반영):
  ┌──────────────────────────┬────────────────────┐
  │ SIZE       42.03 mm      │ sample_1           │
  ├──────────────────────────┤ ● DONE             │
  │ STIFFNESS  1.97 N/mm (MID)├───────────────────┤
  ├──────────────────────────┤ RECENT             │
  │ WEIGHT     68.35 g       │  sample_2: 42.03…  │
  │                          │  sample_1: 20.12…  │
  └──────────────────────────┴────────────────────┘
  · 왼쪽: 물성 3개를 큰 숫자로 표시 (SIZE / STIFFNESS / WEIGHT)
  · 강성 값 오른쪽에는 LOW / MID / HIGH 그룹 배지 — 기준은 상단 STIFFNESS_MID_RANGE
  · 오른쪽 상단: 샘플 이름 + 상태 표시
  · 오른쪽 하단: 최근 측정 15개 리스트 — 강성 높은 순으로 정렬(현재 샘플은 진하게),
    이름 옆에 LOW / MID / HIGH 그룹 표시
  · 상태 표시는 READY / MEASURING / DONE 3가지
  · 글자 크기는 **실제 창 크기에서 자동 산출**한다(전체화면 기준). 조절은 상단
    R_* / VALUE_TRIM 비율만 만지면 되고, 창 크기가 바뀌면 자동으로 다시 잡힌다.

deploy 측이 발행하는 /property/result (std_msgs/String, JSON) 를 구독해 tkinter 창에 표시.
로봇 제어 프로세스와 완전히 분리 → 실시간 루프에 영향 없음.

메시지 JSON (예):
  {"phase":"done", "sample":"sample_1",
   "stiffness":1.97, "stiffness_std":0.3, "stiffness_max":6.06,
   "weight":68.35, "weight_std":6.0,
   "diameter":42.03, "diameter_std":2.0}
  phase: idle | measuring | done | error.  (구 stiffness 메시지의 std/norm_max 도 호환)
  · 추론 모델은 품종/과일 이름도, 등급(soft/mid/hard)도 내지 않으므로
    GUI 는 값(크기/강성/무게)만 표시한다.
  · sample 없으면 sample_1, sample_2 … 자동 채번.
  · diameter 대신 size 필드로 보내도 동작.
  · image 필드는 무시한다 — 샘플 사진 표시는 제거됨.

실행:
  source env.sh && python3 .../gui/property_gui.py           # ROS 실데이터
  /usr/bin/python3 .../gui/property_gui.py --demo            # ROS 없이 레이아웃 확인
"""
from __future__ import annotations

import colorsys
import json
import os
import sys
import tkinter as tk
from collections import deque
from tkinter import font as tkfont

try:  # ROS 없는 PC 에서도 GUI 만 확인 가능하게 rclpy 는 선택적.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                           QoSDurabilityPolicy, QoSHistoryPolicy)
    from std_msgs.msg import String
    _HAVE_ROS = True
except ImportError:
    _HAVE_ROS = False
    Node = object

RESULT_TOPIC = "/property/result"

SAMPLE_PREFIX = "sample"          # 자동 채번 이름: sample_1, sample_2 …

# ── 표시할 물성 3개 (스케치 순서: size → stiffness → weight) ──────
#   key      : 메시지의 값 필드 (std=<key>_std, max=<key>_max)
#   alias    : 대체 필드명
#   gradient : True 면 값 글자에 빨강(작음)→초록(큼) 그라데이션. 기본은 전부 검정.
PROPS = [
    {"key": "diameter",  "alias": "size",  "label": "SIZE",      "unit": "mm",
     "gradient": False, "default_max": 120.0},
    {"key": "stiffness", "alias": None,    "label": "STIFFNESS", "unit": "N/mm",
     "gradient": False, "default_max": 10.0, "group": True},
    {"key": "weight",    "alias": None,    "label": "WEIGHT",    "unit": "g",
     "gradient": False, "default_max": 300.0},
]

# ── 강성 그룹 (LOW / MID / HIGH) ────────────────────────────────
#   MID 구간만 지정하면 나머지는 자동으로 갈린다.
#     val <  MID 하한          → LOW
#     MID 하한 ≤ val ≤ MID 상한 → MID
#     val >  MID 상한          → HIGH
#   실제 강성은 대략 0 ~ 15 N/mm 범위로 나온다.
STIFFNESS_MID_RANGE = (3.5, 8.5)      # (MID 하한, MID 상한) 단위 N/mm
GROUP_NAMES = ("LOW", "MID", "HIGH")  # 순서 고정: 낮음 / 중간 / 높음
GROUP_COLORS = {                      # 배지 글자·테두리 색
    "LOW":  "#0a84ff",
    "MID":  "#c67c0a",
    "HIGH": "#d92d20",
}
GROUP_KEY = next((p["key"] for p in PROPS if p.get("group")), None)   # 그룹 판정 대상 물성

SORT_KEY = "stiffness"            # 최근 리스트 정렬 기준 (내림차순)
RECENT_MAX = 15

# ── 글자 크기 (전체화면 기준 자동 산출) ─────────────────────────
#   창은 시작 시 최대화된다. 왼쪽 물성 숫자를 **화면에 들어가는 최대 크기**로 잡고
#   (세로 3행 + 가로 폭 둘 다 실측), 나머지 글자는 그 숫자에 대한 비율로 정한다.
#   → 1080p 든 4K 든 각각 "꽉 찬" 크기가 나온다. 손으로 만질 곳은 아래 비율뿐.
VALUE_PT_MAX = 280            # 숫자 상한(pt) — 4K 등 큰 화면에서만 도달
VALUE_PT_MIN = 30             # 하한 — 창을 작게 줄여도 이보다 작아지지 않음
R_HEAD = 0.27                 # ★ 항목 라벨(SIZE/STIFFNESS/WEIGHT) / 숫자
R_UNIT = 0.42                 # 단위(mm·N/mm·g)            / 숫자
R_STD = 0.15                  # ± 편차                      / 숫자
R_GROUP_PILL = 0.575          # ★ LOW·MID·HIGH 배지 글자    / 숫자
R_STATUS_PILL = 0.16          # READY·MEASURING·DONE 배지   / 숫자
R_SAMPLE = 0.26               # 오른쪽 샘플명               / 숫자
#   ★ RECENT 리스트 — 15행이 오른쪽 칸에 다 들어와야 해서 숫자만큼 키울 수 없다.
#     숫자에 비례하되 원본(11pt) 대비 아래 배율 범위로 묶는다. 1080p 에서 정확히 2.5배.
R_RECENT = 0.15
RECENT_MULT_MIN, RECENT_MULT_MAX = 2.5, 3.4
#   왼쪽(물성) 칸이 가져갈 수 있는 최대 가로 비율.
LEFT_SHARE_MAX = 0.55
#   ★ 왼쪽 숫자 크기 미세조정 — 창에 들어가는 최대치에 이 배율을 곱한다.
#     1.0 = 꽉 채움(너무 큼), 낮출수록 숫자가 작아진다. 여기만 만지면 된다.
VALUE_TRIM = 0.75

# ── 팔레트 (라이트 테마) ────────────────────────────────────────
BG = "#ffffff"
PANEL = "#f4f6f9"
STROKE = "#dde2ea"
FG = "#14171c"
DIM = "#59636f"
FAINT = "#9aa3b2"
ACCENT = "#0a84ff"
AMBER = "#c67c0a"
RED = "#d92d20"
GREEN = "#1a8f3c"

MARGIN = 22


def latched_qos() -> "QoSProfile":
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def pick_ui_font(root: tk.Misc) -> str:
    prefs = ("Ubuntu", "Inter", "Roboto", "Noto Sans", "Cantarell", "Open Sans",
             "Noto Sans CJK KR", "DejaVu Sans", "Liberation Sans", "Helvetica")
    fams = set(tkfont.families(root))
    for cand in prefs:
        if cand in fams:
            return cand
    for cand in prefs:
        try:
            actual = tkfont.Font(root=root, family=cand, size=12).actual("family")
        except tk.TclError:
            continue
        if actual and actual.lower().replace(" ", "").startswith(cand.lower().replace(" ", "")[:5]):
            return cand
    return "TkDefaultFont"


def grad_hex(frac: float, s: float = 0.75, v: float = 0.82) -> str:
    """frac 0→빨강(작음), 1→초록(큼)."""
    frac = max(0.0, min(1.0, frac))
    r, g, b = colorsys.hsv_to_rgb((1.0 / 3.0) * frac, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def grad_text(frac: float) -> str:
    return grad_hex(frac, s=0.90, v=0.56)


def group_of(val) -> str | None:
    """강성값 → LOW / MID / HIGH (STIFFNESS_MID_RANGE 기준)."""
    if val is None:
        return None
    lo, hi = STIFFNESS_MID_RANGE
    if val < lo:
        return GROUP_NAMES[0]
    if val <= hi:
        return GROUP_NAMES[1]
    return GROUP_NAMES[2]


def _round_rect(c: tk.Canvas, x1, y1, x2, y2, r, **kw):
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


if _HAVE_ROS:
    class ResultSub(Node):
        """/property/result 구독 → 최신 상태 dict 만 보관."""

        def __init__(self) -> None:
            super().__init__("property_gui")
            self.latest: dict = {"phase": "idle"}
            self.create_subscription(String, RESULT_TOPIC, self._cb, latched_qos())

        def _cb(self, msg) -> None:
            try:
                self.latest = json.loads(msg.data)
            except (ValueError, TypeError):
                pass
else:
    ResultSub = None


class PropertyGui:
    def __init__(self, root: tk.Tk, node=None) -> None:
        self.root = root
        self.node = node
        self.state: dict = {"phase": "idle"}
        self.ff = pick_ui_font(root)
        self.nf = self.ff
        if self.ff == "TkDefaultFont":
            print("[GUI] 폰트 미적용(TkDefaultFont) — miniforge python 대신 /usr/bin/python3 로 실행하세요.")

        self._anim = 0
        self._done_sig = None                                # 새 결과 판별용 서명
        self._disp = {p["key"]: 0.0 for p in PROPS}          # 카운트업 애니메이션 현재값
        self._count = 0                                      # 샘플 자동 채번 카운터
        self._sample = None                                  # 현재 샘플 이름
        self._recent = deque(maxlen=RECENT_MAX)              # 최근 측정 보관(최신이 앞), 표시는 강성순
        self._recent_sig = None

        root.title("Property Estimator")
        root.configure(bg=BG)
        #   1차: 화면 크기로 잡아 두고, 최대화가 끝난 뒤 실제 창 크기로 다시 계산한다.
        self._setup_sizes(root.winfo_screenwidth(), root.winfo_screenheight())
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
        root.minsize(900, 600)
        #   전체화면(최대화) 로 시작 — 폰트 크기를 이 상태 기준으로 잡았다.
        for attempt in (lambda: root.attributes("-zoomed", True),
                        lambda: root.state("zoomed")):
            try:
                attempt()
                break
            except tk.TclError:
                continue

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=MARGIN, pady=MARGIN)
        #   오른쪽은 실측 수요(_setup_sizes)만큼 고정하고, 남는 폭은 전부 왼쪽 숫자에.
        outer.columnconfigure(0, weight=1)                    # 왼쪽 물성부
        outer.columnconfigure(1, weight=0)                    # 세로 구분선
        outer.columnconfigure(2, weight=0, minsize=self.right_need)   # 오른쪽 사진/리스트
        outer.rowconfigure(0, weight=1)

        left = tk.Frame(outer, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        tk.Frame(outer, bg=STROKE, width=1).grid(row=0, column=1, sticky="ns")
        right = tk.Frame(outer, bg=BG)
        right.grid(row=0, column=2, sticky="nsew", padx=(18, 0))

        self.outer = outer
        self._build_props(left)
        self._build_right(right)

        self._render(self.state)
        #   WM 이 최대화를 끝낸 뒤 실제 창 크기로 재계산 (듀얼 모니터 대응).
        self._resize_job = None
        root.bind("<Configure>", self._on_configure)
        root.after(150, self._relayout)
        root.after(33, self._pump)

    # ── 창 크기 변화 → 글자 크기 재계산 ──────────────────────────
    def _on_configure(self, ev) -> None:
        if ev.widget is not self.root:
            return
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(180, self._relayout)   # 디바운스

    def _relayout(self) -> None:
        self._resize_job = None
        w, h = self.root.winfo_width(), self.root.winfo_height()
        if w < 200 or h < 200:                      # 아직 매핑 전
            return
        if abs(w - self.win_w) < 24 and abs(h - self.win_h) < 24:
            return
        self._setup_sizes(w, h)
        self._apply_fonts()

    def _apply_fonts(self) -> None:
        """_setup_sizes 결과를 이미 만들어진 위젯에 반영."""
        for lbl in self._head_lbl:
            lbl.configure(font=self.NF(self.pt_head, True))
        for k in self._val_lbl:
            self._val_lbl[k].configure(font=self.NF(self.pt_value, True))
            self._unit_lbl[k].configure(font=self.NF(self.pt_unit))
            self._std_lbl[k].configure(font=self.NF(self.pt_std, True))
        for c in self._grp_pill.values():
            c.configure(width=self.gpill_w, height=self.gpill_h)
        self.sample_lbl.configure(font=self.NF(self.pt_sample, True))
        self.pill.configure(width=self.spill_w, height=self.spill_h)
        self.recent_hdr.configure(font=self.NF(self.pt_recent, True))
        self.outer.columnconfigure(2, minsize=self.right_need)
        #   캔버스·리스트는 좌표를 다시 그려야 반영된다.
        self._set_status(self.state.get("phase", "idle"))
        self._recent_sig = None
        self._render(self.state)

    # ── 크기 산출 (전체화면 기준) ────────────────────────────────
    def _setup_sizes(self, win_w: int, win_h: int) -> None:
        """왼쪽 물성 숫자를 창에 들어가는 최대 크기로 잡고 나머지를 비례 배분.

        ★ 화면(winfo_screen*) 이 아니라 **실제 창 크기**로 계산한다 — 듀얼 모니터면
          가상 화면은 5120 이어도 최대화는 모니터 하나(2560)에만 되기 때문.
        """
        self.win_w, self.win_h = win_w, win_h
        avail_h = win_h - 90                # 상하 MARGIN + 구분선 여유
        body_w = win_w - 2 * MARGIN - 36

        def m(text, pt, bold=False):
            """해당 pt 에서 text 의 실제 픽셀 폭 (폰트마다 자간이 달라 상수로는 못 맞춘다)."""
            f = tkfont.Font(root=self.root, family=self.nf, size=pt,
                            weight="bold" if bold else "normal")
            return max(1, f.measure(text))

        def recent_row_w(pt):
            """RECENT 한 행의 실제 폭 — Label(width=N) 은 N × '0' 폭이다."""
            small = max(8, int(pt * 0.92))
            z_s, z_n = m("0", small, True), m("0", pt, True)
            vals = " · ".join(f"888.88 {p['unit']}" for p in PROPS)
            return (7 + 5) * z_s + 11 * z_n + m(vals, pt) + 24

        # 1) 세로 제약으로 숫자 후보를 먼저 잡는다 (한 행 ≈ 숫자 pt 의 2.25배).
        by_h = avail_h / len(PROPS) / 2.25

        # 2) RECENT pt — 숫자에 비례시키되 15행이 세로로 들어와야 하고,
        #    가로도 화면의 절반을 넘지 않아야 한다. 실측으로 1pt 씩 줄여 맞춘다.
        want = min(by_h * R_RECENT, 11 * RECENT_MULT_MAX)
        want = max(want, 11 * RECENT_MULT_MIN)          # 되도록 이 배율 이상
        by_rows = (avail_h - 120) / RECENT_MAX / 1.66
        pt_r = int(max(9, min(want, by_rows)))
        while pt_r > 9 and recent_row_w(pt_r) > body_w * 0.52:
            pt_r -= 1
        self.pt_recent = pt_r

        # 3) 오른쪽 칸은 실측 수요만큼만 차지하고, 남는 폭을 왼쪽 숫자가 쓴다.
        #    (5:3 고정 비율이면 좁은 화면에서 RECENT 가 잘린다)
        #    단 왼쪽이 지나치게 넓어지지 않도록 LEFT_SHARE_MAX 로 제한한다.
        self.right_need = int(recent_row_w(pt_r))
        left_w = min(body_w - self.right_need - 36, body_w * LEFT_SHARE_MAX)

        # 4) 숫자 pt — 숫자 + 단위 + 편차 + 배지 가 왼쪽 칸에 들어가야 한다.
        #    전부 숫자 pt 에 비례하므로 100pt 로 재서 1pt 당 폭(k)을 구해 역산한다.
        k = (m("888.88", 100, True) / 100
             + m("mm", 100) / 100 * R_UNIT
             + m("+-0.00", 100, True) / 100 * R_STD
             + R_GROUP_PILL * 5.4                # LOW/MID/HIGH 배지 폭(gpill_w 와 동일 계수)
             + 0.20)                             # 좌측 들여쓰기
        by_w = (left_w - 60) / k                 # 60 = 라벨 사이 고정 여백 합
        v = int(max(VALUE_PT_MIN,
                    min(by_h, by_w, VALUE_PT_MAX) * VALUE_TRIM))
        self.pt_value = v

        self.pt_head = max(9, int(v * R_HEAD))
        self.pt_unit = max(9, int(v * R_UNIT))
        self.pt_std = max(8, int(v * R_STD))
        self.pt_gpill = max(9, int(v * R_GROUP_PILL))     # LOW/MID/HIGH
        self.pt_spill = max(8, int(v * R_STATUS_PILL))    # READY/MEASURING/DONE
        self.pt_sample = max(10, int(v * R_SAMPLE))

        #   캔버스로 그리는 요소는 픽셀 좌표라 글자와 같이 키워야 잘리지 않는다.
        self.gpill_h = int(self.pt_gpill * 2.4)
        self.gpill_w = int(self.pt_gpill * 5.4)           # "HIGH" 4글자 + 좌우 여백
        self.spill_h = int(self.pt_spill * 2.6)
        self.spill_w = int(self.pt_spill * 13.0)
        self.indent = int(v * 0.20)         # 숫자 행 좌측 들여쓰기

        print(f"[GUI] 창 {win_w}x{win_h} → 숫자 {v}pt(원본 38, {v / 38:.1f}배) · "
              f"단위 {self.pt_unit}pt(19) · "
              f"RECENT {self.pt_recent}pt(11, {self.pt_recent / 11:.1f}배)")

    def NF(self, size: int, bold: bool = False):
        return (self.nf, size, "bold" if bold else "normal")

    # ── 왼쪽: 물성 3행 ───────────────────────────────────────────
    def _build_props(self, parent) -> None:
        self._val_lbl, self._unit_lbl, self._std_lbl = {}, {}, {}
        self._grp_pill = {}
        self._head_lbl = []                 # 창 크기 바뀌면 폰트 다시 먹인다
        for i, p in enumerate(PROPS):
            r = i * 2
            parent.rowconfigure(r, weight=1)
            cell = tk.Frame(parent, bg=BG)
            cell.grid(row=r, column=0, sticky="nsew")
            cell.columnconfigure(0, weight=1)

            head = tk.Frame(cell, bg=BG)
            head.pack(fill="x", anchor="w", pady=(10, 0))
            hl = tk.Label(head, text=p["label"], bg=BG, fg=DIM,
                          font=self.NF(self.pt_head, True))
            hl.pack(side="left")
            self._head_lbl.append(hl)

            body = tk.Frame(cell, bg=BG)
            body.pack(fill="x", anchor="w", pady=(2, 10), padx=(self.indent, 0))
            self._val_lbl[p["key"]] = tk.Label(body, text="--.--", bg=BG, fg=FAINT,
                                               font=self.NF(self.pt_value, True))
            self._val_lbl[p["key"]].pack(side="left", anchor="s")
            self._unit_lbl[p["key"]] = tk.Label(body, text=p["unit"], bg=BG, fg=DIM,
                                                font=self.NF(self.pt_unit))
            self._unit_lbl[p["key"]].pack(side="left", anchor="s", padx=(9, 0), pady=(0, 6))
            self._std_lbl[p["key"]] = tk.Label(body, text="", bg=BG, fg=FAINT,
                                               font=self.NF(self.pt_std, True))
            self._std_lbl[p["key"]].pack(side="left", anchor="s", padx=(12, 0), pady=(0, 8))

            if p.get("group"):      # 값 오른쪽 빈 공간에 LOW / MID / HIGH 배지
                pill = tk.Canvas(body, width=self.gpill_w, height=self.gpill_h,
                                 bg=BG, highlightthickness=0)
                pill.pack(side="right", anchor="s", padx=(12, 6), pady=(0, 6))
                self._grp_pill[p["key"]] = pill

            if i < len(PROPS) - 1:
                sep = tk.Frame(parent, bg=STROKE, height=1)
                sep.grid(row=r + 1, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)

    # ── 오른쪽: 사진 + 샘플명 + 상태, 그리고 최근 리스트 ──────────
    def _build_right(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        top = tk.Frame(parent, bg=BG)
        top.grid(row=0, column=0, sticky="ew", pady=(10, 12))

        meta = tk.Frame(top, bg=BG)
        meta.pack(side="left", fill="both", expand=True)
        self.sample_lbl = tk.Label(meta, text="—", bg=BG, fg=FG,
                                   font=self.NF(self.pt_sample, True),
                                   anchor="w", justify="left")
        self.sample_lbl.pack(anchor="w", pady=(6, 8))
        self.pill = tk.Canvas(meta, width=self.spill_w, height=self.spill_h,
                              bg=BG, highlightthickness=0)
        self.pill.pack(anchor="w")

        tk.Frame(parent, bg=STROKE, height=1).grid(row=1, column=0, sticky="ew")

        lst = tk.Frame(parent, bg=BG)
        lst.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self.recent_hdr = tk.Label(lst, text=f"RECENT {RECENT_MAX}  ·  STIFFNESS ↓",
                                   bg=BG, fg=DIM,
                                   font=self.NF(self.pt_recent, True))
        self.recent_hdr.pack(anchor="w", pady=(0, 6))
        self.recent_box = tk.Frame(lst, bg=BG)
        self.recent_box.pack(fill="both", expand=True)

    def _set_status(self, phase: str) -> None:
        # 상태는 READY / MEASURING / DONE 3가지. idle·error 등 나머지는 READY 로 표시.
        text, col = {"measuring": ("MEASURING", AMBER),
                     "done": ("DONE", GREEN)}.get(phase, ("READY", DIM))
        c = self.pill
        c.delete("all")
        h, w = self.spill_h, self.spill_w
        _round_rect(c, 1, 2, w - 1, h - 2, h / 2 - 1, fill=PANEL, outline=STROKE)
        dot = col
        if phase == "measuring" and (self._anim // 12) % 2 == 0:
            dot = "#eccf9c"
        d, cx, cy = h / 3.0, h * 0.667, h / 2.0     # 원본(30px 기준) 비율 유지
        c.create_oval(cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2, fill=dot, outline="")
        c.create_text(h * 1.13, cy, anchor="w", text=text, fill=col,
                      font=self.NF(self.pt_spill, True))

    def _draw_group(self, key: str, name) -> None:
        """LOW / MID / HIGH 배지. name=None 이면 비움."""
        c = self._grp_pill.get(key)
        if c is None:
            return
        c.delete("all")
        if not name:
            return
        col = GROUP_COLORS.get(name, DIM)
        h, w = self.gpill_h, self.gpill_w
        _round_rect(c, 1, 2, w - 1, h - 2, h / 2 - 1, fill=PANEL, outline=col)
        c.create_text(w / 2, h / 2, text=name, fill=col,
                      font=self.NF(self.pt_gpill, True))

    # ── 렌더 루프 ────────────────────────────────────────────────
    def _pump(self) -> None:
        if self.node is not None:
            try:
                rclpy.spin_once(self.node, timeout_sec=0.0)
            except Exception:  # noqa: BLE001
                pass
            self.state = self.node.latest
        self._anim += 1
        self._render(self.state)
        self.root.after(33, self._pump)

    def _pget(self, st: dict, key: str, sub: str):
        """<key>_<sub> 우선, 없으면 구 stiffness 메시지 필드(std/norm_max)로 폴백."""
        v = st.get(f"{key}_{sub}")
        if v is not None:
            return v
        if key == "stiffness":
            return {"std": st.get("std"), "max": st.get("norm_max")}.get(sub)
        return None

    @staticmethod
    def _pval(st: dict, p: dict):
        v = st.get(p["key"])
        if v is None and p.get("alias"):
            v = st.get(p["alias"])
        return v

    @staticmethod
    def _ratio(val, hi):
        if val is None or not hi or hi <= 0:
            return 0.0
        return max(0.0, min(1.0, val / hi))

    @staticmethod
    def _msg_name(st: dict):
        """메시지가 직접 준 샘플 이름 (없으면 None)."""
        for k in ("sample", "sample_id", "name", "id"):
            if st.get(k):
                return str(st[k])
        return None

    def _name_for(self, st: dict) -> str:
        """메시지의 sample/name 우선, 없으면 자동 채번 (sample_1, sample_2 …)."""
        name = self._msg_name(st)
        if name:
            return name
        self._count += 1
        return f"{SAMPLE_PREFIX}_{self._count}"

    def _render(self, st: dict) -> None:
        phase = st.get("phase", "idle")
        self._set_status(phase)

        # ── 새 결과 감지 → 샘플 채번 + 최근 리스트 갱신 + 애니메이션 리셋 ──
        if phase == "done":
            sig = (self._msg_name(st),
                   tuple(round(self._pval(st, p) or -1, 4) for p in PROPS))
            if sig != self._done_sig:
                self._done_sig = sig
                self._sample = self._name_for(st)
                for p in PROPS:
                    self._disp[p["key"]] = 0.0
                self._recent.appendleft(
                    (self._sample, {p["key"]: self._pval(st, p) for p in PROPS}))
            for p in PROPS:                              # 카운트업 애니메이션
                tgt = self._pval(st, p)
                if tgt is not None and self._disp[p["key"]] < tgt:
                    cur = self._disp[p["key"]]
                    self._disp[p["key"]] = min(tgt, cur + (tgt - cur) * 0.22 + tgt * 0.01)
        else:
            self._done_sig = None
            for p in PROPS:
                self._disp[p["key"]] = 0.0

        # ── 왼쪽 물성 값 ──────────────────────────────────────────
        for p in PROPS:
            key = p["key"]
            val = self._pval(st, p)
            std = self._pget(st, key, "std")
            hi = self._pget(st, key, "max") or p["default_max"]
            if phase == "done" and val is not None:
                shown = self._disp[key]
                self._val_lbl[key].config(
                    text=f"{shown:.2f}",
                    fg=grad_text(self._ratio(val, hi)) if p["gradient"] else FG)
                self._unit_lbl[key].config(fg=DIM)
                self._std_lbl[key].config(text=f"± {std:.2f}" if std else "")
                self._draw_group(key, group_of(val))
            else:
                self._val_lbl[key].config(text="--.--",
                                          fg=FAINT if phase == "idle" else DIM)
                self._unit_lbl[key].config(fg=FAINT)
                self._std_lbl[key].config(text="")
                self._draw_group(key, None)

        # ── 오른쪽 샘플명 ─────────────────────────────────────────
        name = (self._sample if phase == "done" and self._sample
                else self._msg_name(st) if phase == "measuring" else None)
        if phase in ("measuring", "done"):
            self.sample_lbl.config(text=(name or SAMPLE_PREFIX).lower(), fg=FG)
        else:
            self.sample_lbl.config(text="—", fg=FAINT)

        # ── 최근 리스트 (강성 높은 순) ─────────────────────────────
        rows = self._recent_sorted()
        sig = tuple((n, v.get(SORT_KEY)) for n, v in rows)
        if sig != self._recent_sig:
            self._recent_sig = sig
            self._draw_recent(rows)

    def _recent_sorted(self) -> list:
        """최근 RECENT_MAX 개를 강성 내림차순으로 정렬 (값 없는 것은 뒤로)."""
        return sorted(self._recent,
                      key=lambda nv: (nv[1].get(SORT_KEY) is not None,
                                      nv[1].get(SORT_KEY) or 0.0),
                      reverse=True)

    def _draw_recent(self, rows: list) -> None:
        for w in self.recent_box.winfo_children():
            w.destroy()
        if not rows:
            tk.Label(self.recent_box, text="아직 측정 결과가 없습니다", bg=BG, fg=FAINT,
                     font=self.NF(self.pt_recent)).pack(anchor="w", pady=4)
            return
        small = max(8, int(self.pt_recent * 0.92))     # rank·그룹 열은 살짝 작게(원본 10 vs 11)
        for rank, (name, vals) in enumerate(rows, 1):
            cur = (name == self._sample)              # 현재 샘플만 강조
            row = tk.Frame(self.recent_box, bg=BG)
            row.pack(fill="x", pady=(0, 3))
            tk.Label(row, text=f"rank{rank}", bg=BG, fg=DIM if cur else FAINT,
                     font=self.NF(small, True), width=7, anchor="w").pack(side="left")
            tk.Label(row, text=name.lower(), bg=BG, fg=FG if cur else DIM,
                     font=self.NF(self.pt_recent, True), width=11,
                     anchor="w").pack(side="left")
            grp = group_of(vals.get(GROUP_KEY)) if GROUP_KEY else None
            tk.Label(row, text=grp or "--", bg=BG,
                     fg=GROUP_COLORS.get(grp, FAINT),
                     font=self.NF(small, True), width=5, anchor="w").pack(side="left")
            parts = []
            for p in PROPS:
                v = vals.get(p["key"])
                parts.append(f"{v:.2f} {p['unit']}" if v is not None else f"-- {p['unit']}")
            tk.Label(row, text=" · ".join(parts), bg=BG,
                     fg=DIM if cur else FAINT, font=self.NF(self.pt_recent),
                     anchor="w", justify="left").pack(side="left")

    # ── 데모 모드 ────────────────────────────────────────────────
    def start_demo(self) -> None:
        self._demo_steps = build_demo_steps()
        self._demo_i = 0
        self._demo_tick()

    def _demo_tick(self) -> None:
        st, dur_ms = self._demo_steps[self._demo_i]
        self.state = st
        self._demo_i = (self._demo_i + 1) % len(self._demo_steps)
        self.root.after(dur_ms, self._demo_tick)


def build_demo_steps():
    """ROS 없이 GUI 를 보기 위한 가짜 상태 시퀀스 (크기/강성/무게)."""
    # (stiffness, s_std, s_max), (weight, w_std), (diameter, d_std)
    plan = [
        ((1.97, 0.31, 6.06),  (68.35, 3.1), (42.03, 1.2)),
        ((3.03, 0.68, 7.63),  (59.02, 2.6), (20.12, 0.9)),
        ((2.01, 0.44, 7.63),  (48.89, 2.2), (18.11, 0.8)),
        ((1.02, 0.55, 7.65),  (58.00, 4.0), (44.00, 2.0)),
        ((7.04, 0.42, 10.14), (112.0, 6.0), (70.00, 2.5)),
    ]
    steps = []
    for s, w, d in plan:
        sval, sstd, smax = s
        common = {"stiffness_max": smax}
        steps.append(({"phase": "measuring", **common}, 1600))
        steps.append(({"phase": "done",
                       "stiffness": sval, "stiffness_std": sstd,
                       "weight": w[0], "weight_std": w[1],
                       "diameter": d[0], "diameter_std": d[1], **common}, 3400))
    return steps


def _reexec_with_system_python() -> None:
    """conda/miniforge python 의 Tk 는 시스템 폰트를 못 봐 폰트가 폴백된다.
       시스템 python(/usr/bin/python3)로 자동 재실행. env.sh 실행이면 no-op."""
    sys_py = "/usr/bin/python3"
    if os.environ.get("PROPERTY_GUI_NO_REEXEC") == "1" or not os.path.exists(sys_py):
        return
    try:
        already = os.path.samefile(sys.executable, sys_py)
    except OSError:
        already = os.path.realpath(sys.executable) == os.path.realpath(sys_py)
    if already:
        return
    env = dict(os.environ, PROPERTY_GUI_NO_REEXEC="1")
    print(f"[GUI] 시스템 폰트 적용을 위해 {sys_py} 로 자동 재실행합니다 (현재: {sys.executable}).", flush=True)
    os.execve(sys_py, [sys_py, os.path.abspath(__file__), *sys.argv[1:]], env)


def main() -> None:
    _reexec_with_system_python()

    import argparse
    ap = argparse.ArgumentParser(description="물성(크기/강성/무게) 추정 GUI")
    ap.add_argument("--demo", action="store_true", help="ROS 없이 데모 데이터로 GUI 만 확인")
    args = ap.parse_args()

    use_ros = _HAVE_ROS and not args.demo
    node = None
    if use_ros:
        rclpy.init()
        node = ResultSub()
    elif not _HAVE_ROS and not args.demo:
        print("[GUI] rclpy 없음 → 데모 모드로 표시합니다 (실데이터는 ROS PC 에서 source env.sh 후).")

    root = tk.Tk()
    gui = PropertyGui(root, node)
    if not use_ros:
        gui.start_demo()

    def on_close():
        try:
            if node is not None:
                node.destroy_node()
        finally:
            if _HAVE_ROS and rclpy.ok():
                rclpy.shutdown()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_close()


if __name__ == "__main__":
    main()