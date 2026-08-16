#!/usr/bin/env python3
"""stiffness_gui.py — 과일 강성 추정 결과 GUI (별도 프로세스).

deploy_task3_ros2 / deploy_ros2 가 발행하는 /stiffness/result (std_msgs/String, JSON) 를
구독해 tkinter 창에 표시한다. 로봇 제어 프로세스와 완전히 분리 → 실시간 루프에 영향 없음.

디자인: 밝은(라이트) 테마, 영문/숫자 위주 UI(폰트 미적용 환경에서도 안 깨짐).
  글씨는 검정 통일, 중요 요소(결과 수치·등급칩·상태·막대·선택)만 색.
  강성 막대: 측정 전 회색 → 결과 시 왼쪽→오른쪽으로 채워지며(온도계) 그라데이션
             (빨강=말랑/soft → 초록=단단/hard) 표시. 양끝에 0 ~ (과일별 최대강성) 눈금.

사진 넣기:  gui/assets/{plum,kiwi,tomato,lemon}  (확장자 무관: .jpg/.jpeg/.png/없어도 됨)
  · PIL(python3-pil) 있으면 JPEG/PNG 모두 + 부드러운 축소. 없으면 PNG/GIF 만.

실행:
  # ROS 있는 PC (실제 데이터)
  source env.sh && python3 .../gui/stiffness_gui.py
  # ROS 없는 PC (레이아웃/사진 확인). 폰트가 제대로 나오려면 반드시 시스템 python:
  /usr/bin/python3 .../gui/stiffness_gui.py --demo
"""
from __future__ import annotations

import base64
import colorsys
import glob
import json
import os
import sys
import tkinter as tk
from io import BytesIO
from tkinter import font as tkfont

try:  # JPEG 지원 (python3-pil). 없으면 PNG/GIF 만.
    from PIL import Image
    _PIL_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
except ImportError:
    Image = None
    _PIL_RESAMPLE = None

try:  # ROS 없는 PC 에서도 GUI 만 확인 가능하게 rclpy 는 선택적.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                           QoSDurabilityPolicy, QoSHistoryPolicy)
    from std_msgs.msg import String
    _HAVE_ROS = True
except ImportError:
    _HAVE_ROS = False
    Node = object  # ResultSub 정의용 더미 (데모 모드에선 인스턴스화 안 함)

RESULT_TOPIC = "/stiffness/result"
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

FRUITS = ["plum", "kiwi", "tomato", "lemon"]
FRUIT_EN = {"plum": "PLUM", "kiwi": "KIWI", "tomato": "TOMATO", "lemon": "LEMON"}
FRUIT_MAX = {"plum": 7.65, "kiwi": 7.63, "tomato": 6.06, "lemon": 10.14}  # 과일별 최대강성(yaml)

# ── 팔레트 (라이트 테마) — 글씨는 검정 통일, 색은 중요 요소만 ─────────────
BG = "#ffffff"
PANEL = "#f4f6f9"
PANEL_HI = "#e9f1fe"
STROKE = "#dde2ea"
TRACK = "#e6eaf0"     # 게이지 빈 트랙(회색)
FG = "#14171c"        # 주 텍스트 (검정)
DIM = "#59636f"       # 보조 텍스트 (회색)
FAINT = "#9aa3b2"     # 3차 텍스트/눈금 (연회색)
ACCENT = "#0a84ff"    # 선택 강조 (파랑)
AMBER = "#c67c0a"     # 측정 중 상태
RED = "#d92d20"       # 오류 상태

MARGIN = 30
CARD_W = 660
THUMB = 96
BIG_W, BIG_H = 380, 240
METER_H = 176


def latched_qos() -> "QoSProfile":
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def pick_ui_font(root: tk.Misc) -> str:
    """디자인/UI 에서 흔한 깔끔한 sans 우선(Ubuntu 등). 목록에 없어도 Xft 가 해석하면 사용.
       ※ conda/miniforge python 의 Tk 는 시스템 폰트를 못 봐 전부 폴백됨 → /usr/bin/python3 권장."""
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
    """frac 0→빨강(말랑), 1→초록(단단). 막대 채움용(선명)."""
    frac = max(0.0, min(1.0, frac))
    r, g, b = colorsys.hsv_to_rgb((1.0 / 3.0) * frac, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def grad_text(frac: float) -> str:
    """흰 배경에서 읽히도록 더 진한 버전 (결과 수치·등급칩·상태 텍스트용)."""
    return grad_hex(frac, s=0.90, v=0.56)


def _round_rect(c: tk.Canvas, x1, y1, x2, y2, r, **kw):
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


def _find_asset(fruit: str):
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(_ASSET_DIR, fruit + ext)
        if os.path.isfile(p):
            return p
    bare = os.path.join(_ASSET_DIR, fruit)
    if os.path.isfile(bare):
        return bare
    for p in sorted(glob.glob(os.path.join(_ASSET_DIR, fruit + ".*"))):
        if os.path.isfile(p):
            return p
    return None


if _HAVE_ROS:
    class ResultSub(Node):
        """/stiffness/result 구독 → 최신 상태 dict 만 보관 (렌더는 tk 쪽에서)."""

        def __init__(self) -> None:
            super().__init__("stiffness_gui")
            self.latest: dict = {"phase": "idle"}
            self.create_subscription(String, RESULT_TOPIC, self._cb, latched_qos())

        def _cb(self, msg) -> None:
            try:
                self.latest = json.loads(msg.data)
            except (ValueError, TypeError):
                pass
else:
    ResultSub = None


class StiffnessGui:
    def __init__(self, root: tk.Tk, node=None) -> None:
        self.root = root
        self.node = node
        self.state: dict = {"phase": "idle"}
        self._imgs: dict = {}
        self.ff = pick_ui_font(root)             # 전체 통일 (깔끔한 UI sans)
        self.nf = self.ff
        if self.ff == "TkDefaultFont":
            print("[GUI] 폰트 미적용(TkDefaultFont) — 시스템 폰트를 못 봤습니다. "
                  "miniforge python 대신 /usr/bin/python3 로 실행하세요.")
        self._cur_fruit = "__none__"
        self._anim = 0
        self._meter_sig = None
        self._fill_cur = 0.0
        self._done_rsig = None
        self._fnt_val = tkfont.Font(root=root, family=self.nf, size=34, weight="bold")
        self._fnt_rank = tkfont.Font(root=root, family=self.nf, size=11)
        self._fnt_rankv = tkfont.Font(root=root, family=self.nf, size=12)

        root.title("Fruit Stiffness Estimator")
        root.configure(bg=BG)
        root.minsize(CARD_W + 2 * MARGIN, 720)

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=MARGIN, pady=(20, 24))
        self._build_header(outer)
        self._build_top(outer)
        self._build_meter(outer)
        self._build_rank(outer)
        self._build_metrics(outer)

        self._render(self.state)
        root.after(33, self._pump)

    def F(self, size: int, bold: bool = False):
        return (self.ff, size, "bold" if bold else "normal")

    def NF(self, size: int, bold: bool = False):
        return (self.nf, size, "bold" if bold else "normal")

    # ── 헤더 (타이틀 + 상태 pill) ─────────────────────────────────
    def _build_header(self, parent) -> None:
        head = tk.Frame(parent, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="FRUIT  STIFFNESS  ESTIMATOR", bg=BG, fg=FG,
                 font=self.NF(19, True)).pack(side="left", pady=(4, 0))
        self.pill = tk.Canvas(head, width=168, height=34, bg=BG, highlightthickness=0)
        self.pill.pack(side="right", pady=6)
        tk.Frame(parent, bg=STROKE, height=1).pack(fill="x", pady=(14, 16))

    def _set_status(self, phase: str, st: dict) -> None:
        if phase == "measuring":
            text, col = "MEASURING", AMBER
        elif phase == "done":
            text, col = "DONE", grad_text(self._ratio(st) or 0.0)
        elif phase == "error":
            text, col = "ERROR", RED
        else:
            text, col = "IDLE", DIM
        c = self.pill
        c.delete("all")
        _round_rect(c, 1, 4, 167, 30, 13, fill=PANEL, outline=STROKE)
        dot = col
        if phase == "measuring" and (self._anim // 12) % 2 == 0:
            dot = "#eccf9c"
        c.create_oval(16, 12, 26, 22, fill=dot, outline="")
        c.create_text(36, 17, anchor="w", text=text, fill=col, font=self.NF(11, True))

    # ── 상단: 왼쪽 2×2 사진(붙여서=테이블) + 오른쪽 과일별 최대강성 표 ──
    def _build_top(self, parent) -> None:
        top = tk.Frame(parent, bg=BG)
        top.pack(fill="x", pady=(0, 4))

        # 1×4 사진 (가운데 정렬) + 각 아래 과일명. 측정 중인 과일은 파란색.
        grid = tk.Frame(top, bg=BG)
        grid.pack()                              # 중앙 정렬
        self._cells, self._cell_name = {}, {}
        for i, fr in enumerate(FRUITS):
            cell = tk.Frame(grid, bg=BG)
            cell.grid(row=0, column=i, padx=6, pady=0)
            cv = tk.Canvas(cell, width=THUMB, height=THUMB, bg=PANEL,
                           highlightthickness=2, highlightbackground=STROKE)
            cv.pack()
            img = self._get_thumb_image(fr)
            if img is not None:
                cv.create_image(THUMB // 2, THUMB // 2, image=img)
            else:
                cv.create_text(THUMB // 2, THUMB // 2, text="—", fill=FAINT, font=self.F(14))
            nm = tk.Label(cell, text=FRUIT_EN[fr], bg=BG, fg=DIM, font=self.F(9, True))
            nm.pack(pady=(3, 0))
            self._cells[fr] = cv
            self._cell_name[fr] = nm
        # MAX 표는 순위 리스트에 통합됨 (아래 _draw_rank). 행 참조용 dict만 빈 채로 유지.
        self._row_name, self._row_val = {}, {}

    # ── 강성 게이지 ──────────────────────────────────────────────
    def _build_meter(self, parent) -> None:
        self.canvas = tk.Canvas(parent, width=CARD_W, height=METER_H,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(pady=(10, 0))

    # ── 강성 순위 리스트 (단단 → 무름) ───────────────────────────
    def _build_rank(self, parent) -> None:
        tk.Frame(parent, bg=STROKE, height=1).pack(fill="x", pady=(6, 12))
        head = tk.Frame(parent, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="STIFFNESS  RANKING", bg=BG, fg=DIM,
                 font=self.F(10, True)).pack(side="left")
        tk.Label(head, text="측정값 ±std  / 기준max        hard \u2192 soft",
                 bg=BG, fg=FAINT, font=self.NF(9)).pack(side="right")
        # 순위 막대를 그릴 캔버스 (측정된 과일들이 누적)
        self.rank_canvas = tk.Canvas(parent, width=CARD_W, height=4 * 26 + 6,
                                     bg=BG, highlightthickness=0)
        self.rank_canvas.pack(pady=(8, 0))
        self._measured: dict = {}     # fruit -> (stiffness, std)

    def _draw_rank(self) -> None:
        c = self.rank_canvas
        c.delete("all")
        if not self._measured:
            c.create_text(CARD_W // 2, 20, text="측정을 시작하면 순위가 표시됩니다",
                          fill=FAINT, font=self.NF(11))
            return
        # 강성 내림차순 정렬 (단단 → 무름)
        items = sorted(self._measured.items(), key=lambda kv: -kv[1][0])
        gmax = max((FRUIT_MAX.get(f, 10.14) for f in self._measured), default=10.14)
        row_h, bx0, bx1 = 26, 92, CARD_W - 150
        for i, (fr, (val, std)) in enumerate(items):
            y = i * row_h + 12
            cur = (fr == self._cur_fruit)
            col = grad_text(min(1.0, val / (FRUIT_MAX.get(fr, 10.14) or 10.14)))
            # 과일명
            c.create_text(bx0 - 8, y, anchor="e", text=FRUIT_EN.get(fr, fr),
                          fill=(FG if cur else DIM),
                          font=self.F(11, cur))
            # 트랙
            c.create_line(bx0, y, bx1, y, fill=TRACK, width=6, capstyle="round")
            frac = max(0.0, min(1.0, val / gmax))
            fx = bx0 + (bx1 - bx0) * frac
            # std 밴드
            if std and std > 0:
                sfr = min(0.4, std / gmax)
                s_lo = bx0 + (bx1 - bx0) * max(0.0, frac - sfr)
                s_hi = bx0 + (bx1 - bx0) * min(1.0, frac + sfr)
                c.create_line(s_lo, y, s_hi, y, fill=STROKE, width=10, capstyle="round")
            # 채움 막대
            c.create_line(bx0, y, fx, y, fill=col, width=6, capstyle="round")
            c.create_oval(fx - 4, y - 4, fx + 4, y + 4, fill=col, outline=BG)
            # 측정값 ± std
            txt = f"{val:.2f}"
            c.create_text(bx1 + 12, y, anchor="w", text=txt,
                          fill=(FG if cur else DIM), font=self.NF(12, cur))
            if std is not None:
                c.create_text(bx1 + 12 + self._fnt_rankv.measure(txt) + 4, y, anchor="w",
                              text=f"\u00b1{std:.2f}", fill=FAINT, font=self.NF(9))
            # 기준 max (MAX 표 흡수) — 오른쪽 끝
            c.create_text(CARD_W - 6, y, anchor="e",
                          text=f"/ {FRUIT_MAX.get(fr, 0):.2f}",
                          fill=FAINT, font=self.NF(10))

    # ── 성능 지표 카드 ───────────────────────────────────────────
    def _build_metrics(self, parent) -> None:
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="x", pady=(16, 0))
        self._metric_val = {}
        # 실시간 계산 가능한 지표만 (정답 라벨 불필요).
        specs = [("MEASURED", "count", FG),
                 ("AVG \u00b1STD", "avgstd", FG),
                 ("RANK CONFIDENCE", "rankconf", FG)]
        for i, (label, key, col) in enumerate(specs):
            cell = tk.Frame(wrap, bg=PANEL)
            cell.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            wrap.grid_columnconfigure(i, weight=1)
            tk.Label(cell, text=label, bg=PANEL, fg=DIM,
                     font=self.F(9, True)).pack(anchor="w", padx=12, pady=(10, 0))
            v = tk.Label(cell, text="--", bg=PANEL, fg=col, font=self.NF(20, True))
            v.pack(anchor="w", padx=12, pady=(0, 10))
            self._metric_val[key] = v

    def _rank_confidence(self) -> "tuple":
        """순위 신뢰도 — 정답 없이 계산.
           인접 순위 과일 쌍의 강성차가 그들의 std(불확실성) 대비 충분히 크면
           '순서가 확실'. 신뢰구간이 겹치면 불확실.
           반환: (확실히 갈리는 인접쌍 수, 전체 인접쌍 수)."""
        items = sorted(self._measured.items(), key=lambda kv: -kv[1][0])
        if len(items) < 2:
            return (0, 0)
        clear = 0
        pairs = 0
        for a, b in zip(items[:-1], items[1:]):
            (va, sa), (vb, sb) = a[1], b[1]
            pairs += 1
            # 두 신뢰구간(±std)이 안 겹치면 순서 확실
            #   va - sa (위 과일 하한) > vb + sb (아래 과일 상한)
            if (va - sa) > (vb + sb):
                clear += 1
        return (clear, pairs)

    def _update_metrics(self, st: dict) -> None:
        n = len(self._measured)
        self._metric_val["count"].config(text=str(n))
        if self._measured:
            avg_std = sum(s for _, s in self._measured.values()) / n
            self._metric_val["avgstd"].config(text=f"\u00b1{avg_std:.2f}")
        # 순위 신뢰도: 인접 순위가 신뢰구간으로 확실히 갈리는 비율 (정답 불필요)
        clear, pairs = self._rank_confidence()
        if pairs > 0:
            pct = clear / pairs * 100
            col = "#1a8f3c" if pct >= 80 else (AMBER if pct >= 50 else DIM)
            self._metric_val["rankconf"].config(text=f"{pct:.0f}%", fg=col)
        else:
            self._metric_val["rankconf"].config(text="--", fg=FG)

    # ── 이미지 로드 ──────────────────────────────────────────────
    def _get_thumb_image(self, fruit: str):
        """썸네일: 중앙을 정사각으로 크롭 후 THUMB×THUMB 로 → 모든 과일 동일 크기."""
        key = (fruit, "thumb")
        if key in self._imgs:
            return self._imgs[key]
        path = _find_asset(fruit)
        img = None
        if path:
            try:
                if Image is not None:
                    pim = Image.open(path)
                    pim.load()
                    w, h = pim.size
                    s = min(w, h)
                    l, t = (w - s) // 2, (h - s) // 2
                    pim = pim.crop((l, t, l + s, t + s)).convert("RGB").resize(
                        (THUMB, THUMB), _PIL_RESAMPLE)
                    buf = BytesIO()
                    pim.save(buf, "PNG")
                    img = tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode("ascii"))
                else:
                    raw = tk.PhotoImage(file=path)
                    factor = max(1, round(max(raw.width(), raw.height()) / THUMB))
                    img = raw.subsample(factor) if factor > 1 else raw
            except (tk.TclError, OSError):
                img = None
        self._imgs[key] = img
        return img

    def _get_big_image(self, fruit: str):
        """큰 사진: 모든 과일을 동일 높이(BIG_H)로 리사이즈(폭은 비율, BIG_W 초과 시 폭 기준)."""
        key = (fruit, "big")
        if key in self._imgs:
            return self._imgs[key]
        path = _find_asset(fruit)
        img = None
        if path:
            try:
                if Image is not None:
                    pim = Image.open(path)
                    pim.load()
                    w, h = pim.size
                    scale = BIG_H / h
                    if w * scale > BIG_W:
                        scale = BIG_W / w
                    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
                    pim = pim.convert("RGB").resize((nw, nh), _PIL_RESAMPLE)
                    buf = BytesIO()
                    pim.save(buf, "PNG")
                    img = tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode("ascii"))
                else:
                    raw = tk.PhotoImage(file=path)
                    factor = max(1, round(raw.height() / BIG_H))
                    img = raw.subsample(factor) if factor > 1 else raw
            except (tk.TclError, OSError):
                img = None
        self._imgs[key] = img
        return img

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

    def _render(self, st: dict) -> None:
        phase = st.get("phase", "idle")
        fruit = st.get("fruit")
        self._set_status(phase, st)

        if fruit != self._cur_fruit:
            self._cur_fruit = fruit
            self._update_selection(fruit)
            if self._measured:
                self._draw_rank()

        # 막대 채움 애니메이션 (done 일 때 왼쪽→오른쪽으로 상승)
        if phase == "done":
            rsig = (fruit, round(st.get("stiffness", 0), 4))
            if rsig != self._done_rsig:
                self._done_rsig = rsig
                self._fill_cur = 0.0
                # 측정 완료된 과일을 순위에 기록 (강성, std)
                if fruit and st.get("stiffness") is not None:
                    self._measured[fruit] = (st.get("stiffness"),
                                             st.get("std") or 0.0)
                    self._draw_rank()
                    self._update_metrics(st)
            target = self._ratio(st) or 0.0
            if self._fill_cur < target:
                self._fill_cur = min(target, self._fill_cur
                                     + (target - self._fill_cur) * 0.16 + 0.006)
        else:
            self._done_rsig = None
            self._fill_cur = 0.0

        sig = (phase, fruit, tuple(st.get("boundaries", []) or []),
               st.get("norm_min"), st.get("norm_max"),
               round(self._fill_cur, 3), round(st.get("stiffness", -1), 4))
        if sig != self._meter_sig:
            self._meter_sig = sig
            self._draw_meter(st, phase)

    def _update_selection(self, fruit) -> None:
        for fr in FRUITS:
            on = (fr == fruit)
            col = ACCENT if on else DIM        # 측정 중인 과일 = 파란색
            self._cells[fr].config(highlightbackground=(ACCENT if on else STROKE))
            self._cell_name[fr].config(fg=col)   # 사진 아래 이름

    def _disp_max(self, st: dict):
        """표시용 최대강성: 상단 표와 동일하게 과일별 고정값(FRUIT_MAX). 없으면 메시지 norm_max."""
        return FRUIT_MAX.get(st.get("fruit"), st.get("norm_max"))

    def _ratio(self, st: dict):
        hi = self._disp_max(st)
        s = st.get("stiffness")
        if s is None or not hi or hi <= 0:
            return None
        return max(0.0, min(1.0, s / hi))     # min=0 기준 (0 ~ 과일 max)

    def _track(self):
        return 2, CARD_W - 2, 92, 120     # x0, x1, y0, y1

    def _draw_meter(self, st: dict, phase: str) -> None:
        c = self.canvas
        c.delete("all")
        x0, x1, ty0, ty1 = self._track()
        W = x1 - x0
        lo, hi = 0.0, self._disp_max(st)      # 표시 스케일 0 ~ 과일 max (상단 표와 일치)

        # 1) 회색 빈 트랙 (측정 전 기본)
        c.create_rectangle(x0, ty0, x1, ty1, fill=TRACK, outline=STROKE)

        # 2) 그라데이션 채움 (done: 왼쪽→오른쪽으로 fill_cur 까지, 온도계)
        if phase == "done" and self._fill_cur > 0:
            fillw = max(0.0, min(1.0, self._fill_cur)) * W
            step = 3
            i = 0.0
            while i < fillw:
                col = grad_hex(i / (W - 1))
                c.create_rectangle(x0 + i, ty0, x0 + min(i + step, fillw), ty1,
                                   fill=col, outline=col)
                i += step

        # 2-b) 신뢰구간(±std) 밴드 — 추정 위치 주변 옅은 박스 (이분산 σ 시각화)
        if phase == "done" and self._fill_cur > 0 and hi and hi > 0:
            std = st.get("std")
            if std is not None and std > 0:
                sfrac = min(0.5, std / hi)              # std 를 스케일 비율로
                cxr = max(0.0, min(1.0, self._fill_cur))
                bx_lo = x0 + max(0.0, cxr - sfrac) * W
                bx_hi = x0 + min(1.0, cxr + sfrac) * W
                c.create_rectangle(bx_lo, ty0 - 4, bx_hi, ty1 + 4,
                                   fill="", outline=DIM, width=1, dash=(3, 2))

        # 3) 눈금 (10%)
        for k in range(0, 11):
            tx = x0 + W * k / 10
            h = 8 if k % 5 == 0 else 5
            c.create_line(tx, ty1 + 3, tx, ty1 + 3 + h, fill=FAINT)

        # 4) 등급 구간 경계(흰 노치) + 이름(SOFT/MID/HARD, 회색)
        bounds = st.get("boundaries", []) or []
        names = st.get("class_names", []) or []
        if lo is not None and hi is not None and hi > lo and bounds:
            edges = [0.0] + [(b - lo) / (hi - lo) for b in bounds] + [1.0]
            for br in edges[1:-1]:
                bx = x0 + br * W
                c.create_line(bx, ty0, bx, ty1, fill=BG, width=2)
            for i, nm in enumerate(names):
                if i + 1 < len(edges):
                    cx = x0 + (edges[i] + edges[i + 1]) / 2 * W
                    c.create_text(cx, ty0 - 15, text=str(nm).upper(),
                                  fill=DIM, font=self.NF(10, True))

        # 5) 양끝 눈금값: 0 (왼) ~ 과일별 최대강성 (오, 상단 표와 동일한 값·소수점)
        if hi is not None and hi > 0:
            c.create_text(x0, ty1 + 26, anchor="w", text="0", fill=DIM, font=self.NF(11, True))
            c.create_text(x1, ty1 + 26, anchor="e", text=f"{hi:.2f}",
                          fill=DIM, font=self.NF(11, True))

        # 6) 수치 + 등급칩 (수치·칩만 색)
        if phase == "done":
            val, vcol = f"{st.get('stiffness', 0):.2f}", grad_text(self._ratio(st) or 0.0)
        elif phase == "measuring":
            val, vcol = "--.--", DIM
        else:
            val, vcol = "--.--", FAINT
        c.create_text(x0 + 2, 30, anchor="w", text=val, fill=vcol, font=self.NF(34, True))

        # 6-b) ± std (신뢰구간 수치) — 값 오른쪽에 작게
        if phase == "done":
            std = st.get("std")
            if std is not None:
                sx = x0 + 2 + self._fnt_val.measure(val) + 8
                c.create_text(sx, 34, anchor="w", text=f"\u00b1 {std:.2f}",
                              fill=DIM, font=self.NF(14, True))

        if phase == "done":
            gcol = grad_text(self._ratio(st) or 0.0)
            chip = f" {str(st.get('cname', '')).upper()} "
            cx = x0 + 2 + self._fnt_val.measure(val) + 16
            std = st.get("std")
            if std is not None:      # std 표시 폭만큼 칩을 오른쪽으로
                _sf = tkfont.Font(root=self.root, family=self.nf, size=14, weight="bold")
                cx += _sf.measure(f"\u00b1 {std:.2f}") + 12
            tw = tkfont.Font(root=self.root, family=self.ff, size=13,
                             weight="bold").measure(chip)
            _round_rect(c, cx, 13, cx + tw + 14, 45, 9, fill="", outline=gcol)
            c.create_text(cx + 7 + tw / 2, 29, text=chip, fill=gcol, font=self.NF(13, True))

        # 7) 채움 끝 팁 마커 (검정)
        if phase == "done" and self._fill_cur > 0:
            mx = x0 + max(0.0, min(1.0, self._fill_cur)) * W
            c.create_line(mx, ty0 - 8, mx, ty1 + 8, fill=FG, width=2)
            c.create_polygon(mx, ty0 - 16, mx + 7, ty0 - 8, mx, ty0, mx - 7, ty0 - 8,
                             fill=FG, outline=BG)

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
    """ROS 없이 GUI 를 보기 위한 가짜 상태 시퀀스 (실제 yaml 범위/경계와 동일)."""
    cn = ["soft", "mid", "hard"]
    ranges = {
        "plum": (0.0, 7.65, [1.761, 3.84]),
        "kiwi": (0.0, 7.63, [2.07, 3.672]),
        "tomato": (0.0, 6.06, [1.905, 3.075]),
        "lemon": (0.0, 10.14, [3.416, 4.894]),
    }
    plan = [("plum", 1.0, 0.55), ("kiwi", 3.0, 0.68),
            ("tomato", 4.5, 0.31), ("lemon", 7.0, 0.42)]
    steps = []
    for fr, s, std in plan:
        lo, hi, b = ranges[fr]
        cls = sum(1 for x in b if s >= x)
        common = {"fruit": fr, "norm_min": lo, "norm_max": hi,
                  "boundaries": b, "class_names": cn}
        steps.append(({"phase": "measuring", **common}, 2200))
        steps.append(({"phase": "done", "stiffness": s, "cls": cls,
                       "cname": cn[cls], "std": std, **common}, 3600))
    return steps


def _reexec_with_system_python() -> None:
    """conda/miniforge python 의 Tk 는 시스템 폰트를 못 봐 디자인(폰트)이 TkDefaultFont 로
       폴백된다. 시스템 python(/usr/bin/python3)로 자동 재실행해 의도한 폰트/디자인이 그대로
       나오게 한다. 이미 /usr/bin/python3 이거나(=env.sh 실행) 재실행 방지 플래그가 있으면 건너뜀.
       (deploy 의 spawn_gui 는 env.sh 상 이미 /usr/bin/python3 → no-op.)"""
    sys_py = "/usr/bin/python3"
    if os.environ.get("STIFFNESS_GUI_NO_REEXEC") == "1" or not os.path.exists(sys_py):
        return
    try:
        already = os.path.samefile(sys.executable, sys_py)
    except OSError:
        already = os.path.realpath(sys.executable) == os.path.realpath(sys_py)
    if already:
        return
    env = dict(os.environ, STIFFNESS_GUI_NO_REEXEC="1")
    print(f"[GUI] 시스템 폰트 적용을 위해 {sys_py} 로 자동 재실행합니다 (현재: {sys.executable}).",
          flush=True)   # execve 는 Python 버퍼를 안 비우므로 flush 필수
    os.execve(sys_py, [sys_py, os.path.abspath(__file__), *sys.argv[1:]], env)


def main() -> None:
    _reexec_with_system_python()   # conda python 이면 /usr/bin/python3 로 재실행(폰트/디자인 보장)

    import argparse
    ap = argparse.ArgumentParser(description="과일 강성 추정 GUI")
    ap.add_argument("--demo", action="store_true", help="ROS 없이 데모 데이터로 GUI 만 확인")
    args = ap.parse_args()

    use_ros = _HAVE_ROS and not args.demo
    node = None
    if use_ros:
        rclpy.init()
        node = ResultSub()
    elif not _HAVE_ROS and not args.demo:
        print("[GUI] rclpy 없음 → 데모 모드로 표시합니다 "
              "(실제 데이터는 ROS 있는 PC 에서 'source env.sh' 후 실행).")

    root = tk.Tk()
    gui = StiffnessGui(root, node)
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