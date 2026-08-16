#!/usr/bin/env python3
"""property_gui.py — 과일 물성(크기/강성/무게) 추정 결과 GUI (별도 프로세스).

레이아웃 (스케치 반영):
  ┌──────────────────────────┬────────────────────┐
  │ SIZE       42.03 mm      │ [사진]   tomato_1  │
  ├──────────────────────────┤          ● DONE    │
  │ STIFFNESS  1.97 N/mm     ├────────────────────┤
  ├──────────────────────────┤ RECENT             │
  │ WEIGHT     68.35 g       │  tomato_1: 42.03…  │
  │                          │  kiwi_2:   20.12…  │
  └──────────────────────────┴────────────────────┘
  · 왼쪽: 물성 3개를 큰 숫자로 표시 (SIZE / STIFFNESS / WEIGHT)
  · 오른쪽 상단: 과일 사진 + 샘플 이름 + 상태 표시
  · 오른쪽 하단: 최근 측정 10개 리스트 (최신이 위)
  · 상태 표시는 READY / MEASURING / DONE 3가지

deploy 측이 발행하는 /property/result (std_msgs/String, JSON) 를 구독해 tkinter 창에 표시.
로봇 제어 프로세스와 완전히 분리 → 실시간 루프에 영향 없음.

메시지 JSON (예):
  {"phase":"done", "fruit":"tomato", "sample":"tomato_1",
   "stiffness":1.97, "stiffness_std":0.3, "stiffness_max":6.06,
     "cname":"mid", "boundaries":[1.9,3.1], "class_names":["soft","mid","hard"],
   "weight":68.35, "weight_std":6.0,
   "diameter":42.03, "diameter_std":2.0,
   "image":"/abs/path/tomato_1.jpg"}
  phase: idle | measuring | done | error.  (구 stiffness 메시지의 std/norm_max/cname 도 호환)
  · sample 없으면 과일별로 tomato_1, tomato_2 … 자동 채번.
  · image 없으면 assets/<fruit>.png|jpg 를 찾고, 그것도 없으면 플레이스홀더 표시.
  · diameter 대신 size 필드로 보내도 동작.

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

try:  # 사진 리사이즈/JPEG 지원 (없으면 Tk 기본 PhotoImage: PNG/GIF 만)
    from PIL import Image, ImageTk
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False

RESULT_TOPIC = "/property/result"

FRUIT_EN = {"plum": "plum", "kiwi": "kiwi", "tomato": "tomato", "lemon": "lemon"}
FRUIT_STIFF_MAX = {"plum": 7.65, "kiwi": 7.63, "tomato": 6.06, "lemon": 10.14}

# ── 표시할 물성 3개 (스케치 순서: size → stiffness → weight) ──────
#   key        : 메시지의 값 필드 (std=<key>_std, max=<key>_max)
#   alias      : 대체 필드명
#   gradient   : True 면 값 글자에 빨강(작음)→초록(큼) 그라데이션. 기본은 전부 검정.
#   show_grade : True 면 행 오른쪽에 등급(SOFT/MID/HARD) 표시
PROPS = [
    {"key": "diameter",  "alias": "size",  "label": "SIZE",      "unit": "mm",
     "gradient": False, "show_grade": False, "default_max": 120.0},
    {"key": "stiffness", "alias": None,    "label": "STIFFNESS", "unit": "N/mm",
     "gradient": False, "show_grade": True,  "default_max": 10.0},
    {"key": "weight",    "alias": None,    "label": "WEIGHT",    "unit": "g",
     "gradient": False, "show_grade": False, "default_max": 300.0},
]

RECENT_MAX = 10

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
PHOTO_PX = 170


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


def _round_rect(c: tk.Canvas, x1, y1, x2, y2, r, **kw):
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


def _asset_dirs() -> list:
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = [os.environ.get("PROPERTY_GUI_ASSETS"),
            os.path.join(here, "assets"),
            os.path.join(here, os.pardir, "assets"),
            os.path.join(os.getcwd(), "assets")]
    return [d for d in dirs if d and os.path.isdir(d)]


def find_image(st: dict, fruit, sample) -> str | None:
    """메시지의 image 경로 → assets/<sample>.* → assets/<fruit>.* 순으로 탐색."""
    p = st.get("image") or st.get("photo")
    if p and os.path.isfile(p):
        return p
    stems = [s for s in (sample, fruit) if s]
    for d in _asset_dirs():
        for stem in stems:
            for ext in (".png", ".jpg", ".jpeg", ".gif", ".ppm"):
                cand = os.path.join(d, f"{stem}{ext}")
                if os.path.isfile(cand):
                    return cand
    return None


def load_photo(path: str, box: int):
    """box×box 안에 들어가도록 축소한 PhotoImage 반환 (실패 시 None)."""
    try:
        if _HAVE_PIL:
            im = Image.open(path)
            im.thumbnail((box, box), Image.LANCZOS)
            return ImageTk.PhotoImage(im)
        img = tk.PhotoImage(file=path)          # PNG/GIF/PPM 만 가능
        k = max(1, max(img.width(), img.height()) // box + 1)
        return img.subsample(k, k) if k > 1 else img
    except Exception:  # noqa: BLE001
        return None


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
        self._counts: dict = {}                              # 과일별 자동 채번
        self._sample = None                                  # 현재 샘플 이름
        self._recent = deque(maxlen=RECENT_MAX)              # [(name, {key:val}), …] 최신이 앞
        self._photo_ref = None                               # GC 방지용 참조
        self._photo_path = None
        self._recent_sig = None

        root.title("Fruit Property Estimator")
        root.configure(bg=BG)
        root.geometry("1020x640")
        root.minsize(900, 600)

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=MARGIN, pady=MARGIN)
        outer.columnconfigure(0, weight=5, uniform="col")     # 왼쪽 물성부
        outer.columnconfigure(1, weight=0)                    # 세로 구분선
        outer.columnconfigure(2, weight=3, uniform="col")     # 오른쪽 사진/리스트
        outer.rowconfigure(0, weight=1)

        left = tk.Frame(outer, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        tk.Frame(outer, bg=STROKE, width=1).grid(row=0, column=1, sticky="ns")
        right = tk.Frame(outer, bg=BG)
        right.grid(row=0, column=2, sticky="nsew", padx=(18, 0))

        self._build_props(left)
        self._build_right(right)

        self._render(self.state)
        root.after(33, self._pump)

    def NF(self, size: int, bold: bool = False):
        return (self.nf, size, "bold" if bold else "normal")

    # ── 왼쪽: 물성 3행 ───────────────────────────────────────────
    def _build_props(self, parent) -> None:
        self._val_lbl, self._unit_lbl, self._std_lbl, self._grade_lbl = {}, {}, {}, {}
        for i, p in enumerate(PROPS):
            r = i * 2
            parent.rowconfigure(r, weight=1)
            cell = tk.Frame(parent, bg=BG)
            cell.grid(row=r, column=0, sticky="nsew")
            cell.columnconfigure(0, weight=1)

            head = tk.Frame(cell, bg=BG)
            head.pack(fill="x", anchor="w", pady=(10, 0))
            tk.Label(head, text=p["label"], bg=BG, fg=DIM,
                     font=self.NF(12, True)).pack(side="left")
            self._grade_lbl[p["key"]] = tk.Label(head, text="", bg=BG, fg=DIM,
                                                 font=self.NF(11, True))
            self._grade_lbl[p["key"]].pack(side="right")

            body = tk.Frame(cell, bg=BG)
            body.pack(fill="x", anchor="w", pady=(2, 10), padx=(26, 0))
            self._val_lbl[p["key"]] = tk.Label(body, text="--.--", bg=BG, fg=FAINT,
                                               font=self.NF(38, True))
            self._val_lbl[p["key"]].pack(side="left", anchor="s")
            self._unit_lbl[p["key"]] = tk.Label(body, text=p["unit"], bg=BG, fg=DIM,
                                                font=self.NF(19))
            self._unit_lbl[p["key"]].pack(side="left", anchor="s", padx=(9, 0), pady=(0, 6))
            self._std_lbl[p["key"]] = tk.Label(body, text="", bg=BG, fg=FAINT,
                                               font=self.NF(12, True))
            self._std_lbl[p["key"]].pack(side="left", anchor="s", padx=(12, 0), pady=(0, 8))

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

        self.photo = tk.Canvas(top, width=PHOTO_PX, height=PHOTO_PX,
                               bg=BG, highlightthickness=0)
        self.photo.pack(side="left")

        meta = tk.Frame(top, bg=BG)
        meta.pack(side="left", fill="both", expand=True, padx=(14, 0))
        self.sample_lbl = tk.Label(meta, text="—", bg=BG, fg=FG,
                                   font=self.NF(20, True), anchor="w", justify="left")
        self.sample_lbl.pack(anchor="w", pady=(6, 8))
        self.pill = tk.Canvas(meta, width=150, height=30, bg=BG, highlightthickness=0)
        self.pill.pack(anchor="w")

        tk.Frame(parent, bg=STROKE, height=1).grid(row=1, column=0, sticky="ew")

        lst = tk.Frame(parent, bg=BG)
        lst.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        tk.Label(lst, text=f"RECENT {RECENT_MAX}", bg=BG, fg=DIM,
                 font=self.NF(11, True)).pack(anchor="w", pady=(0, 6))
        self.recent_box = tk.Frame(lst, bg=BG)
        self.recent_box.pack(fill="both", expand=True)

        self._draw_photo(None)

    def _set_status(self, phase: str) -> None:
        # 상태는 READY / MEASURING / DONE 3가지. idle·error 등 나머지는 READY 로 표시.
        text, col = {"measuring": ("MEASURING", AMBER),
                     "done": ("DONE", GREEN)}.get(phase, ("READY", DIM))
        c = self.pill
        c.delete("all")
        _round_rect(c, 1, 2, 149, 28, 13, fill=PANEL, outline=STROKE)
        dot = col
        if phase == "measuring" and (self._anim // 12) % 2 == 0:
            dot = "#eccf9c"
        c.create_oval(15, 10, 25, 20, fill=dot, outline="")
        c.create_text(34, 15, anchor="w", text=text, fill=col, font=self.NF(11, True))

    def _draw_photo(self, path) -> None:
        """사진 영역 갱신. path=None 이면 플레이스홀더."""
        if path == self._photo_path:
            return
        self._photo_path = path
        c = self.photo
        c.delete("all")
        img = load_photo(path, PHOTO_PX - 8) if path else None
        _round_rect(c, 1, 1, PHOTO_PX - 1, PHOTO_PX - 1, 12,
                    fill=PANEL if img is None else BG, outline=STROKE)
        if img is not None:
            self._photo_ref = img                       # GC 방지
            c.create_image(PHOTO_PX / 2, PHOTO_PX / 2, image=img)
            _round_rect(c, 1, 1, PHOTO_PX - 1, PHOTO_PX - 1, 12, fill="", outline=STROKE)
        else:
            self._photo_ref = None
            c.create_text(PHOTO_PX / 2, PHOTO_PX / 2 - 8, text="NO IMAGE",
                          fill=FAINT, font=self.NF(11, True))
            c.create_text(PHOTO_PX / 2, PHOTO_PX / 2 + 12, text="assets/<fruit>.png",
                          fill=FAINT, font=self.NF(9))

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
        """<key>_<sub> 우선, 없으면 구 stiffness 메시지 필드(std/norm_max/cname)로 폴백."""
        v = st.get(f"{key}_{sub}")
        if v is not None:
            return v
        if key == "stiffness":
            return {"std": st.get("std"), "max": st.get("norm_max"),
                    "grade": st.get("cname")}.get(sub)
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

    def _name_for(self, st: dict, fruit) -> str:
        """메시지의 sample/name 우선, 없으면 과일별 자동 채번 (tomato_1, tomato_2 …)."""
        for k in ("sample", "sample_id", "name", "id"):
            if st.get(k):
                return str(st[k])
        fr = fruit or "sample"
        self._counts[fr] = self._counts.get(fr, 0) + 1
        return f"{fr}_{self._counts[fr]}"

    def _render(self, st: dict) -> None:
        phase = st.get("phase", "idle")
        fruit = st.get("fruit")
        self._set_status(phase)

        # ── 새 결과 감지 → 샘플 채번 + 최근 리스트 갱신 + 애니메이션 리셋 ──
        if phase == "done":
            sig = (fruit, tuple(round(self._pval(st, p) or -1, 4) for p in PROPS))
            if sig != self._done_sig:
                self._done_sig = sig
                self._sample = self._name_for(st, fruit)
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
            hi = (self._pget(st, key, "max")
                  or (FRUIT_STIFF_MAX.get(fruit) if key == "stiffness" else None)
                  or p["default_max"])
            if phase == "done" and val is not None:
                shown = self._disp[key]
                self._val_lbl[key].config(
                    text=f"{shown:.2f}",
                    fg=grad_text(self._ratio(val, hi)) if p["gradient"] else FG)
                self._unit_lbl[key].config(fg=DIM)
                self._std_lbl[key].config(text=f"± {std:.2f}" if std else "")
                grade = self._pget(st, key, "grade") if p["show_grade"] else None
                self._grade_lbl[key].config(text=str(grade).upper() if grade else "",
                                            fg=DIM)
            else:
                self._val_lbl[key].config(text="--.--",
                                          fg=FAINT if phase == "idle" else DIM)
                self._unit_lbl[key].config(fg=FAINT)
                self._std_lbl[key].config(text="")
                self._grade_lbl[key].config(text="")

        # ── 오른쪽 사진 / 샘플명 ──────────────────────────────────
        if phase in ("measuring", "done") and fruit:
            name = self._sample if phase == "done" and self._sample else FRUIT_EN.get(
                fruit, str(fruit).upper())
            self.sample_lbl.config(text=name, fg=FG)
            self._draw_photo(find_image(st, fruit, self._sample if phase == "done" else None))
        else:
            self.sample_lbl.config(text="—", fg=FAINT)
            self._draw_photo(None)

        # ── 최근 5개 리스트 ───────────────────────────────────────
        sig = tuple(n for n, _ in self._recent)
        if sig != self._recent_sig:
            self._recent_sig = sig
            self._draw_recent()

    def _draw_recent(self) -> None:
        for w in self.recent_box.winfo_children():
            w.destroy()
        if not self._recent:
            tk.Label(self.recent_box, text="아직 측정 결과가 없습니다", bg=BG, fg=FAINT,
                     font=self.NF(11)).pack(anchor="w", pady=4)
            return
        for i, (name, vals) in enumerate(self._recent):
            row = tk.Frame(self.recent_box, bg=BG)
            row.pack(fill="x", pady=(0, 3))
            tk.Label(row, text=name, bg=BG, fg=FG if i == 0 else DIM,
                     font=self.NF(11, True), width=11, anchor="w").pack(side="left")
            parts = []
            for p in PROPS:
                v = vals.get(p["key"])
                parts.append(f"{v:.2f} {p['unit']}" if v is not None else f"-- {p['unit']}")
            tk.Label(row, text=" · ".join(parts), bg=BG,
                     fg=DIM if i == 0 else FAINT, font=self.NF(11),
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
    cn = ["soft", "mid", "hard"]
    # fruit: (stiffness, s_std, s_max, boundaries), (weight, w_std), (diameter, d_std)
    plan = [
        ("tomato", (1.97, 0.31, 6.06, [1.905, 3.075]), (68.35, 3.1), (42.03, 1.2)),
        ("kiwi",   (3.03, 0.68, 7.63, [2.07, 3.672]),  (59.02, 2.6), (20.12, 0.9)),
        ("kiwi",   (2.01, 0.44, 7.63, [2.07, 3.672]),  (48.89, 2.2), (18.11, 0.8)),
        ("plum",   (1.02, 0.55, 7.65, [1.761, 3.84]),  (58.00, 4.0), (44.00, 2.0)),
        ("lemon",  (7.04, 0.42, 10.14, [3.416, 4.894]),(112.0, 6.0), (70.00, 2.5)),
    ]
    steps = []
    for fr, s, w, d in plan:
        sval, sstd, smax, b = s
        cls = sum(1 for x in b if sval >= x)
        common = {"fruit": fr, "stiffness_max": smax,
                  "boundaries": b, "class_names": cn}
        steps.append(({"phase": "measuring", **common}, 1600))
        steps.append(({"phase": "done",
                       "stiffness": sval, "stiffness_std": sstd, "cname": cn[cls],
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
    ap = argparse.ArgumentParser(description="과일 물성(크기/강성/무게) 추정 GUI")
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