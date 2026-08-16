#!/usr/bin/env python3
"""sensor_update_rate.py — **센서 실측 update rate** 계측.

`ros2 topic hz` 는 **토픽 발행률(통신 속도)** 이다. 퍼블리셔가 SHM 의 같은 값을 200Hz 로
재발행하면 hz 는 200Hz 로 보이지만 **센서는 그보다 느리게 갱신**될 수 있다. 이 도구는
"값이 실제로 바뀌는 빈도" = **센서 update rate** 를 재고, 값이 잘 변하는지 채널별로 확인한다.

왜 이렇게 밖에 못 재나:
  `/paxini/right/ft` 는 timestamp/seq 없는 Float32MultiArray(12) 다 — 브리지의 seq 는 그냥
  '수신 메시지 카운터'(deploy_ros2.py `_on_ft`)이고 t 는 로컬 시각이라, 원본 센서의 샘플
  시각을 알 방법이 없다. 그래서 **payload 값의 변화**로 역산한다.
  (header 가 있는 메시지는 stamp 갱신 빈도도 함께 보고한다.)

사용:
  source env.sh
  python3 tools/sensor_update_rate.py --duration 20
  python3 tools/sensor_update_rate.py --duration 20 --topics /paxini/right/ft /hand/right/kin
  python3 tools/sensor_update_rate.py --duration 20 --out docs/rate_log/<run>   # CSV/JSON 저장

해석:
  · 메시지율 ≈ 변화율      → 매 메시지가 새 샘플 (오버샘플 없음)
  · 메시지율 >> 변화율     → 재발행/스테일. **추론 입력의 실제 정보량은 '변화율' 기준**
  · 변화 0 인 채널         → 값이 안 변함(고정). 접촉 없는 paxini 는 정상적으로 0 고정일 수 있음
                             → 접촉/모션 중에 다시 재보고 판단할 것
"""
from __future__ import annotations

import argparse
import json
import signal
import statistics as st
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message

# 기본 감시 대상 — 이 패키지가 구독하는 센서/상태 토픽 (README §6 토픽 계약).
DEFAULT_TOPICS = [
    "/paxini/right/ft",             # 촉각 합력(4x3) — 힘 판정 + 추론 입력
    "/hand/right/joint_states",     # 손 관절(count)
    "/hand/right/kin",              # 손 kinesthetic(12) — 추론 입력
    "/franka/right/joint_states",   # 팔 관절
]

# 값 변화 판정 임계. paxini Fz 는 0.1N 단위로 양자화돼 오므로 그보다 작게 잡는다.
EPS = 1e-6

# 이보다 작은 변화만 있으면 '실제 갱신'이 아니라 부동소수 LSB 지터로 본다.
# (Franka 관절이 정지 상태인데도 매 메시지 1e-6 수준으로 흔들려 '변화'로 잡히는 것을 걸러냄)
JITTER_EPS = 1e-4


class Tracker:
    """토픽 1개의 메시지/값변화 통계."""

    def __init__(self, topic: str, type_str: str, nch_hint: int = 0):
        self.topic = topic
        self.type_str = type_str
        self.n_msgs = 0
        self.n_changed = 0
        self.t_first: float | None = None
        self.t_last: float | None = None
        self.prev: np.ndarray | None = None
        self.change_times: list[float] = []      # 값이 바뀐 시각 (간격 통계용)
        self.ch_changes: np.ndarray | None = None    # 채널별 변화 횟수
        self.ch_min_delta: np.ndarray | None = None  # 채널별 최소 |변화폭| → 양자화 단위 추정
        self.ch_min: np.ndarray | None = None
        self.ch_max: np.ndarray | None = None
        self.ch_sum: np.ndarray | None = None
        self.ch_sumsq: np.ndarray | None = None
        self.stamps: set[int] = set()             # header.stamp (있으면) 고유값
        self.rows: list[tuple] = []               # CSV 저장용 (t, changed, values…)
        self.delta_peaks: list[float] = []        # 변화 1회당 max|Δ| → 지터/실갱신 구분

    def _init_channels(self, n: int) -> None:
        self.ch_changes = np.zeros(n, np.int64)
        self.ch_min_delta = np.full(n, np.inf)
        self.ch_min = np.full(n, np.inf)
        self.ch_max = np.full(n, -np.inf)
        self.ch_sum = np.zeros(n, np.float64)
        self.ch_sumsq = np.zeros(n, np.float64)

    def add(self, vec: np.ndarray, stamp_ns: int | None) -> None:
        now = time.monotonic()
        self.n_msgs += 1
        if self.t_first is None:
            self.t_first = now
        self.t_last = now
        if stamp_ns is not None:
            self.stamps.add(stamp_ns)

        if self.ch_changes is None or len(self.ch_changes) != len(vec):
            self._init_channels(len(vec))

        self.ch_min = np.minimum(self.ch_min, vec)
        self.ch_max = np.maximum(self.ch_max, vec)
        self.ch_sum += vec
        self.ch_sumsq += vec.astype(np.float64) ** 2

        changed = False
        if self.prev is not None:
            d = np.abs(vec - self.prev)
            moved = d > EPS
            if moved.any():
                changed = True
                self.n_changed += 1
                self.change_times.append(now)
                self.ch_changes[moved] += 1
                self.delta_peaks.append(float(d.max()))
                # 채널별 최소 변화폭(양자화 단위 추정) — 0 이 아닌 델타만.
                self.ch_min_delta = np.where(
                    moved & (d < self.ch_min_delta), d, self.ch_min_delta)
        self.prev = vec.copy()
        self.rows.append((now, int(changed), *vec.tolist()))

    # ── 결과 ─────────────────────────────────────────────────────────
    def report(self) -> dict:
        span = (self.t_last - self.t_first) if (self.t_first and self.t_last) else 0.0
        msg_hz = (self.n_msgs - 1) / span if span > 0 and self.n_msgs > 1 else 0.0
        chg_hz = self.n_changed / span if span > 0 else 0.0
        gaps = [b - a for a, b in zip(self.change_times, self.change_times[1:])]
        n = len(self.ch_changes) if self.ch_changes is not None else 0
        mean = (self.ch_sum / self.n_msgs) if self.n_msgs else np.zeros(n)
        var = (self.ch_sumsq / self.n_msgs - mean ** 2) if self.n_msgs else np.zeros(n)
        std = np.sqrt(np.clip(var, 0, None))

        stamp_hz = None
        if self.stamps:
            stamp_hz = (len(self.stamps) - 1) / span if span > 0 else 0.0

        # 채널 진폭(range) 과 변화폭(Δ) 으로 '실제 갱신 / LSB 지터 / 완전 정지' 판정.
        rng = (self.ch_max - self.ch_min) if n else np.zeros(0)
        max_range = float(rng.max()) if n else 0.0
        dmed = st.median(self.delta_peaks) if self.delta_peaks else 0.0
        # 유의미 변화율: LSB 지터를 뺀 갱신 빈도.
        n_signif = sum(1 for v in self.delta_peaks if v >= JITTER_EPS)
        signif_hz = n_signif / span if span > 0 else 0.0

        if self.n_changed == 0:
            verdict = "FROZEN"          # 값이 단 한 번도 안 바뀜
        elif n_signif == 0:
            # 매 메시지 흔들려도 전부 LSB 수준이면 실질 정지. (정지한 팔의 관절값이 이 경우 —
            # 누적 range 가 JITTER_EPS 를 넘어도 스텝이 1e-6 이면 갱신으로 볼 수 없다.)
            verdict = "JITTER"
        else:
            verdict = "LIVE"

        # 활성 구간 갱신율 — idle(값 고정) 구간이 섞이면 전체 평균은 실제보다 낮게 나온다.
        # 변화 간격의 '중앙값' 역수 = 실제로 갱신될 때의 rate.
        # ※ LSB 지터를 뺀 '유의미 변화' 시각만 써야 한다 — 전체 변화로 계산하면 정지한 팔이
        #   200Hz 로 갱신되는 것처럼 보인다(매 메시지 1e-6 흔들림).
        sig_t = [t for t, d in zip(self.change_times, self.delta_peaks)
                 if d >= JITTER_EPS]
        sgaps = [b - a for a, b in zip(sig_t, sig_t[1:])]
        gmed = st.median(sgaps) if sgaps else 0.0
        active_hz = (1.0 / gmed) if gmed > 0 else 0.0

        return {
            "verdict": verdict,
            "change_hz_significant": round(signif_hz, 2),
            "change_hz_active": round(active_hz, 2),
            "delta_peak_median": None if not self.delta_peaks else float(f"{dmed:.3g}"),
            "delta_peak_max": (None if not self.delta_peaks
                               else float(f"{max(self.delta_peaks):.3g}")),
            "max_channel_range": float(f"{max_range:.6g}"),
            "frozen_value": (self.prev.tolist() if verdict in ("FROZEN", "JITTER")
                             and self.prev is not None else None),
            "topic": self.topic,
            "type": self.type_str,
            "channels": n,
            "span_sec": round(span, 3),
            "n_msgs": self.n_msgs,
            "msg_hz": round(msg_hz, 2),
            "n_changed": self.n_changed,
            "change_hz": round(chg_hz, 2),
            "dup_pct": round(100 * (1 - self.n_changed / max(1, self.n_msgs - 1)), 1),
            "oversample_x": round(msg_hz / chg_hz, 2) if chg_hz > 0 else None,
            "gap_ms": {
                "mean": round(1000 * st.fmean(gaps), 2) if gaps else None,
                "median": round(1000 * st.median(gaps), 2) if gaps else None,
                "min": round(1000 * min(gaps), 2) if gaps else None,
                "max": round(1000 * max(gaps), 2) if gaps else None,
            },
            "stamp_unique": len(self.stamps) or None,
            "stamp_hz": round(stamp_hz, 2) if stamp_hz is not None else None,
            "ch_changes": self.ch_changes.tolist() if n else [],
            "ch_static": [i for i in range(n) if self.ch_changes[i] == 0],
            "ch_quant": [None if not np.isfinite(v) else round(float(v), 6)
                         for v in (self.ch_min_delta if n else [])],
            "ch_std": [round(float(v), 6) for v in std] if n else [],
            "ch_min": [round(float(v), 4) for v in self.ch_min] if n else [],
            "ch_max": [round(float(v), 4) for v in self.ch_max] if n else [],
        }


class SensorRateNode(Node):
    def __init__(self, topics: list[str], depth: int, discovery_sec: float = 5.0):
        super().__init__("sensor_update_rate")
        self.trackers: dict[str, Tracker] = {}

        # DDS 디스커버리가 끝나기 전에 조회하면 토픽 목록이 비어 있다 → 다 찾을 때까지 재시도.
        t0 = time.monotonic()
        available: dict[str, list[str]] = {}
        while time.monotonic() - t0 < discovery_sec:
            available = dict(self.get_topic_names_and_types())
            if all(t in available for t in topics):
                break
            time.sleep(0.3)

        # 센서 갱신을 놓치지 않으려면 depth 를 넉넉히 (배포 루프는 depth=1 이라 더 많이 흘린다).
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=depth)

        for t in topics:
            types = available.get(t)
            if not types:
                print(f"[sensor_rate] ⚠ 토픽 없음(퍼블리셔 미발견): {t}")
                continue
            type_str = types[0]
            try:
                msg_cls = get_message(type_str)
            except Exception as e:                       # noqa: BLE001
                print(f"[sensor_rate] ⚠ 타입 로드 실패 {t} ({type_str}): {e}")
                continue
            tr = Tracker(t, type_str)
            self.trackers[t] = tr
            self.create_subscription(
                msg_cls, t, lambda m, _tr=tr: self._on_msg(m, _tr), qos)
            print(f"[sensor_rate] 감시: {t}  ({type_str})")

    @staticmethod
    def _extract(msg) -> tuple[np.ndarray, int | None]:
        """메시지 → (값 벡터, header stamp ns 또는 None)."""
        stamp = None
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            s = msg.header.stamp
            stamp = int(s.sec) * 1_000_000_000 + int(s.nanosec)

        if hasattr(msg, "position") and hasattr(msg, "name"):        # JointState
            vec = np.asarray(msg.position, np.float64)
        elif hasattr(msg, "data"):                                   # *MultiArray
            vec = np.asarray(msg.data, np.float64).ravel()
        else:
            vec = np.zeros(0)
        return vec, stamp

    def _on_msg(self, msg, tr: Tracker) -> None:
        vec, stamp = self._extract(msg)
        if vec.size:
            tr.add(vec, stamp)


def _short(seq, n: int = 16) -> str:
    """채널 수가 많은 토픽(/paxini/right/raw = 1524ch)의 리스트를 잘라서 출력."""
    seq = list(seq)
    if len(seq) <= n:
        return str(seq)
    return f"{seq[:n]}… (총 {len(seq)}개)"


_VERDICT_MSG = {
    "LIVE": "✅ LIVE — 값이 실제로 갱신됨",
    "JITTER": "⚠ JITTER — LSB 수준만 흔들림(실질 정지). 센서 갱신으로 볼 수 없음",
    "FROZEN": "❌ FROZEN — 값이 전혀 안 바뀜(같은 값 재발행)",
}


def print_report(rep: dict) -> None:
    q = [v for v in rep["ch_quant"] if v]
    print(f"\n[{rep['topic']}]  {rep['type']}, {rep['channels']}ch, {rep['span_sec']}s")
    print(f"  판정            : {_VERDICT_MSG[rep['verdict']]}")
    print(f"  메시지 발행률   : {rep['n_msgs']:6d}개  {rep['msg_hz']:7.2f} Hz"
          "   ← ros2 topic hz 와 같은 값(통신)")
    print(f"  값 변화율(raw)  : {rep['n_changed']:6d}회  {rep['change_hz']:7.2f} Hz"
          "   (LSB 지터 포함)")
    print(f"  ★ 유의미 갱신율 : {rep['change_hz_significant']:7.2f} Hz"
          f"   ← 전체 평균 (idle 포함, |Δ| ≥ {JITTER_EPS:g})")
    print(f"  ★ 활성 갱신율   : {rep['change_hz_active']:7.2f} Hz"
          "   ← 갱신될 때의 rate (1/변화간격 중앙값)")
    print(f"  변화폭 max|Δ|   : 중앙 {rep['delta_peak_median']}  최대 {rep['delta_peak_max']}"
          f"   / 채널 최대진폭 {rep['max_channel_range']}")
    print(f"  중복(스테일)    : {rep['dup_pct']}%"
          + (f"   → {rep['oversample_x']}배 오버샘플" if rep["oversample_x"] else ""))
    if rep["frozen_value"] is not None:
        vals = [round(v, 4) for v in rep["frozen_value"]]
        allzero = not any(vals)
        print(f"  고정된 값       : {_short(vals)}"
              + ("   ← 전 채널 정확히 0 (센서 미동작 의심)" if allzero else ""))
    g = rep["gap_ms"]
    if g["mean"]:
        print(f"  변화 간격       : 평균 {g['mean']}ms  중앙 {g['median']}ms  "
              f"최소 {g['min']}ms  최대 {g['max']}ms")
    if rep["stamp_hz"] is not None:
        print(f"  header stamp    : 고유 {rep['stamp_unique']}개  {rep['stamp_hz']} Hz")
    else:
        print("  header stamp    : 없음 (timestamp/seq 미제공 → 값 변화로만 판정 가능)")
    print(f"  채널별 변화횟수 : {_short(rep['ch_changes'])}")
    n_static, n_ch = len(rep["ch_static"]), rep["channels"]
    if n_static:
        print(f"  ⚠ 정적 채널     : {n_static}/{n_ch}개 값 고정 "
              f"{_short(rep['ch_static'], 12)}")
    else:
        print(f"  ✅ 전 채널({n_ch}) 변함")
    if q:
        print(f"  최소 변화폭     : {min(q)} (양자화 단위 추정)")
    print(f"  채널 std        : {_short([round(v, 4) for v in rep['ch_std']])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="센서 실측 update rate(값 변화 빈도) 계측")
    ap.add_argument("--duration", type=float, default=20.0, help="계측 시간(s)")
    ap.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    ap.add_argument("--depth", type=int, default=200, help="구독 큐 depth(갱신 누락 방지)")
    ap.add_argument("--out", type=Path, default=None,
                    help="결과 저장 디렉토리 (sensor_change.json + 토픽별 CSV)")
    args = ap.parse_args(argv)

    rclpy.init()
    node = SensorRateNode(args.topics, args.depth)
    if not node.trackers:
        print("[sensor_rate] 감시할 토픽이 없습니다 — 제어 PC 스택/paxini_writer 실행 확인.")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    ex = SingleThreadedExecutor()
    ex.add_node(node)
    print(f"[sensor_rate] {args.duration}s 수집 중... "
          "(모션·접촉 중에 재는 것이 의미 있음)")

    # measure_update_rate.sh 가 배포 종료 후 TERM 을 보내 멈춘다 → 그때까지 모은 것으로 리포트.
    stop = {"flag": False}

    def _stop(_sig, _frm):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    t0 = time.monotonic()
    try:
        while not stop["flag"] and time.monotonic() - t0 < args.duration:
            ex.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    if stop["flag"]:
        print("\n[sensor_rate] 중단 신호 — 수집분으로 리포트")

    reports = [tr.report() for tr in node.trackers.values()]
    print("\n" + "=" * 72)
    print("센서 실측 update rate — '발행률'(통신) vs '값 변화율'(센서)")
    print("=" * 72)
    for rep in reports:
        print_report(rep)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "sensor_change.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2))
        for tr in node.trackers.values():
            name = "sensor" + tr.topic.replace("/", "_") + ".csv"
            n = len(tr.ch_changes) if tr.ch_changes is not None else 0
            header = "t_mono,changed," + ",".join(f"ch{i}" for i in range(n))
            # t 는 monotonic(수백만 초)이라 %g 로 쓰면 자릿수가 뭉개져 간격 분석이 불가능해진다
            # → 시간만 고정소수점(%.6f)으로, 값은 %g 로.
            lines = [header] + [
                ",".join([f"{row[0]:.6f}", str(int(row[1]))]
                         + [f"{v:.6g}" for v in row[2:]])
                for row in tr.rows]
            (args.out / name).write_text("\n".join(lines) + "\n")
        print(f"\n[sensor_rate] 저장 완료 → {args.out}/sensor_change.json (+ 토픽별 CSV)")

    node.destroy_node()
    ex.shutdown()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
