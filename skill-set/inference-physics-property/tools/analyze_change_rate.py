#!/usr/bin/env python3
"""analyze_change_rate.py — **값 변화 간격 분포**로 센서 update rate 분석 (최소~최대·평균).

`sensor_update_rate.py` 가 저장한 CSV(`sensor_*.csv`)를 읽어, "발행 tick" 이 아니라
**실제 데이터 값이 바뀐 시각들 사이의 간격**을 통계로 낸다.

  · 변화간격(ms)   : 최소 / 25% / 중앙 / 평균 / 75% / 최대
  · rate 환산      : ★평균 = 변화횟수/시간,  robust 범위 = 사분위 간격 역수,
                     1초창 최소~최대 = 시간에 따른 실제 변동폭
  · 1초 창 rate    : 슬라이딩 윈도우별 rate 의 최소/평균/최대 (순간 편차 확인용)
  · 채널별         : 채널마다 같은 분석 (어느 손가락/축이 얼마나 자주 바뀌는지)

구간 분리:
  전체 = 측정 전체.  활성 = 변화가 실제로 일어나던 시간대만(±WIN 내 변화 존재).
  촉각은 무접촉 구간이 길어 전체 평균이 크게 낮아지므로 **활성 구간 값을 봐야 한다.**

⚠ 측정 분해능: 값 변화는 **발행 tick 단위로만** 관측된다(paxini 11.1ms, 관절 5ms).
   개별 간격은 tick 의 정수배로 양자화되므로 '개별간격 극단값'(1/최소간격)은 도착 지터를
   반영할 뿐 센서 rate 가 아니다. → **평균 rate**(변화횟수/시간)와 **1초창 범위**로 판정한다.

사용:
  python3 tools/analyze_change_rate.py docs/rate_log/<run>
  python3 tools/analyze_change_rate.py docs/rate_log/<run> --channels     # 채널별까지
  python3 tools/analyze_change_rate.py docs/rate_log/<run> --md           # 마크다운 표로
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

EPS = 1e-4          # 이보다 작은 변화는 LSB 지터로 무시 (sensor_update_rate.py 와 동일)
ACTIVE_WIN = 0.25   # 활성 구간 판정: ±0.25s 내에 변화가 있으면 '활성'
WINDOW = 1.0        # 슬라이딩 창 크기(s)


def load(path: Path):
    rows = list(csv.reader(open(path)))[1:]
    if not rows:
        return None, None
    t = np.array([float(r[0]) for r in rows])
    v = np.array([[float(x) for x in r[2:]] for r in rows])
    return t, v


def topic_names(run: Path) -> dict[str, str]:
    """CSV 파일명은 '/'→'_' 로 뭉개지므로(q_target 구분 불가) JSON 에서 원본 토픽명을 복원."""
    j = run / "sensor_change.json"
    if not j.exists():
        return {}
    import json
    return {"sensor" + r["topic"].replace("/", "_") + ".csv": r["topic"]
            for r in json.load(open(j))}


def stats(gaps_ms: np.ndarray, n_changes: int, span: float) -> dict:
    """변화간격 배열 → 간격/rate 통계."""
    if n_changes == 0 or span <= 0:
        return {}
    out = {"n": n_changes, "span": span, "mean_rate": n_changes / span}
    if len(gaps_ms):
        p25, p75 = np.percentile(gaps_ms, [25, 75])
        out.update(
            g_min=float(gaps_ms.min()), g_p25=float(p25),
            g_med=float(np.median(gaps_ms)), g_mean=float(gaps_ms.mean()),
            g_p75=float(p75), g_max=float(gaps_ms.max()),
            # 개별 간격의 극단값 — 도착 지터(tick 보다 짧은 간격)와 idle 갭 때문에
            # 센서 rate 로 읽으면 안 된다. 참고용으로만 보관.
            r_inst_min=1000.0 / gaps_ms.max(), r_inst_max=1000.0 / gaps_ms.min(),
            # robust 범위: 사분위 간격 역수 → "대부분의 변화가 일어나는 rate 대역"
            r_p25=1000.0 / float(p75), r_p75=1000.0 / float(p25))
    return out


def window_rates(chg_t: np.ndarray, t0: float, t1: float, win: float) -> np.ndarray:
    """[t0,t1] 을 win 초 창으로 나눠 창별 변화 횟수/win = rate."""
    if t1 - t0 < win:
        return np.array([])
    edges = np.arange(t0, t1, win)
    cnt, _ = np.histogram(chg_t, bins=np.append(edges, edges[-1] + win))
    return cnt / win


def analyze(t: np.ndarray, v: np.ndarray, mag_thresh: float | None = None) -> dict:
    """토픽 1개 분석 → {tick, 전체, 활성, 채널별}."""
    dt = np.diff(t)
    tick_ms = 1000.0 * float(np.median(dt)) if len(dt) else 0.0

    d = np.abs(np.diff(v, axis=0))
    changed = (d >= EPS).any(axis=1)          # 프레임별 '유의미 변화' 여부
    chg_t = t[1:][changed]

    res = {"tick_ms": tick_ms, "n_frames": len(t), "channels": v.shape[1]}

    # ── 전체 구간 ──
    span = float(t[-1] - t[0])
    gaps = 1000.0 * np.diff(chg_t)
    res["all"] = stats(gaps, len(chg_t), span)
    if len(chg_t) >= 2:
        res["all"]["win"] = window_rates(chg_t, t[0], t[-1], WINDOW)

    # ── 활성 구간: 변화 시각 ±ACTIVE_WIN 에 걸치는 프레임만 ──
    if len(chg_t):
        # 프레임별로 가장 가까운 변화까지의 거리
        idx = np.searchsorted(chg_t, t)
        lo = np.clip(idx - 1, 0, len(chg_t) - 1)
        hi = np.clip(idx, 0, len(chg_t) - 1)
        near = np.minimum(np.abs(t - chg_t[lo]), np.abs(t - chg_t[hi]))
        active = near <= ACTIVE_WIN
        a_span = float(dt[active[1:]].sum()) if len(dt) else 0.0
        a_mask = active[1:] & changed
        a_t = t[1:][a_mask]
        a_gaps = 1000.0 * np.diff(a_t)
        # 활성 구간이 끊길 때 생기는 큰 갭은 제외(구간 경계) — 2*ACTIVE_WIN 초과 갭 제거
        a_gaps = a_gaps[a_gaps <= 2000 * ACTIVE_WIN]
        res["active"] = stats(a_gaps, int(a_mask.sum()), a_span)
        res["active_ratio"] = float(active.mean())

    # ── 부하 구간: 신호 크기가 임계 이상인 프레임만 (촉각의 '실접촉 중' 판정) ──
    if mag_thresh is not None:
        mag = np.abs(v).max(axis=1)
        load_m = mag[1:] >= mag_thresh
        l_span = float(dt[load_m].sum()) if len(dt) else 0.0
        l_mask = load_m & changed
        l_gaps = 1000.0 * np.diff(t[1:][l_mask])
        l_gaps = l_gaps[l_gaps <= 2000 * ACTIVE_WIN]
        res["load"] = stats(l_gaps, int(l_mask.sum()), l_span)
        res["load_thresh"] = mag_thresh
        if res["load"] and len(t[1:][l_mask]) >= 2:
            lt = t[1:][l_mask]
            res["load"]["win"] = window_rates(lt, lt[0], lt[-1], WINDOW)

    # ── 채널별 ──
    ch = []
    for i in range(v.shape[1]):
        ci = d[:, i] >= EPS
        ct = t[1:][ci]
        cg = 1000.0 * np.diff(ct)
        s = stats(cg, len(ct), span)
        s["ch"] = i
        ch.append(s)
    res["ch"] = ch
    return res


def fmt_gap(s: dict) -> str:
    if not s or "g_min" not in s:
        return "-"
    return (f"최소 {s['g_min']:.2f} / 25% {s['g_p25']:.2f} / 중앙 {s['g_med']:.2f} / "
            f"평균 {s['g_mean']:.2f} / 75% {s['g_p75']:.2f} / 최대 {s['g_max']:.2f} ms")


def print_text(topic: str, r: dict, show_ch: bool) -> None:
    print(f"\n{'='*78}\n[{topic}]  {r['channels']}ch, {r['n_frames']}프레임")
    print(f"  발행 tick(측정 분해능) = {r['tick_ms']:.2f}ms → {1000/r['tick_ms']:.2f}Hz"
          if r["tick_ms"] else "  tick 불명")
    if not r.get("all"):
        print("  ❌ 값 변화 0회 (FROZEN) — rate 분석 불가")
        return
    labels = [("all", "전체 구간"), ("active", "활성 구간")]
    if r.get("load"):
        labels.append(("load", f"부하 구간(|값|≥{r['load_thresh']:g})"))
    for key, label in labels:
        s = r.get(key)
        if not s or "g_min" not in s:
            continue
        extra = ""
        if key == "active" and "active_ratio" in r:
            extra = f" (전체의 {100*r['active_ratio']:.0f}%)"
        print(f"\n  ── {label}{extra}: {s['span']:.1f}s, 변화 {s['n']}회")
        print(f"     ★ 평균 rate   : {s['mean_rate']:7.2f} Hz   (= 변화횟수/시간)")
        print(f"     robust 범위   : {s['r_p25']:7.2f} ~ {s['r_p75']:6.2f} Hz"
              "   (사분위 간격 역수 — 변화 대부분이 이 대역)")
        w = s.get("win")
        if w is not None and len(w):
            print(f"     1초창 최소~최대: {w.min():7.2f} ~ {w.max():6.2f} Hz"
                  f"   (평균 {w.mean():.2f})")
        print(f"     변화간격      : {fmt_gap(s)}")
        print(f"     (참고) 개별간격 극단: {s['r_inst_min']:.2f} ~ {s['r_inst_max']:.2f} Hz"
              " — 도착 지터·idle 갭 탓, 센서 rate 아님")
    if show_ch:
        print(f"\n  ── 채널별 (전체 구간 기준)")
        print(f"     {'ch':>3} {'변화횟수':>8} {'평균Hz':>8} {'25%Hz':>8} {'75%Hz':>8}"
              f"  {'평균간격ms':>10}")
        for s in r["ch"]:
            if not s or "g_min" not in s:
                print(f"     {s.get('ch','?'):>3} {s.get('n',0):>8}"
                      "        —        —        —           —")
                continue
            print(f"     {s['ch']:>3} {s['n']:>8} {s['mean_rate']:>8.2f} "
                  f"{s['r_p25']:>8.2f} {s['r_p75']:>8.2f}  {s['g_mean']:>10.2f}")


def print_md(rows: list[tuple[str, dict]]) -> None:
    print("\n| 토픽 | 구간 | 변화횟수 | 평균 rate | robust 범위 | 1초창 최소~최대 "
          "| 평균간격 |")
    print("|---|---|---|---|---|---|---|")
    for topic, r in rows:
        for key, label in (("all", "전체"), ("active", "활성"), ("load", "부하")):
            s = r.get(key)
            if not s or "g_min" not in s:
                continue
            w = s.get("win")
            wtxt = (f"{w.min():.0f}~{w.max():.0f}Hz" if w is not None and len(w) else "-")
            print(f"| `{topic}` | {label} | {s['n']} | **{s['mean_rate']:.2f}Hz** | "
                  f"{s['r_p25']:.1f}~{s['r_p75']:.1f}Hz | {wtxt} | {s['g_mean']:.2f}ms |")


def main() -> int:
    ap = argparse.ArgumentParser(description="값 변화 간격 기반 update rate 분석")
    ap.add_argument("run", type=Path, help="docs/rate_log/<run> 디렉토리")
    ap.add_argument("--channels", action="store_true", help="채널별 표까지 출력")
    ap.add_argument("--md", action="store_true", help="마크다운 표로 출력")
    args = ap.parse_args()

    csvs = sorted(args.run.glob("sensor_*.csv"))
    if not csvs:
        print(f"CSV 없음: {args.run}/sensor_*.csv")
        return 1

    names = topic_names(args.run)
    rows = []
    for f in csvs:
        topic = names.get(f.name, "/" + f.stem[len("sensor_"):])
        t, v = load(f)
        if t is None or len(t) < 3:
            print(f"[skip] {topic}: 데이터 부족")
            continue
        if float(np.median(np.diff(t))) <= 0:
            print(f"[skip] {topic}: 타임스탬프 정밀도 손실(구버전 CSV) → 간격 분석 불가")
            continue
        # 촉각은 '실접촉 중' 구간을 따로 본다(무접촉 0N 구간이 평균을 끌어내리므로).
        r = analyze(t, v, mag_thresh=1.0 if "paxini" in topic else None)
        rows.append((topic, r))
        if not args.md:
            print_text(topic, r, args.channels)

    if args.md:
        print_md(rows)
        print("\n> 측정 분해능 = 발행 tick "
              + ", ".join(f"{t} {r['tick_ms']:.2f}ms" for t, r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
