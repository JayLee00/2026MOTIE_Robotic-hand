#!/usr/bin/env python3
"""rate_summary.py — 계측 로그(docs/rate_log/<run>/) → 마크다운 요약.

measure_update_rate.sh 가 마지막에 자동 호출하지만, **나중에 따로 재분석**해도 된다:

    python3 tools/rate_summary.py docs/rate_log/20260727_101500_kiwi_baseline
    python3 tools/rate_summary.py docs/rate_log/*            # 여러 run 을 한 표로

파싱 대상:
  hz_*.log        — `<epoch> average rate: <Hz>` 시계열 (와이어 rate)
  exp_stdout.log  — deploy_ros2_exp 의 `[measure]` 블록 (루프 유효 rate / steps / thumb Fz)

출력: 각 run 디렉토리의 summary.md + stdout. (여러 run 이면 통합표를 stdout 에만 출력)
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

# 모터가 도는 구간만 평균에 넣는다. 과일 선택 대기 등 idle 구간(0Hz 근처)이 섞이면
# q_target 평균이 의미 없이 낮게 나온다.
ACTIVE_MIN_HZ = 1.0

_RATE_RE = re.compile(r"^(?P<ts>\d+\.\d+)\s+average rate:\s+(?P<hz>[\d.]+)")

# deploy_ros2_exp 의 [measure] 블록 (deploy_ros2_exp.py::MeasureEngine._report 출력과 1:1)
_M_CALLS = re.compile(r"add_sample 호출 = (\d+),\s+valid\(적재\) = (\d+)")
_M_RATE = re.compile(r"유효 rate\s+=\s+([\d.]+) Hz\s+\(수집시간 ([\d.]+)s\)")
_M_STEPS = re.compile(r"downsample 스텝 = (\d+)\s+\(FACTOR=(\d+), MIN_LEN=(\d+)\)")
_M_THUMB = re.compile(r"thumb 최대 = ([\d.-]+) N\s+\(스퀴즈 임계 ([\d.]+) N\)")
_M_FZ = re.compile(r"finger별 최대 Fz = \[(.*?)\]")
_M_FRUIT = re.compile(r"\[threshold\] (\w+): 파지=([\d.]+)N, 스퀴즈=([\d.]+)N")


def _fmt(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def parse_hz(path: Path) -> dict:
    """hz 로그 → {topic, samples, active, mean, median, min, max, span}."""
    ts, hz = [], []
    topic = path.stem                             # 헤더가 없을 때의 폴백(파일명)
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("# topic:"):           # 스크립트가 남긴 원본 토픽명
            topic = line.split(":", 1)[1].strip()
            continue
        m = _RATE_RE.match(line)
        if m:
            ts.append(float(m["ts"]))
            hz.append(float(m["hz"]))
    active = [v for v in hz if v >= ACTIVE_MIN_HZ]
    return {
        "topic": topic,
        "samples": len(hz),
        "active": len(active),
        "mean": st.fmean(active) if active else None,
        "median": st.median(active) if active else None,
        "min": min(active) if active else None,
        "max": max(active) if active else None,
        "span": (ts[-1] - ts[0]) if len(ts) >= 2 else 0.0,
    }


def parse_exp(path: Path) -> tuple[list[dict], dict]:
    """exp stdout → ([스퀴즈별 계측 dict], {fruit, grip_thr, squeeze_thr})."""
    text = path.read_text(errors="replace")
    cfg = {}
    m = _M_FRUIT.search(text)
    if m:
        cfg = {"fruit": m[1], "grip_thr": float(m[2]), "squeeze_thr": float(m[3])}

    blocks = []
    for chunk in text.split("[measure] 스퀴즈 계측")[1:]:
        chunk = chunk[:1200]                      # 블록 하나 범위로 제한
        rec: dict = {}
        if (m := _M_CALLS.search(chunk)):
            rec["calls"], rec["valid"] = int(m[1]), int(m[2])
        if (m := _M_RATE.search(chunk)):
            rec["rate"], rec["dur"] = float(m[1]), float(m[2])
        if (m := _M_STEPS.search(chunk)):
            rec["steps"], rec["factor"], rec["min_len"] = int(m[1]), int(m[2]), int(m[3])
        if (m := _M_THUMB.search(chunk)):
            rec["thumb"], rec["sq_thr"] = float(m[1]), float(m[2])
        if (m := _M_FZ.search(chunk)):
            rec["fz"] = [round(float(v), 2) for v in m[1].split(",") if v.strip()]
        if rec:
            blocks.append(rec)
    return blocks, cfg


def summarize(run: Path) -> str:
    """run 디렉토리 하나 → 마크다운 문자열."""
    out = [f"# update rate 계측 요약 — `{run.name}`", ""]

    meta = run / "meta.txt"
    if meta.exists():
        out += ["## 실행 조건", "", "```", meta.read_text().rstrip(), "```", ""]

    # ── 와이어 rate (ros2 topic hz) ──
    hz_logs = sorted(run.glob("hz_*.log"))
    if hz_logs:
        out += [
            "## 와이어 rate — `ros2 topic hz`",
            "",
            f"(idle 구간 제외: {ACTIVE_MIN_HZ}Hz 미만 샘플은 평균에서 제외)",
            "",
            "| 토픽 | 유효샘플/전체 | 평균 Hz | 중앙값 | 최소 | 최대 |",
            "|---|---|---|---|---|---|",
        ]
        for log in hz_logs:
            r = parse_hz(log)
            out.append(
                f"| `{r['topic']}` | {r['active']}/{r['samples']} | {_fmt(r['mean'])} | "
                f"{_fmt(r['median'])} | {_fmt(r['min'])} | {_fmt(r['max'])} |")
        out.append("")

    # ── 센서 실측 update rate (sensor_update_rate.py) ──
    sensor = run / "sensor_change.json"
    if sensor.exists():
        reps = json.loads(sensor.read_text())
        out += [
            "## 센서 실측 update rate — 값 변화 기준",
            "",
            "`ros2 topic hz`(발행률)와 달리 **값이 실제로 바뀌는 빈도**. 퍼블리셔가 같은 값을",
            "재발행하면 발행률은 정상이어도 센서는 멈춰 있다.",
            "",
            "| 토픽 | 판정 | 발행률(Hz) | 갱신(전체 평균) | 갱신(활성) | 중복 | 정적채널 | 비고 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in reps:
            note = ""
            if r["verdict"] == "FROZEN":
                fv = r.get("frozen_value") or []
                note = "전 채널 0 고정" if fv and not any(fv) else "값 고정"
            elif r["verdict"] == "JITTER":
                # 표 셀 안의 '|' 는 escape 해야 마크다운 표가 깨지지 않는다.
                note = f"max\\|Δ\\| 중앙 {r['delta_peak_median']} (LSB)"
            elif r.get("ch_quant"):
                q = [v for v in r["ch_quant"] if v]
                note = f"양자화 {min(q)}" if q else ""
            out.append(
                f"| `{r['topic']}` | {r['verdict']} | {r['msg_hz']} | "
                f"{r['change_hz_significant']} | {r.get('change_hz_active', '-')} | "
                f"{r['dup_pct']}% | "
                f"{len(r['ch_static'])}/{r['channels']} | {note} |")
        out += [
            "",
            "- **갱신(전체 평균)** = idle 구간까지 포함한 평균 → 실제보다 낮게 보인다.",
            "- **갱신(활성)** = 1/변화간격 중앙값 → 값이 갱신될 때의 실제 rate. **이쪽을 볼 것.**",
            "",
        ]

    # ── 루프 rate + 힘 (deploy_ros2_exp [measure]) ──
    exp = run / "exp_stdout.log"
    if exp.exists():
        blocks, cfg = parse_exp(exp)
        if cfg:
            out += [f"## 배포 계측 — `[measure]` (과일={cfg['fruit']}, "
                    f"파지임계={cfg['grip_thr']}N, 스퀴즈임계={cfg['squeeze_thr']}N)", ""]
        else:
            out += ["## 배포 계측 — `[measure]`", ""]
        if blocks:
            out += [
                "| # | 유효 rate(Hz) | 수집(s) | valid/calls | steps (MIN_LEN) | thumb Fz(N) | 임계(N) | 도달률 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for i, b in enumerate(blocks):
                ratio = (100 * b["thumb"] / b["sq_thr"]) if b.get("sq_thr") else None
                out.append(
                    f"| {i} | {_fmt(b.get('rate'))} | {_fmt(b.get('dur'), 2)} | "
                    f"{b.get('valid','-')}/{b.get('calls','-')} | "
                    f"{b.get('steps','-')} ({b.get('min_len','-')}) | "
                    f"{_fmt(b.get('thumb'), 2)} | {_fmt(b.get('sq_thr'), 1)} | "
                    f"{_fmt(ratio, 0)}% |")
            rates = [b["rate"] for b in blocks if "rate" in b]
            thumbs = [b["thumb"] for b in blocks if "thumb" in b]
            out.append("")
            if rates:
                out.append(f"- 유효 rate 평균 **{st.fmean(rates):.1f} Hz** "
                           f"(학습 기준 100Hz, n={len(rates)})")
            if thumbs:
                out.append(f"- thumb 최대 Fz 평균 **{st.fmean(thumbs):.2f} N** (n={len(thumbs)})")
            out.append("")
        else:
            out += ["_`[measure]` 블록 없음 — 스퀴즈까지 진행되지 않은 run._", ""]

    # ── 판정 (README/TROUBLESHOOTING F4 완료 기준) ──
    out += ["## 판정", ""]
    verdicts = []
    if sensor.exists():
        reps = json.loads(sensor.read_text())
        # 통합모델(USE_JKIN=False)은 kin 을 입력으로 쓰지 않으므로 FROZEN 이어도 무해하다.
        unused = {"/hand/right/kin": "통합모델은 USE_JKIN=False 로 미사용 → 무해"}
        frozen = [r["topic"] for r in reps if r["verdict"] == "FROZEN"]
        blocking = [t for t in frozen if t not in unused]
        if blocking:
            verdicts.append(
                f"❌ **센서 값이 갱신되지 않음**: {', '.join(blocking)} "
                "— 발행률은 정상이나 값이 고정. 소스(paxini_writer 등) 확인 전에는 "
                "힘·추론 결과 전부 무의미")
        for t in frozen:
            if t in unused:
                verdicts.append(f"⚠ {t} FROZEN — 단 {unused[t]}")
        live = [r["topic"] for r in reps if r["verdict"] == "LIVE"]
        if live:
            verdicts.append(f"✅ 센서 갱신 정상: {', '.join(live)}")
    if exp.exists():
        blocks, _ = parse_exp(exp)
        rates = [b["rate"] for b in blocks if "rate" in b]
        steps = [b["steps"] for b in blocks if "steps" in b]
        mins = [b["min_len"] for b in blocks if "min_len" in b]
        thumbs = [(b["thumb"], b["sq_thr"]) for b in blocks if "thumb" in b and "sq_thr" in b]
        if rates:
            avg = st.fmean(rates)
            verdicts.append(f"{'✅' if avg >= 85 else '❌'} 루프 유효 rate {avg:.1f}Hz "
                            f"({'학습 100Hz 에 정합' if avg >= 85 else '학습 대비 저하 → F4 #2 루프 경량화 검토'})")
        if steps and mins:
            ok = min(steps) >= max(mins)
            verdicts.append(f"{'✅' if ok else '❌'} downsample 스텝 최소 {min(steps)} "
                            f"(MIN_LEN={max(mins)}) — {'샘플 충분' if ok else 'FACTOR 정합 필요(F4 #1)'}")
        if thumbs:
            worst = min(t / s for t, s in thumbs)
            verdicts.append(f"{'✅' if worst >= 0.9 else '❌'} thumb Fz 도달률 최저 {100*worst:.0f}% "
                            f"— {'힘 임계 도달' if worst >= 0.9 else '힘 미도달 → F6 분기(힘-도달 curl / Path B)'}")
    if not verdicts:
        verdicts = ["_자동 판정할 데이터 없음._"]
    out += [f"- {v}" for v in verdicts] + [""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv[1:]] or sorted(Path("docs/rate_log").glob("*"))
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        print("계측 run 디렉토리가 없습니다. measure_update_rate.sh 를 먼저 실행하세요.",
              file=sys.stderr)
        return 1

    for run in runs:
        md = summarize(run)
        (run / "summary.md").write_text(md)
        print(md)
        print()

    if len(runs) > 1:      # 여러 run 통합 비교표 (파일로는 저장하지 않음)
        print("## run 통합 비교", "", sep="\n")
        print("| run | 루프 rate 평균(Hz) | thumb Fz 평균(N) |")
        print("|---|---|---|")
        for run in runs:
            exp = run / "exp_stdout.log"
            if not exp.exists():
                continue
            blocks, _ = parse_exp(exp)
            rates = [b["rate"] for b in blocks if "rate" in b]
            thumbs = [b["thumb"] for b in blocks if "thumb" in b]
            print(f"| {run.name} | {_fmt(st.fmean(rates) if rates else None)} | "
                  f"{_fmt(st.fmean(thumbs) if thumbs else None, 2)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
