#!/usr/bin/env python3
"""verify_parity.py — P5: Option 1(in-loop) HDF5 ↔ bag→HDF5(Option 2a) 대조.

무엇을 확인하나:
  같은 수집 세션의 두 산출물이 '동일한 모델 입력 프레임'인지 데모·프레임 단위로 대조한다.
    A = collect_ros2 가 실시간으로 쓴 HDF5 (Option 1)
    B = bag_to_hdf5 가 rosbag 재생으로 만든 HDF5 (Option 2a)
  두 파일 모두 recording_engine.HDF5DemoWriter 스키마(데모별 그룹) 이므로 직접 비교 가능.

정렬(프레임 수가 살짝 다를 때):
  q_target tick 재현이 완벽하면 프레임이 1:1 로 맞아 lag=0·오차≈0 이어야 한다.
  타이밍 edge 로 앞뒤 몇 프레임이 밀릴 수 있어, joint(정수·구별력 큼)를 기준으로
  [-max_lag, max_lag] 최적 lag 를 찾아 겹치는 구간에서 채널별 오차를 보고한다.

판정(데모별):
  프레임 단위로 채널별 최대오차를 보고, 채널마다 셋 중 하나로 분류한다.
    exact  — 모든 프레임이 |Δ| <= tol            → PASS
    timing — tol 초과 프레임이 (a) 소수(<= max_bad_frac) 이고 (b) 그 오차가
             '신호 자체의 1 샘플 스텝'(= median|diff(B)|) 의 step_k 배 이내
                                                  → PASS*  (샘플 시점 차, 데이터 불일치 아님)
    fail   — 그 외                                → FAIL

  왜 절대 tol 만으로는 안 되나(todolist 9번): joint·ft(kin) 은 스케일 안 된 raw 단위라
  신호 자체가 프레임당 joint≈4 counts / ft≈116 씩 움직인다. 절대 tol=1e-3 은 물리적으로
  도달 불가라 두 채널은 구조적으로 FAIL 이 되고, 그러면 '진짜 불일치'와 '시점 1스텝 차'를
  구분할 수 없다(= 진짜 회귀를 놓친다). 그래서 스텝 정규화 + 불일치 프레임 수로 판정한다.
  --exact 를 주면 timing 도 FAIL 로 취급한다(bit-exact 회귀 감시용).

  step_k=2.0 의 근거(임의값 아님):
    A↔B 불일치의 실체는 **원본 메시지 1개(5ms) 만큼의 staleness** 다. 라이브(A)는 tick 순간
    executor 가 아직 처리하지 못한 메시지를 못 보고, bag(B)은 도착시각 기준이라 그 메시지를
    포함한다(tools/check_parity_timing.py 로 A 값이 bag 원본 스트림에 실제로 존재함을 확인).
    그런데 step=median|diff(B)| 는 **프레임 격자(≈10.2ms)** 에서 재므로, 힘이 급변하는
    프레임에서는 '메시지 1개 변화'가 중앙값의 ~2배까지 커진다. 실측 교정:
      정상 2세션(160837·193657) 불일치 프레임의 스텝비 최대 = 1.72
      인위 교란 4종 중 '스텝비로 잡아야 하는' 최소값        = 8.00 (3프레임 +10스텝)
    → 1.72 < k < 8.00 이면 양쪽을 가른다. k=2.0 (정상 최대의 1.16배, 교란 최소의 1/4).
    나머지 교란 2종(전 프레임 +1스텝 / ft ×1.01)은 스텝비가 0.8~1.98 로 작지만
    불일치 프레임이 100% 라 **프레임 수 기준**이 잡는다 — 두 기준을 함께 유지해야 한다.
    ※ 국소 스텝(해당 프레임 주변 max|diff|)은 쓰면 안 된다: 교란이 B 자신의 분모를
      부풀려 3프레임 교란의 국소스텝비가 1.00 이 되고 정상과 구분되지 않는다(실측).
    ※ 위 수치는 step 을 median 으로 재던 시절의 교정값이다. 이후 정적 채널
      (tactile/resultant) 오탐 때문에 **p90 으로 교체**했다 — 재교정 근거는
      frame_judge 의 docstring 참고(정상 최대 1.25 vs 교란 최소 4.13).

실행:
  python3 stiffness_deploy_ros2/launch/verify_parity.py A.h5 B.h5
  python3 .../verify_parity.py collect_..._.h5 from_bag.h5 --tol 1e-3 --max-lag 5
종료코드: 0=PASS, 1=FAIL (스크립트/CI 용).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import h5py

CHANNELS = ["joint", "ft", "resultant", "tactile"]   # squeeze_on/valid/t_mono_ns 는 제외(자명/시계)
REF = "joint"                                         # 정렬 기준 채널


def load_demo(g):
    d = {}
    for c in CHANNELS:
        if c in g:
            d[c] = np.asarray(g[c][:], dtype=np.float64)
    return d


def load_time(g):
    """샘플 시각[ns] → float64 초. 없으면 None."""
    for k in ("t_mono_ns", "t_ns", "stamp_ns"):
        if k in g:
            return np.asarray(g[k][:], dtype=np.float64) / 1e9
    return None


def align_by_time(ta, tb, max_dt=0.005):
    """A 의 각 샘플을 B 의 '가장 가까운 시각' 샘플에 대응(무손실 정렬).

    index+lag 정렬은 A·B 의 기록 창(window)이 다르면 원리적으로 맞출 수 없다.
    실제로 라이브(A)는 스퀴즈 모션 구간만(1.52s/150), bag(B)은 segment 라벨 창
    전체(3.04s/300)를 담아 길이가 2배다. 두 파일 모두 샘플 시각을 갖고 있으므로
    **시각으로 짝지어 겹치는 부분만** 비교하는 것이 P5 의 질문("같은 데이터가
    bag 에서 재현되는가")에 맞는 정렬이다.

    반환: (ia, ib, max_offset_s) — 매칭된 A/B 인덱스 배열과 최대 시각차.
    """
    if ta is None or tb is None or len(ta) == 0 or len(tb) == 0:
        return None, None, None
    # 시계 기준 불일치 감지: A 는 time.monotonic_ns(부팅 이후), B(bag)는 epoch 를
    # 같은 't_mono_ns' 이름으로 저장한다 → 절대시각 짝지음이 원리적으로 불가.
    if abs(float(np.median(tb)) - float(np.median(ta))) > 1e6:
        return None, None, "clock_mismatch"
    idx = np.searchsorted(tb, ta)
    lo = np.clip(idx - 1, 0, len(tb) - 1)
    hi = np.clip(idx, 0, len(tb) - 1)
    pick = np.where(np.abs(tb[lo] - ta) <= np.abs(tb[hi] - ta), lo, hi)
    off = np.abs(tb[pick] - ta)
    keep = off <= max_dt
    return np.nonzero(keep)[0], pick[keep], (float(off[keep].max()) if keep.any() else None)


def aligned(na, nb, d):
    """a[i] 를 b[i+d] 에 대응. 겹치는 (a0, b0, n) 반환."""
    a0, b0 = (0, d) if d >= 0 else (-d, 0)
    n = min(na - a0, nb - b0)
    return a0, b0, max(0, n)


def ref_err(a, b, d):
    a0, b0, n = aligned(len(a), len(b), d)
    if n <= 0:
        return np.inf
    x = a[a0:a0 + n].reshape(n, -1)
    y = b[b0:b0 + n].reshape(n, -1)
    return float(np.mean(np.abs(x - y)))


def best_lag(a, b, max_lag):
    lags = range(-max_lag, max_lag + 1)
    errs = [(d, ref_err(a, b, d)) for d in lags]
    return min(errs, key=lambda e: e[1])   # (lag, ref_mean_err)


BAD = {"exact": 0, "timing": 1, "fail": 2}       # 나쁜 순서(리포트에서 최악 채널 뽑기용)


def frame_judge(x, y, *, tol, step_k, max_bad_frac):
    """정렬된 A/B (n,m) 을 프레임 단위로 판정. todolist 9번의 (b) 스텝 정규화.

    step = **p90**(|diff(B)|) = 'B 가 한 프레임 사이에 움직일 때 움직이는 양'.
    A·B 가 같은 신호를 1 샘플 어긋나 뽑았다면 오차는 이 스케일이 된다.

    ★ 왜 median 이 아니라 p90 인가 (2026-07-28 실측 교정):
      median 은 **거의 변하지 않는 채널에서 0 에 수렴**한다. tactile/resultant 는 대부분
      프레임이 그대로라 median≈0 → 진짜 1 메시지 시점차도 배수가 3~10 으로 폭발해
      정상 세션이 FAIL 로 오탐됐다(세션 224840). p90 은 '움직일 때의 변화량' 을 재므로
      정적 채널에서도 무너지지 않는다. 같은 세션 + 교란 4종 실측:
        정규화   정상 최대   교란 최소(스텝으로 잡아야 하는 것)   판정
        /median   5.00        10.00                              여유 좁음(k=2.0 오탐)
        /p90      1.25         4.13                              ✔ 3.3배 여유
        /max      1.00         0.83                              ✘ 역전(사용 불가)
      → p90 + step_k=2.0. (교란 'tactile×1.01' 은 /p90 이 1.00 으로 작지만 불일치
        프레임이 100% 라 **프레임 수 기준**이 잡는다 — 두 기준을 함께 유지해야 한다.)
      p90 이 0 이면(B 가 완전 상수) 시점차로 설명될 수 없으므로 아래 step>0 조건에서 fail 이 된다.
    """
    n = len(x)
    if n == 0:
        return {"max": np.nan, "n_bad": 0, "n": 0, "step": 0.0, "ratio": np.nan,
                "status": "fail"}
    per = np.abs(x - y).max(axis=1)                                  # 프레임별 최대오차
    bad = np.nonzero(per > tol)[0]
    step = (float(np.percentile(np.abs(np.diff(y, axis=0)).max(axis=1), 90))
            if n >= 2 else 0.0)
    ratio = float(per[bad].max() / step) if (len(bad) and step > 0) else 0.0
    if len(bad) == 0:
        status = "exact"
    elif step > 0 and ratio <= step_k and len(bad) <= max(1, int(max_bad_frac * n)):
        status = "timing"
    else:
        status = "fail"
    return {"max": float(per.max()), "n_bad": int(len(bad)), "n": n,
            "step": step, "ratio": ratio, "status": status}


def attr(f, k, default="?"):
    v = f.attrs.get(k, default)
    return v.decode() if isinstance(v, bytes) else v


def main():
    ap = argparse.ArgumentParser(description="P5 parity: Option1 HDF5 ↔ bag→HDF5")
    ap.add_argument("file_a", help="A = collect_ros2 (Option 1) HDF5")
    ap.add_argument("file_b", help="B = bag_to_hdf5 (Option 2a) HDF5")
    ap.add_argument("--tol", type=float, default=1e-3, help="채널 max_abs_err 허용치")
    ap.add_argument("--frame-tol", type=int, default=2,
                    help="(미사용) A⊂B 는 설계상 정상이라 Δn 으로 판정하지 않는다")
    ap.add_argument("--max-lag", type=int, default=0,
                    help="정렬 lag 탐색 범위(0=자동: |Δn|+10)")
    ap.add_argument("--time-tol", type=float, default=0.005,
                    help="시각정렬 시 짝지음 허용 시간차[s]")
    ap.add_argument("--index-align", action="store_true",
                    help="시각정렬 대신 기존 index+lag 정렬 강제")
    ap.add_argument("--step-k", type=float, default=2.0,
                    help="스텝 정규화 배수: 오차가 median|diff(B)| 의 이 배 이내면 '시점 차'")
    ap.add_argument("--max-bad-frac", type=float, default=0.05,
                    help="'시점 차' 로 허용할 불일치 프레임 비율 상한")
    ap.add_argument("--exact", action="store_true",
                    help="bit-exact 강제: 시점 차(timing)도 FAIL 로 취급")
    args = ap.parse_args()

    fa = h5py.File(args.file_a, "r")
    fb = h5py.File(args.file_b, "r")

    print("=" * 78)
    print(f"A: {args.file_a}")
    print(f"   demos={attr(fa,'n_demos')}  paxini={attr(fa,'paxini_source')}  "
          f"FACTOR={attr(fa,'FACTOR')}  USE_TACTILE={attr(fa,'USE_TACTILE')}")
    print(f"B: {args.file_b}")
    print(f"   demos={attr(fb,'n_demos')}  paxini={attr(fb,'paxini_source')}  "
          f"tick={attr(fb,'tick_mode')}")
    if attr(fa, "paxini_source") != attr(fb, "paxini_source"):
        print("  ⚠ paxini_source 불일치 → resultant/tactile 은 당연히 다를 수 있음(설계상).")
    # A(라이브)는 t_mono_ns=monotonic, B(bag)는 t_ns=epoch 로 시계가 다르다.
    # collect 가 root attr 로 남긴 t_offset_ns 로 A 를 epoch 로 환산해 짝짓는다:
    # epoch = t_mono_ns + t_offset_ns. attr 없는 구파일은 0 → 기존 clock_mismatch 경로로 폴백.
    t_off_s = float(attr(fa, "t_offset_ns", 0)) / 1e9
    if t_off_s:
        print(f"  ℹ A t_offset_ns={int(t_off_s * 1e9)} 로 monotonic→epoch 환산 후 시각정렬")
    print("=" * 78)

    # 그룹 수집: 9단계 시퀀스 도입(7048c84) 이후 그룹명은 '{segment}__run{NNN}' 이다.
    # 이 스크립트는 그보다 먼저 작성돼(fa9eccd) 'demo_' 접두사를 찾았기 때문에 항상
    # common=0 → FAIL 이었다. 데이터셋을 가진 그룹을 전부 대상으로 하고, 구버전
    # 'demo_*' 네이밍도 계속 지원한다.
    def _groups(f):
        return {n for n, o in f.items() if isinstance(o, h5py.Group) and len(o) > 0}

    demos_a, demos_b = _groups(fa), _groups(fb)
    common = sorted(demos_a & demos_b)
    only_a, only_b = sorted(demos_a - demos_b), sorted(demos_b - demos_a)
    if only_a:
        print(f"⚠ A 에만 있는 데모: {only_a}")
    if only_b:
        print(f"ℹ B 에만 있는 구간(설계상 bag 전용): {only_b}")

    hdr = f"{'demo':<9}{'n_A':>6}{'n_B':>6}{'Δn':>5}{'lag':>5}  "
    hdr += "  ".join(f"{c[:5]}_max" for c in CHANNELS) + "   verdict"
    print(hdr)
    print("-" * len(hdr))
    # 채널별 판정 근거(불일치 프레임 수·스텝 배수)를 표 아래 줄에 남긴다 — max 만으로는
    # '진짜 불일치'와 '시점 1스텝 차'가 구분되지 않는다(todolist 9번).
    detail_lines = []

    # only_b(예: move_palm_down)는 설계상 bag 전용이라 실패 사유가 아니다
    # (A root attr note: 'HDF5 groups = squeeze ★ (A,B) only; palm_down ... in bag').
    all_pass = bool(common) and not only_a
    for name in common:
        A, B = load_demo(fa[name]), load_demo(fb[name])
        chans = [c for c in CHANNELS if c in A and c in B]
        na = len(A[REF]) if REF in A else (len(next(iter(A.values()))) if A else 0)
        nb = len(B[REF]) if REF in B else (len(next(iter(B.values()))) if B else 0)
        dn = nb - na

        # 정렬: 양쪽에 샘플 시각이 있으면 '시각 기준'(창 길이가 달라도 유효),
        #       없으면 기존 index+lag 폴백.
        ia, ib, off = (None, None, None)
        if not args.index_align:
            ta = load_time(fa[name])
            if ta is not None:
                ta = ta + t_off_s            # A: monotonic → epoch (B 와 같은 시계)
            ia, ib, off = align_by_time(ta, load_time(fb[name]),
                                        max_dt=args.time_tol)

        maxes, judged = {}, {}
        if ia is not None and len(ia) > 0:
            lag = 0
            n_match = len(ia)
            # 창 길이 차이는 시각정렬에서 정상(설계상 bag 창이 더 넓다) → Δn 으로 실패시키지 않는다.
            ok = n_match >= max(1, int(0.9 * na))     # A 샘플의 90% 이상이 짝을 찾아야 함
            for c in chans:
                r = frame_judge(A[c][ia].reshape(n_match, -1),
                                B[c][ib].reshape(n_match, -1),
                                tol=args.tol, step_k=args.step_k,
                                max_bad_frac=args.max_bad_frac)
                maxes[c], judged[c] = r["max"], r
            note = f"t정렬 {n_match}/{na}쌍 Δt≤{off*1e3:.2f}ms" if off is not None else ""
        else:
            # 창 길이가 다르면(A ⊂ B) 필요한 lag 가 |Δn| 까지 커진다. 기본 5 로는
            # 절대 못 찾으므로 자동 확대한다(--max-lag 를 명시하면 그 값을 쓴다).
            ml = args.max_lag if args.max_lag > 0 else abs(dn) + 10
            if REF in A and REF in B and na > 0 and nb > 0:
                lag, _ = best_lag(A[REF], B[REF], ml)
            else:
                lag = 0
            # A ⊂ B (bag 창이 더 넓음)는 설계상 정상 → Δn 자체로 실패시키지 않고,
            # 겹친 구간의 채널 오차로만 판정한다.
            ok = True
            for c in chans:
                a0, b0, n = aligned(len(A[c]), len(B[c]), lag)
                if n <= 0:
                    maxes[c] = np.nan
                    judged[c] = {"status": "fail", "n_bad": 0, "n": 0, "ratio": np.nan}
                    continue
                r = frame_judge(A[c][a0:a0 + n].reshape(n, -1),
                                B[c][b0:b0 + n].reshape(n, -1),
                                tol=args.tol, step_k=args.step_k,
                                max_bad_frac=args.max_bad_frac)
                maxes[c], judged[c] = r["max"], r
            note = f"index정렬 lag={lag} (범위±{ml})"
            if off == "clock_mismatch":
                note += " ※A=monotonic/B=epoch 로 시각정렬 불가"

        # 채널 판정 합산: fail 이 하나라도 있으면 FAIL, timing 만 있으면 PASS*(시점 차).
        worst = max((judged[c]["status"] for c in judged), key=lambda s: BAD[s],
                    default="fail")
        if args.exact and worst == "timing":
            worst = "fail"                        # bit-exact 강제 모드
        ok = ok and worst != "fail"
        timing = [c for c in judged if judged[c]["status"] == "timing"]
        if timing:
            label = "bit-exact 위반(--exact)" if args.exact else "시점 차로 판정된 채널"
            detail_lines.append(
                f"    {name}: {label} — " + ", ".join(
                    f"{c} {judged[c]['n_bad']}/{judged[c]['n']}프레임"
                    f"(≤{judged[c]['ratio']:.2f}스텝)" for c in timing))
        failed = [c for c in judged if judged[c]["status"] == "fail"]
        if failed:
            detail_lines.append(
                f"    {name}: 불일치 채널 — " + ", ".join(
                    f"{c} {judged[c]['n_bad']}/{judged[c]['n']}프레임"
                    f"({judged[c]['ratio']:.1f}스텝)" for c in failed))

        cells = "  ".join(f"{maxes.get(c, float('nan')):9.3g}" for c in CHANNELS)
        verdict = "FAIL" if not ok else ("PASS*" if timing else "PASS")
        all_pass = all_pass and ok
        print(f"{name:<9}{na:>6}{nb:>6}{dn:>5}{lag:>5}  {cells}   {verdict}  {note}")

    print("-" * len(hdr))
    for ln in detail_lines:
        print(ln)
    verdict = "PASS" if all_pass else "FAIL"
    print(f"OVERALL: {verdict}  (common demos={len(common)}, tol={args.tol}, "
          f"step_k={args.step_k}, max_bad_frac={args.max_bad_frac}"
          f"{', exact' if args.exact else ''})")
    if all_pass:
        print("  PASS* = tol 초과 프레임이 소수이고 그 오차가 신호 자체의 1 샘플 스텝 이내 "
              "→ 샘플 시점 차(데이터 불일치 아님). bit-exact 를 요구하려면 --exact.")
    else:
        print("  → FAIL 원인 후보: (1) q_target tick ≠ add_sample tick 미세차, "
              "(2) valid 필터 edge, (3) ft mN /tmp 파일 사용(bag 미포함), "
              "(4) paxini_source 불일치. 위 '불일치 채널' 의 프레임 수·스텝 배수로 판단: "
              "스텝 배수가 크거나 불일치 프레임이 많으면 시점 차가 아니라 실제 불일치.")

    fa.close()
    fb.close()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
