#!/usr/bin/env python3
"""test_ecoflex_engine_offline.py — 로봇 없이 엔진↔학습 파이프라인 패리티 검증.

ecoflex 원본 데모 h5 하나를 골라 같은 구간을 두 경로로 통과시킨다:
  기준: deep_ws data_preprocessing.read_demo → segment_to_sample → 정규화 → 모델
  엔진: 같은 원시 프레임(카운트·kin·paxini raw)을 가짜 브리지로 재생 →
        EcoflexPropertyEngine (capture_baseline → add_sample → infer)
전 채널(67ch 리샘플 시퀀스)과 최종 3타깃 예측이 일치해야 배포 전처리가 옳다.

실행 (로봇/ROS 불필요 — conda fruit 파이썬):
  /home/yesol/miniforge3/envs/fruit/bin/python \
      stiffness_deploy_ros2/launch/test_ecoflex_engine_offline.py [demo.h5 ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_LAUNCH = Path(__file__).resolve().parent
_BUNDLE = _LAUNCH.parent / "models" / "ecoflex2fruit"
DEEP_WS = os.environ.get("ECO_DEEP_WS", "/home/yesol/deep_ws/src/ecoflex2fruit")

# ── 대상 모델 선택 (ECO_MODEL=champ|anchor|rc|gru — 엔진과 동일 키) ──────────
_VARIANT = os.environ.get("ECO_MODEL", "champ").lower()
_FILES = {"champ": "Champ_repair_s42.pth", "anchor": "Anchor_s42.pth",
          "rc": "RC_v2_5_s42.pth", "gru": "gru_anchor_s42.pth"}

# ── 학습 환경 재현 ───────────────────────────────────────────────────────────
#    train_env 에는 EXP_COND 만 있고 전처리 env(SEQ_GROUPS 등)는 빌드 시점 것이라,
#    ckpt 의 config **스냅샷**에서 전처리 관련 상수를 config 모듈에 직접 주입한다.
_ck = torch.load(str(_BUNDLE / _FILES[_VARIANT]), map_location="cpu",
                 weights_only=False)
os.environ.update(_ck.get("train_env", {}))
os.environ.setdefault("LABEL_DIR",
                      "/home/yesol/deep_ws/data/ecoflex_new/object_labels_oldstif")

sys.path.insert(0, DEEP_WS)
import config as C                    # noqa: E402

_SNAP_KEYS = [
    "SEQ_GROUPS", "FORCE_CHANNELS", "SAT_MODE", "SAT_REPAIR_MAX",
    "TACTILE_SAT_FRAC", "TACTILE_SAT_MAX", "STIF_LOG", "RESAMPLE_MODE",
    "RESAMPLE_LEN", "SEG_STEPS", "USE_ONLY_SQUEEZE_ON", "FORCE_DELTA",
    "REP_FT", "REP_RES", "REP_JOINT", "JOINT_ABS", "JOINT_DELTA",
    "USE_JOINT_ERR", "SCALARS", "MIN_FRAMES", "PAD_FRAMES",
    "FALLBACK_BASELINE_FRAMES", "JOINT_SCALE", "SQUEEZE_CHANNEL",
    "PRELOAD_DRIFT_CORRECT",
]
_snap = _ck.get("config", {})
for _k in _SNAP_KEYS:
    if _k in _snap:
        setattr(C, _k, _snap[_k])

import data_preprocessing as DP       # noqa: E402
from ecoflex_engine import EcoflexPropertyEngine, JOINT_SCALE  # noqa: E402


# ── 가짜 브리지: 저장된 프레임을 순서대로 재생 ──────────────────────────────
class _Msg:
    def __init__(self, joint_counts, kin):
        self.j_pos = [list(joint_counts)]
        self.j_kin = [np.asarray(kin, np.float32).reshape(4, 3)]


class FakeBridges:
    """joint(counts)·ft·paxini raw 프레임 열을 shm/paxini 브리지 인터페이스로 재생."""

    def __init__(self, joint_counts, ft, raw):
        self.j, self.f, self.r = joint_counts, ft, raw
        self.i = -1

    def step(self):
        self.i = min(self.i + 1, len(self.j) - 1)

    # shm.read()
    def read(self):
        return _Msg(self.j[self.i], self.f[self.i])

    # paxini.read() — (tac, ts, valid, seq)
    class _Pax:
        def __init__(self, outer):
            self.o = outer

        def read(self):
            return (self.o.r[self.o.i], 0, 1, self.o.i)

    def pax(self):
        return FakeBridges._Pax(self)


def run_one(path: str, engine: EcoflexPropertyEngine, model, stats, norm) -> bool:
    groups = DP.read_demo(path, segments=["squeeze_A"])
    if not groups:
        print(f"  [skip] squeeze_A 구간 없음: {path}")
        return True
    g = groups[0]
    import copy
    g_ref = copy.deepcopy(g)          # segment_to_sample 은 despike 로 arrays 를 고친다
    sample = DP.segment_to_sample(g_ref)
    if sample is None or sample == "saturated":
        print(f"  [skip] 기준 파이프라인이 샘플 거부({sample}): {path}")
        return True

    # ── 엔진 재생: 같은 창의 원시 프레임 (joint 는 rad→counts 역변환) ──
    arrays, ph = g["arrays"], g["phases"]
    joint_counts = arrays["joint"] / JOINT_SCALE
    ft = arrays["ft"]
    raw = arrays["paxini"].reshape(len(ft), 4, 127, 3)
    b0, b1 = ph["pre"]
    if b1 <= b0:                       # 기준 코드의 pre 폴백과 동일
        b0, b1 = ph["on"][0], ph["on"][0] + C.FALLBACK_BASELINE_FRAMES

    fb = FakeBridges(joint_counts, ft, raw)
    pax = fb.pax()
    # baseline: pre 구간 프레임을 정확히 그 개수만큼 재생 (hz 를 크게 → sleep 무시 수준)
    fb.i = b0 - 1
    n_pre = b1 - b0
    engine.capture_baseline(_Stepper(fb), pax, sec=n_pre / 1000.0, hz=1000.0)
    # 스퀴즈 창(②③ on-window)을 순서대로 적재
    engine.reset()
    fb.i = ph["on"][0] - 1
    for _ in range(ph["on"][0], ph["on"][1]):
        fb.step()
        engine.add_sample(fb, pax)
    res = engine.infer()

    # ── 기준 예측: 저장 파이프라인 산출 seq/scalars → 같은 ckpt 로 forward ──
    x = (sample["seq"] - stats["seq_mean"].reshape(1, -1)) / (stats["seq_std"].reshape(1, -1) + 1e-8)
    sc = np.concatenate([sample["scalars"], [1.0]]).astype(np.float32)  # +is_ecoflex
    scz = (sc - stats["sc_mean"]) / (stats["sc_std"] + 1e-8)
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(scz, dtype=torch.float32).unsqueeze(0))[0].numpy()
    ref = {}
    for i, t in enumerate(("mass", "size", "stif")):
        lo, hi = norm[t]
        if t == "stif" and C.STIF_LOG and lo > 0:
            llo, lhi = np.log10(lo), np.log10(hi)
            ref[t] = float(10.0 ** (out[i] * (lhi - llo) + llo))
        else:
            ref[t] = float(out[i] * (hi - lo) + lo)

    # ── 채널 단위 비교 (엔진 내부 조립 vs 기준 seq) ──
    eng_seq = engine._assemble()
    dch = np.abs(eng_seq - sample["seq"]).max(axis=0)     # (67,)
    dmax = float(dch.max())
    scale = np.abs(sample["seq"]).max(axis=0) + 1e-6
    rel = float((dch / scale).max())
    ok_seq = rel < 1e-3
    ok_pred = res is not None and all(
        abs(res[t] - ref[t]) <= max(1e-3, 1e-3 * abs(ref[t])) for t in ("mass", "size", "stif"))

    name = os.path.basename(path)
    print(f"  {name}: seq Δmax={dmax:.3e} (상대 {rel:.2e}) "
          f"{'OK' if ok_seq else '**불일치**'}")
    if res is None:
        print("    엔진 추론 실패 (샘플 부족)")
        return False
    for t in ("mass", "size", "stif"):
        mark = "OK" if abs(res[t] - ref[t]) <= max(1e-3, 1e-3 * abs(ref[t])) else "**불일치**"
        print(f"    {t:4s}: 엔진 {res[t]:9.3f}  vs 기준 {ref[t]:9.3f}   {mark}")
    return ok_seq and ok_pred


class _Stepper:
    """capture_baseline 의 매 read 마다 프레임을 전진시키는 shm 래퍼."""

    def __init__(self, fb):
        self.fb = fb

    def read(self):
        self.fb.step()
        return self.fb.read()


def main() -> None:
    demos = sys.argv[1:]
    if not demos:
        allp = [d["path"] for d in DP.find_demos()]
        if not allp:
            raise SystemExit("데모 h5 를 못 찾음 — 경로를 인자로 주세요.")
        demos = [allp[0], allp[len(allp) // 2], allp[-1]]     # 앞/중간/끝 3개
    print(f"[test] 데모 {len(demos)}개로 패리티 검증 — 모델 {_VARIANT} ({_FILES[_VARIANT]})")

    engine = EcoflexPropertyEngine(variant=_VARIANT, device="cpu", is_ecoflex=1.0)
    stats = {k: np.asarray(v, np.float32).reshape(-1) for k, v in _ck["stats"].items()
             if k in ("seq_mean", "seq_std", "sc_mean", "sc_std")}
    ok = True
    for p in demos:
        ok = run_one(p, engine, engine.model, stats, _ck["target_norm"]) and ok
    print("\n[test] " + ("전부 일치 — 배포 전처리 = 학습 전처리 ✓"
                         if ok else "불일치 있음 — 위 채널/타깃 로그 확인 ✗"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
