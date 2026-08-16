#!/usr/bin/env python3
"""ecoflex_engine.py — ecoflex2fruit 3타깃(mass·size·stif) 실시간 추론 엔진 (demo용).

deep_ws/src/ecoflex2fruit 의 챔피언(README2 §8-2 `Champ_repair` · 변형2+3+5 67ch)
을 배포에서 돌린다. 전처리는 학습 파이프라인(data_preprocessing.segment_to_sample)
을 **글자 단위로 복제**했고, 모델 클래스·파생 채널 계산(변형3 resul_curved ·
변형5 contact)은 deep_ws 원본의 **vendored 사본**(launch/eco_model.py ·
launch/eco_paxini_features.py — 출처 커밋 주석 참고)을 쓴다 → **deep_ws 없이
자립 실행 가능**. deep_ws 쪽 원본이 바뀌면 사본을 갱신하고
test_ecoflex_engine_offline.py (개발 머신 전용 — 이 테스트만 deep_ws 필요)로
패리티를 재확인한다.

학습 정합 규약 (검증: tools/test_ecoflex_engine_offline.py — 저장 데이터셋과 대조):
  · 시퀀스 = squeeze_on(스퀴즈+hold) 구간만, avgpool 로 32스텝 리샘플
  · 채널 순서 = joint_abs(16) · joint_delta(16) · ft(12) · resul_curved(12) ·
    contact(9) · contact_d(2)  — CHANNEL_ORDER 유도, ckpt 의 config 스냅샷과 일치
  · baseline(①pre_wait) = 파지 유지 구간 프레임 평균 — joint_delta·힘 Δ·contact Δ맵의
    영점. 엔진의 capture_baseline() 이 스퀴즈 직전에 모은다.
  · 포화 글리치 = 학습과 같은 repair(이웃 보간 · SAT_MODE=repair 정합)
  · 정규화·denorm = ckpt 내장 stats/target_norm (STIF_LOG 는 log 공간 역변환)

인터페이스는 기존 StiffnessInferenceEngine 과 동일 duck-type:
  reset() / add_sample(shm, paxini) / infer()
  + capture_baseline(shm, paxini, sec)  ← demo 파일이 스퀴즈 직전에 호출 (신규)
paxini 는 **raw 브리지**(Ros2RawPaxiniBridge — /paxini/right/raw 4×127×3)여야 한다.
point0 트릭 브리지(ft)로는 변형3·5 를 계산할 수 없다.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── vendored 사본 import (deep_ws 불필요 — 같은 launch/ 디렉토리) ───────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eco_model import MultiTargetRegressorAux      # noqa: E402  (torch 만 의존)
import eco_paxini_features as PF                   # noqa: E402  (numpy 만 의존)

_BUNDLE = Path(__file__).resolve().parent.parent / "models" / "ecoflex2fruit"
#   실물 후보 4종 (README3 종합 판정의 축별 대표) — ECO_MODEL 로 선택:
#     champ  = 공식 배포 기준점 · anchor = 차기 챔피언 후보(§12b)
#     rc     = stif·입력 축 대표(RC_v2_5 — resultant 입력) · gru = 전이 축 대표
MODEL_FILES = {"champ": "Champ_repair_s42.pth", "anchor": "Anchor_s42.pth",
               "rc": "RC_v2_5_s42.pth", "gru": "gru_anchor_s42.pth"}

JOINT_SCALE = np.pi / 4096.0        # count → rad (학습 config.JOINT_SCALE 과 동일)
RESAMPLE_LEN = 32
SAT_ABS = 50.0                      # |resultant| 포화 판정 (deep_ws SAT 가드와 동일 자릿수)
MIN_FRAMES = 16                     # 이 미만이면 추론 불가 (스퀴즈 ~1.5s@100Hz ≈ 150)
#   채널 폭 (실제 조립 순서·구성은 ckpt 의 config 스냅샷에서 읽는다)
_DIMS = {"joint_abs": 16, "joint_delta": 16, "ft": 12, "resultant": 12,
         "resul_curved": 12, "contact": 9, "contact_d": 2}
#   힘 채널의 CHANNEL_ORDER (deep_ws config.SEQ_CHANNELS + PAX_DERIVED 순서)
_FORCE_ORDER = ("resultant", "ft", "resul_curved")


def _np(x, dtype=np.float32) -> np.ndarray:
    """ckpt 값(torch.Tensor · numpy · list 무엇이든)을 numpy 로. CUDA 텐서 방어."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype)


# ── ecoflex 개체 라벨 (실제값 대조용) ────────────────────────────────────────
#   deep_ws data/ecoflex_new/object_labels_oldstif 의 사본 (labels/ — 학습 라벨과
#   동일 파일). ⚠ 반드시 oldstif 판이어야 한다 — 챔피언 학습·정규화가 이 라벨 기준.
#   load_labels() 가 ckpt 의 target_norm 과 normalize 범위를 대조해 다른 판이면 경고.
DEFAULT_LABEL_DIR = str(Path(__file__).resolve().parent.parent
                        / "labels" / "object_labels_oldstif")
TARGETS = ("mass", "size", "stif")


def load_labels(label_dir: str = DEFAULT_LABEL_DIR):
    """{mass,size,stif}.yaml → (obj_targets, norm) — deep_ws load_labels 와 동일 규약.

    obj_targets: {obj_id(1-based): {"mass": g, "size": mm, "stif": ...}}
    norm:        {target: (min, max)}  (yaml 의 normalize 절)
    """
    import yaml
    norm, raw = {}, {}
    for t in TARGETS:
        with open(os.path.join(label_dir, f"{t}.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        norm[t] = (float(cfg["normalize"]["min"]), float(cfg["normalize"]["max"]))
        raw[t] = {int(k): float(v) for k, v in cfg["utils"]["dict"].items()}
    common = set.intersection(*[set(raw[t]) for t in TARGETS])
    obj = {oid: {t: raw[t][oid] for t in TARGETS} for oid in sorted(common)}
    return obj, norm


def nearest_specimens(res: dict, obj_targets: dict, norm: dict,
                      stif_log: bool = True, k: int = 3):
    """추론 dict → 라벨 공간 최근접 개체 [(obj_id, dist, {실제값}), ...] 상위 k.

    거리 = 3타깃 각각을 학습과 같은 방식(min–max, stif 는 log₁₀ 공간)으로 0~1
    정규화한 뒤의 L2 — 축 스케일 차이를 지운 대등 비교다.
    """
    def _n(t, v):
        lo, hi = norm[t]
        if t == "stif" and stif_log and lo > 0:
            llo, lhi = np.log10(lo), np.log10(hi)
            return (np.log10(max(v, 1e-9)) - llo) / (lhi - llo) if lhi > llo else 0.0
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    p = np.array([_n(t, float(res[t])) for t in TARGETS])
    scored = sorted(
        (float(np.linalg.norm(p - np.array([_n(t, lab[t]) for t in TARGETS]))),
         oid, lab)
        for oid, lab in obj_targets.items())
    return [(oid, d, lab) for d, oid, lab in scored[:k]]


def check_label_norm(engine, label_norm: dict) -> bool:
    """라벨 yaml 의 normalize 범위 ↔ ckpt target_norm 대조 — oldstif 오적용 방지.

    학습 정규화와 라벨 판이 다르면(예: 새 stif 라벨) 대조표가 조용히 틀어지므로
    불일치 시 경고를 찍고 False 를 반환한다(실행은 막지 않음 — 대조 표시용이므로)."""
    ok = True
    for t, (lo, hi) in engine.norm.items():
        llo, lhi = label_norm.get(t, (lo, hi))
        if abs(lo - llo) > 1e-6 or abs(hi - lhi) > 1e-6:
            print(f"[eco-engine] ⚠ 라벨 normalize({t}: {llo}~{lhi}) ≠ "
                  f"ckpt norm({lo}~{hi}) — oldstif 판이 맞는지 확인!")
            ok = False
    if ok:
        print("[eco-engine] 라벨 normalize = ckpt norm 일치 (oldstif 정합 확인)")
    return ok


def _avgpool(x: np.ndarray, L: int) -> np.ndarray:
    """(T,C)->(L,C). 학습 data_preprocessing._resample(avgpool) 과 동일."""
    T = x.shape[0]
    if T == L:
        return x.astype(np.float32)
    if T > L:
        edge = np.linspace(0, T, L + 1).astype(int)
        edge[-1] = T
        return np.stack([x[max(a, 0):max(b, a + 1)].mean(axis=0)
                         for a, b in zip(edge[:-1], edge[1:])]).astype(np.float32)
    old, new = np.linspace(0, 1, T), np.linspace(0, 1, L)
    return np.stack([np.interp(new, old, x[:, c]) for c in range(x.shape[1])],
                    axis=1).astype(np.float32)


class EcoflexPropertyEngine:
    """3타깃 추론 엔진. shm=Ros2ShmBridge · paxini=Ros2RawPaxiniBridge."""

    def __init__(self, variant: str | None = None, device: str | None = None,
                 is_ecoflex: float = 1.0):
        variant = (variant or os.environ.get("ECO_MODEL", "champ")).lower()
        ckpt_path = _BUNDLE / MODEL_FILES[variant]
        self.variant = variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        #   ckpt 는 항상 CPU 로 읽는다 — stats/target_norm 이 텐서로 저장된 ckpt 가 있고
        #   CUDA 텐서는 np.asarray/np.log10 에서 바로 터진다. 모델만 아래에서 device 로.
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        self.targets = list(ck.get("target_names", ["mass", "size", "stif"]))
        #   (lo, hi) 도 텐서일 수 있으므로 파이썬 float 로 고정
        self.norm = {t: (float(lo), float(hi))
                     for t, (lo, hi) in ck["target_norm"].items()}
        self.stif_log = bool(cfg.get("STIF_LOG", True))
        st = ck["stats"]
        self.seq_mean = _np(st["seq_mean"])
        self.seq_std = _np(st["seq_std"])
        self.sc_mean = _np(st["sc_mean"])
        self.sc_std = _np(st["sc_std"])
        self.is_ecoflex = float(is_ecoflex)

        # ── 채널 구성 = ckpt config 스냅샷 (모델마다 다르다 — RC_v2_5 는 resultant) ──
        groups = set(cfg.get("SEQ_GROUPS", []))
        #   힘 채널은 스냅샷의 FORCE_CHANNELS 순서(=CHANNEL_ORDER 유도)를 그대로 쓴다.
        self.fnames = list(cfg.get("FORCE_CHANNELS")
                           or [n for n in _FORCE_ORDER if n in groups])
        self.joint_abs = bool(cfg.get("JOINT_ABS", True)) and "joint_abs" in groups
        self.joint_delta = bool(cfg.get("JOINT_DELTA", True)) and "joint_delta" in groups
        self.want_contact = "contact" in groups
        self.want_contact_d = "contact_d" in groups
        self.force_delta = bool(cfg.get("FORCE_DELTA", True))
        order = ([n for n in ("joint_abs", "joint_delta")
                  if getattr(self, n)] + self.fnames
                 + [n for n in ("contact", "contact_d")
                    if getattr(self, f"want_{n}")])
        n_ch = sum(_DIMS[n] for n in order)
        assert n_ch == int(ck["in_channels"]) == self.seq_mean.reshape(-1).shape[-1], \
            f"채널 수 불일치: 조립 {order}={n_ch} vs ckpt {ck['in_channels']}"
        self.ch_order = order

        self.model = MultiTargetRegressorAux(
            in_channels=ck["in_channels"], n_scalars=ck["n_scalars"],
            hidden_dim=ck["hidden_dim"], n_targets=len(self.targets),
            dropout=0.0, temporal=ck.get("temporal", "transformer"),
            in_proj=ck.get("in_proj"), pool=ck.get("pool", "meanmax"),
            tf_layers=ck.get("tf_layers", 2), tf_nhead=ck.get("tf_nhead", 4),
            tf_ff_mult=ck.get("tf_ff_mult", 2.0),
            tf_posemb=ck.get("tf_posemb", "learned"),
            #   앵커 키는 §12b 이후 ckpt 에만 있다 — 없으면(None 포함) 0/기본값.
            anchor_n=ck.get("anchor_n") or 0, anchor_dh=ck.get("anchor_dh") or 48,
            anchor_tau=ck.get("anchor_tau") or 1.0).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

        self.geom = PF.load_pad_geometry(str(_BUNDLE / "sensors.json"))
        # GUI 표시용 라벨 범위 (물리 단위)
        self.norm_min, self.norm_max = float(self.norm["stif"][0]), float(self.norm["stif"][1])
        self.ranges = {t: (float(self.norm[t][0]), float(self.norm[t][1]))
                       for t in self.targets}
        self._base = None
        self.reset()
        n_par = sum(p.numel() for p in self.model.parameters())
        print(f"[eco-engine] {MODEL_FILES[variant]} 로드 — {n_par:,} 파라미터 · "
              f"device={self.device} · STIF_LOG={self.stif_log} · "
              f"is_ecoflex={self.is_ecoflex} · 채널 {n_ch} {order}")

    # ── 프레임 읽기 (bridge 인터페이스) ─────────────────────────────────────
    @staticmethod
    def _read_frame(shm, paxini):
        msg = shm.read()
        joint = np.asarray(msg.j_pos[0], np.float32)               # (16,) counts
        ft = np.asarray(msg.j_kin[0], np.float32).reshape(-1)[:12]  # (12,)
        tac, _ts, valid, _seq = paxini.read()                      # (4,127,3)
        return joint, ft, np.asarray(tac, np.float32), int(valid)

    # ── baseline (①pre_wait 등가) — 스퀴즈 직전 파지 유지 상태에서 호출 ─────
    def capture_baseline(self, shm, paxini, sec: float = 0.8, hz: float = 100.0):
        js, fs, ts = [], [], []
        n = max(1, int(sec * hz))
        for _ in range(n):
            j, f, t, v = self._read_frame(shm, paxini)
            if v == 1:
                js.append(j); fs.append(f); ts.append(t)
            time.sleep(1.0 / hz)
        if not js:
            print("[eco-engine] ⚠ baseline 프레임 0 — 영점 없이 진행(Δ채널 부정확)")
            self._base = None
            return 0
        raw = np.stack(ts)                                          # (n,4,127,3)
        self._base = {
            "joint": np.stack(js).mean(axis=0) * JOINT_SCALE,       # (16,) rad
            "Fz": raw[:, 0, :, 2].mean(axis=0).astype(np.float32),  # (127,) thumb 법선
        }
        #   힘 채널 영점 — 이 모델이 쓰는 채널만 (rc=resul_curved · resultant=Σ127)
        fchan = self._force_channels(np.stack(fs).astype(np.float32), raw)
        for n_, a in fchan.items():
            self._base[n_] = a.mean(axis=0).astype(np.float32)      # (12,)
        print(f"[eco-engine] baseline {len(js)}프레임 확보 (①pre_wait 등가)")
        return len(js)

    def _force_channels(self, ft: np.ndarray, raw: np.ndarray) -> dict:
        """이 모델의 FORCE_CHANNELS 만 계산해 {이름:(n,12)} 로 반환.
        ft=kin 그대로 · resultant=Σ127(taxel 합) · resul_curved=paxini_features 정본."""
        out = {}
        for n_ in self.fnames:
            if n_ == "ft":
                out[n_] = ft
            elif n_ == "resultant":
                out[n_] = raw.sum(axis=2).reshape(len(raw), -1).astype(np.float32)
            elif n_ == "resul_curved":
                out[n_] = PF.curved_resultant(raw, self.geom)[
                    "resul_curved"].reshape(len(raw), -1).astype(np.float32)
            else:
                raise KeyError(f"지원하지 않는 힘 채널: {n_}")
        return out

    # ── 기존 엔진 인터페이스 ────────────────────────────────────────────────
    def reset(self):
        """스퀴즈 한 번의 버퍼만 비운다 — baseline 은 유지(§ 학습의 ① vs ②③ 분리)."""
        self.buf = {"joint": [], "ft": [], "raw": []}

    def add_sample(self, shm, paxini) -> bool:
        j, f, t, v = self._read_frame(shm, paxini)
        if v != 1:
            return False                    # 학습의 valid 필터와 동일
        self.buf["joint"].append(j)
        self.buf["ft"].append(f)
        self.buf["raw"].append(t)
        return True

    # ── 학습 파이프라인 복제 ────────────────────────────────────────────────
    def _despike(self, raw: np.ndarray) -> np.ndarray:
        """포화 프레임(합력 |Σ| > SAT_ABS)을 이웃 정상 프레임 보간으로 수리 —
        학습 SAT_MODE=repair(despike_saturation) 정합. raw (n,4,127,3)."""
        res = raw.sum(axis=2)                                   # (n,4,3) Σ127
        bad = np.abs(res).max(axis=(1, 2)) > SAT_ABS            # (n,)
        if not bad.any():
            return raw
        good = np.flatnonzero(~bad)
        if len(good) < 2:
            return raw
        idx = np.flatnonzero(bad)
        flat = raw.reshape(len(raw), -1)
        flat[idx] = np.stack([np.interp(idx, good, flat[good, c])
                              for c in range(flat.shape[1])], axis=1)
        print(f"[eco-engine] 포화 프레임 {len(idx)}개 수리 (repair 정합)")
        return flat.reshape(raw.shape)

    def _assemble(self):
        joint = np.stack(self.buf["joint"]).astype(np.float32) * JOINT_SCALE  # (n,16) rad
        ft = np.stack(self.buf["ft"]).astype(np.float32)                      # (n,12)
        raw = self._despike(np.stack(self.buf["raw"]).astype(np.float32))     # (n,4,127,3)
        b = self._base or {"joint": joint[:1].mean(0), "Fz": raw[:1, 0, :, 2].mean(0)}
        fchan = self._force_channels(ft, raw)      # 이 모델의 힘 채널만 (despike 후 재계산)

        # 조립 순서 = self.ch_order (ckpt config 유도 — CHANNEL_ORDER 정합)
        seq = []
        if self.joint_abs:
            seq.append(joint)
        if self.joint_delta:
            seq.append(joint - b["joint"][None, :])
        for n_ in self.fnames:
            a = fchan[n_]
            base = b.get(n_)
            if base is None:                       # baseline 미확보 폴백 — 첫 프레임 영점
                base = a[:1].mean(0)
            seq.append(a - base[None, :] if self.force_delta else a)
        if self.want_contact or self.want_contact_d:
            # 변형5 — thumb(finger0) 법선 Δ맵 → 프레임별 접촉 특징 (학습과 동일 소스·순서)
            Fz = raw[:, 0, :, 2]
            dmap = np.clip(Fz - b["Fz"][None, :], 0, None)
            ct = np.stack([PF.contact_features(x, self.geom)
                           for x in dmap]).astype(np.float32)
            if self.want_contact:
                seq.append(ct)
            if self.want_contact_d:
                k = max(1, min(5, len(ct) // 4))
                area, nsum = ct[:, 0], ct[:, 8]
                dA = np.zeros_like(area); dF = np.zeros_like(nsum)
                dA[k:] = area[k:] - area[:-k]
                dF[k:] = nsum[k:] - nsum[:-k]
                dAdF = np.where(dF > 0.05, dA / np.where(dF > 0.05, dF, 1.0), 0.0)
                seq.append(np.stack([dA, dAdF], axis=1).astype(np.float32))
        M = np.concatenate(seq, axis=1)
        return _avgpool(M, RESAMPLE_LEN)                                       # (32,C)

    @torch.no_grad()
    def infer(self):
        """→ dict {mass,size,stif,(anchor_stif)} 물리 단위, 샘플 부족이면 None."""
        n = len(self.buf["joint"])
        if n < MIN_FRAMES:
            print(f"[eco-engine] 샘플 부족 ({n} < {MIN_FRAMES})")
            return None
        s = self._assemble()
        x = (s - self.seq_mean.reshape(1, -1)) / (self.seq_std.reshape(1, -1) + 1e-8)
        sc = np.array([1.0, self.is_ecoflex], np.float32)          # palm_up · is_ecoflex
        scz = (sc - self.sc_mean.reshape(-1)) / (self.sc_std.reshape(-1) + 1e-8)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        st_ = torch.tensor(scz, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.model(xt, st_)[0].cpu().numpy()                 # (3,) 0~1
        res = {}
        for i, t in enumerate(self.targets):
            lo, hi = self.norm[t]
            v = float(out[i])
            if t == "stif" and self.stif_log and lo > 0:
                llo, lhi = np.log10(lo), np.log10(hi)
                res[t] = float(10.0 ** (v * (lhi - llo) + llo))
            else:
                res[t] = float(v * (hi - lo) + lo)
        if getattr(self.model, "last_anchor", None) is not None:
            va = float(self.model.last_anchor[0].item())
            lo, hi = self.norm["stif"]
            llo, lhi = np.log10(lo), np.log10(hi)
            res["anchor_stif"] = float(10.0 ** (va * (lhi - llo) + llo)) \
                if self.stif_log else float(va * (hi - lo) + lo)
        res["n_frames"] = n
        return res
