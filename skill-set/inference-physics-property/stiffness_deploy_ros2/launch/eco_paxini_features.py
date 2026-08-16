#!/usr/bin/env python3
"""eco_paxini_features.py — deep_ws paxini_features 의 vendored 라이브러리 부분.

deep_ws/src/ecoflex2fruit/tools/data_build/paxini_features.py 에서 **엔진이 쓰는
라이브러리 함수만**(패드 기하 · 변형3 곡면 합력 · 변형4·5 접촉 특징) 그대로 옮긴
사본이다 — deploy 를 deep_ws 없이 자립시키기 위한 것. 분석/CLI 부분(segment_row·
report·main — config 의존)은 뺐다.

  출처 커밋: deep_ws@07e9f9d (2026-08-14)
  ⚠ deep_ws 쪽 계산이 바뀌면 이 사본도 갱신해야 한다 — 계산이 두 벌로 갈라지면
    학습↔배포 전처리 패리티가 깨진다. 갱신 후 test_ecoflex_engine_offline.py 로
    (deep_ws 가 있는 개발 머신에서) 패리티를 재확인할 것.

COORDS 기본값만 원본과 다르다: config.DATA_ROOT 대신 배포 번들의 sensors.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

#   배포 번들의 sensors.json (학습과 동일 파일 사본 — models/ecoflex2fruit/README.md)
COORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "models", "ecoflex2fruit", "sensors.json")
N_TAXEL = 127
NORMAL_AXIS = 2          # phase1-2 판정: 국소 프레임의 법선축
SAT_TAXEL = 25.5         # taxel 포화값 (README '데이터 주의사항 1')
MIN_CONTACT_N = 1.0      # 총 법선증가가 이보다 작으면 '접촉 없음' → 특징 0
REF_FORCE_N = 3.0        # 고정힘 비교 시점 — 스퀴즈 임계 랜덤화를 지운다
RAMP_POINTS = 40         # 면적성장 기울기 회귀에 쓸 램프 표본 수


# ============================================================== 패드 기하
def load_pad_geometry(path=COORDS, pad_area=None):
    """sensors.json → (P(127,3) mm, N(127,3) 단위법선, A(127,) 셀면적 mm²).

    법선 = 국소 PCA(자기 자신+8이웃) 최소특이벡터를 패드 중심 바깥으로 정렬.
    셀면적 = 최근접 4점 거리의 중앙값². pad_area 를 주면 Σ 를 그 값에 맞춘다.
    """
    with open(path, encoding="utf-8") as f:
        pts = json.load(f)
    assert len(pts) == N_TAXEL, f"sensors.json 점 수 {len(pts)} != {N_TAXEL}"
    P = np.array([[p["x"], p["y"], p["z"]] for p in sorted(pts, key=lambda p: p["id"])])
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    order = np.argsort(D, axis=1)
    ctr = P.mean(0)
    Nrm = np.empty_like(P)
    for i in range(N_TAXEL):
        Q = P[order[i, :9]]                              # 자기 자신 + 8이웃
        n = np.linalg.svd(Q - Q.mean(0))[2][2]
        Nrm[i] = n if n @ (P[i] - ctr) >= 0 else -n      # 볼록면 가정으로 바깥 정렬
    Nrm /= np.linalg.norm(Nrm, axis=1, keepdims=True)
    A = np.median(np.take_along_axis(D, order[:, 1:5], axis=1), axis=1) ** 2
    if pad_area:
        A = A * (pad_area / A.sum())
    return P, Nrm, A


# ============================================================== 변형3 — 곡면 고려 합력
def curved_resultant(raw, geom):
    """raw (T,4,127,3) → 합력 3종 dict.

    resul_curved (T,4,3)  Σᵢ f_i2·nᵢ — 패드 좌표계 벡터합. 현행 Σ127 자리 대체.
    normal_sum   (T,4)    Σᵢ f_i2    — 방향 상쇄가 없는 총 법선력
    shear_sum    (T,4)    Σᵢ|(f_i0,f_i1)| — 전단 크기합.
    """
    _, Nrm, _ = geom
    f = np.nan_to_num(np.asarray(raw, np.float32))
    fn = f[..., NORMAL_AXIS]
    return {
        "resul_curved": np.einsum("tfi,ic->tfc", fn, Nrm).astype(np.float32),
        "normal_sum": fn.sum(-1).astype(np.float32),
        "shear_sum": np.linalg.norm(f[..., :NORMAL_AXIS], axis=-1).sum(-1).astype(np.float32),
    }


# ============================================================== 변형4 — 접촉 면적·위치
CONTACT_NAMES = ("area", "press", "cx", "cy", "cz", "spread", "elong", "wrap", "nsum")


def contact_features(d, geom):
    """Δ법선력 (127,) → 접촉 특징 9개. 접촉이 없으면 전부 0.

    area·press·cx·cy·cz·spread·elong·wrap·nsum — 정의는 deep_ws 원본 docstring 참고.
    """
    P, Nrm, A = geom
    w = np.clip(np.asarray(d, np.float64), 0, None)
    s = float(w.sum())
    if s < 1e-6:
        return np.zeros(len(CONTACT_NAMES), np.float32)
    wa = w * A
    swa = float(wa.sum())
    area = swa ** 2 / max(float((w ** 2 * A).sum()), 1e-12)
    c = (wa[:, None] * P).sum(0) / swa
    dP = P - c
    spread = float(np.sqrt((wa * (dP ** 2).sum(1)).sum() / swa))
    nbar = (w[:, None] * Nrm).sum(0) / s
    wrap = 1.0 - float(np.linalg.norm(nbar))
    # 접선평면(가중 평균법선에 직교) 위 2D 공분산의 주축비
    u = nbar / max(np.linalg.norm(nbar), 1e-12)
    e1 = np.cross(u, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(u, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    xy = np.stack([dP @ e1, dP @ np.cross(u, e1)], axis=1)
    cov = np.einsum("i,ij,ik->jk", wa, xy, xy) / swa
    ev = np.clip(np.linalg.eigvalsh(cov), 1e-12, None)
    elong = float(np.sqrt(ev[1] / ev[0]))
    return np.array([area, s / max(area, 1e-9), *c, spread, elong, wrap, s], np.float32)


BLOCK_NAMES = (tuple(f"{n}3" for n in CONTACT_NAMES)
               + tuple(f"{n}p" for n in CONTACT_NAMES) + ("dAdF",))


def contact_block(dseq, geom):
    """Δ법선맵 시계열 (T,127) → 블록 특징 19개 (@3N · @peak · 면적성장 기울기)."""
    d = np.clip(np.asarray(dseq, np.float64), 0, None)
    tot = d.sum(1)
    if len(tot) < 5 or float(tot.max()) < MIN_CONTACT_N:
        return np.zeros(len(BLOCK_NAMES), np.float32)
    ip = int(np.argmax(tot))
    i3 = int(np.argmin(np.abs(tot[:ip + 1] - REF_FORCE_N)))
    ramp = np.flatnonzero(tot[:ip + 1] > 0.5)
    slope = 0.0
    if len(ramp) >= 5:
        ramp = ramp[:: max(1, len(ramp) // RAMP_POINTS)]
        ar = [float(contact_features(d[i], geom)[0]) for i in ramp]
        slope = float(np.polyfit(tot[ramp], ar, 1)[0])
    return np.concatenate([contact_features(d[i3], geom),
                           contact_features(d[ip], geom),
                           [slope]]).astype(np.float32)
