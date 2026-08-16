"""가림(occlusion) 연산의 **단일 정본** — 학습 augmentation 과 평가 프로브가 같은 함수를 부른다.

왜 이 파일이 생겼나 (done_v6 §2E-4 · LEARNINGS 2026-08-11):
  v6 은 `model.modality_dropout` 으로 **융합에서 rgb 토큰을 0 으로 곱해** 학습하고,
  `tools/occlusion_probe.py` 로 **회색 이미지의 인코더 출력(= 0 이 아닌 상수 토큰)** 을 줘서
  평가했다. 즉 "0 을 학습하고 상수로 평가"했다. 그 서명이 전면 가림에서 VT 열화가 되돌아
  커지는 비단조(+2.74 → +4.23)다. 두 구현이 따로 있었기 때문에 생긴 어긋남이므로,
  **학습과 평가가 같은 함수를 부르게** 하는 것이 이 모듈의 존재 이유 전부다.

교란은 **[0,1] 픽셀 공간**에서 한다 — `data.py:_decode` 가 /255 만 하고 ImageNet 정규화는
인코더 안(`encoders.py:_VisionBase._pre`)이다. `fill="imagenet_mean"` 이면 인코더가 그 패치에서
정확히 0 을 본다.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# 채워 넣을 색 (모두 [0,1] 픽셀 공간)
FILLS = {
    "imagenet_mean": (0.485, 0.456, 0.406),   # 인코더가 이 패치에서 정확히 0 을 본다
    "black": (0.0, 0.0, 0.0),                 # 더 OOD — 부차
    "gray": (0.5, 0.5, 0.5),
}


def side_for_ratio(ratio: float, H: int, W: int) -> int:
    """면적비 → 정사각 패치 한 변. 원본 `occlusion_probe._occlude` 와 같은 산식."""
    side = int(round(math.sqrt(ratio * H * W)))
    return max(1, min(side, min(H, W)))


def topleft(demo: int, t: int, side: int, H: int, W: int, seed: int) -> tuple[int, int]:
    """(demo,t) 로 결정되는 패치 좌상단. 두 arm 이 같은 물리 창에 같은 마스크를 보게 한다."""
    h = (int(seed) * 1000003 + int(demo) * 10007 + int(t) * 31) & 0xFFFFFFFF
    r = np.random.default_rng(h)
    return int(r.integers(0, H - side + 1)), int(r.integers(0, W - side + 1))


def paint(x: torch.Tensor, boxes: list, fill: torch.Tensor) -> torch.Tensor:
    """**칠하는 연산은 여기 한 곳뿐이다.** 학습·평가 두 경로가 이 함수로 합류한다.

    x     : (B,T,3,H,W) [0,1]
    boxes : 길이 B. 원소는 `(top,left,side)` 또는 `None`(그 샘플은 안 가린다).
            `side >= min(H,W)` 면 전면 가림과 같아진다.
    fill  : (1,1,3,1,1) 로 broadcast 되는 색 텐서
    """
    y = x.clone()
    for bi, box in enumerate(boxes):
        if box is None:
            continue
        top, left, side = box
        y[bi, :, :, top:top + side, left:left + side] = fill
    return y


def occlude(x: torch.Tensor, ratio: float, pattern: str, fill: torch.Tensor,
            pairs: list | None = None, seed: int = 0) -> torch.Tensor:
    """평가 경로 — 면적비 `ratio` 만큼 정사각 패치로 덮는다. 마스크는 (demo,t) 로 결정된다.

    `pattern="rand"` 면 `pairs`(길이 B 의 (demo,t))가 필요하다.
    r >= 1.0 은 전면 가림이라 pattern 이 무의미하다.
    """
    B, T, _, H, W = x.shape
    if ratio >= 1.0:
        y = x.clone()
        y[:] = fill
        return y
    side = side_for_ratio(ratio, H, W)
    if pattern == "center":
        top, left = (H - side) // 2, (W - side) // 2
        boxes = [(top, left, side)] * B
    elif pattern == "rand":
        if pairs is None:
            raise ValueError("pattern='rand' 는 pairs=(demo,t) 목록이 필요하다")
        boxes = [topleft(d, t, side, H, W, seed) + (side,) for d, t in pairs]
    else:
        raise ValueError(f"pattern 은 center|rand — 받은 값 {pattern!r}")
    return paint(x, boxes, fill)


def sample_boxes(B: int, H: int, W: int, p: float, ratio_min: float, ratio_max: float,
                 pattern: str, gen: torch.Generator) -> list:
    """학습 경로 — 샘플별 Bernoulli(p) 로 가릴지 정하고, 가리면 ratio ~ U(min,max).

    평가와 달리 **매 step 다른 마스크**여야 한다(같은 창을 여러 epoch 볼 때 한 자리만 외우면
    가림 자체가 데모 신원 단서가 된다) → (demo,t) 결정 대신 `gen` 에서 뽑는다.
    칠하는 연산은 `paint()` 로 평가와 합류하므로 분포 불일치가 생길 자리가 없다.
    """
    u = torch.rand(B, generator=gen, device="cpu")
    r = ratio_min + (ratio_max - ratio_min) * torch.rand(B, generator=gen, device="cpu")
    boxes = []
    for bi in range(B):
        if u[bi].item() >= p:
            boxes.append(None)
            continue
        side = side_for_ratio(float(r[bi]), H, W)
        if pattern == "center":
            boxes.append(((H - side) // 2, (W - side) // 2, side))
        elif pattern == "rand":
            top = int(torch.randint(0, H - side + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, W - side + 1, (1,), generator=gen).item())
            boxes.append((top, left, side))
        else:
            raise ValueError(f"pattern 은 center|rand — 받은 값 {pattern!r}")
    return boxes


def sample_boxes_paired(pairs: list, H: int, W: int, p: float, ratio_min: float, ratio_max: float,
                        p_full: float, epoch: int, seed: int) -> list:
    """학습 경로 v2 (plan_v8) — 마스크를 **`(demo, t, epoch)` 로 결정**한다.

    `sample_boxes` 의 결함 2개를 고친다 (done_v7 §2D 이후 발견):

    ① **V arm 과 VT arm 이 같은 마스크를 못 봤다.** 배치 순서에서 뽑았고 두 arm 의 윈도우 수가
       다르다(82,425 vs 83,111) → 같은 물리 창이 다른 마스크를 받았다. paired 비교의 취지에
       어긋나고 Δ 의 밴드를 넓힌다. 평가 프로브는 처음부터 `(demo,t)` 결정이었는데
       (`occlude(pattern="rand")`) 학습만 안 그랬다.
    ② **`ratio_max=0.75` 라 전면 가림을 한 번도 학습하지 않았다.** 그런데 v7 평가에서 밴드를
       넘은 유일한 실질 조건이 `full_r1` 이다(Δ −2.90±1.93, DiD −3.26±1.92) — **학습 분포
       밖에서만 촉각이 값을 했다.** `p_full` 로 그 조건을 분포 안으로 끌어온다.

    `epoch` 이 시드에 들어가므로 같은 창도 epoch 마다 다른 마스크를 본다(다양성 유지).
    같은 `(demo, t, epoch, seed)` → 같은 마스크. 두 arm 이 같은 seed 를 쓰면 **바이트 동일**하다.

    p_full: **가려진 샘플 중** 전면 가림의 비율. 예: p=0.5, p_full=0.25 →
            무가림 50% · 부분 가림 37.5% · 전면 가림 12.5%.
    """
    boxes = []
    for d, t in pairs:
        # (demo, t, epoch) → 독립 난수 3개. topleft 와 같은 해시 규약을 쓴다.
        h = ((int(seed) * 1000003 + int(d) * 10007 + int(t) * 31) ^ (int(epoch) * 2654435761)) & 0xFFFFFFFF
        r = np.random.default_rng(h)
        if r.random() >= p:
            boxes.append(None)
            continue
        ratio = 1.0 if r.random() < p_full else (ratio_min + (ratio_max - ratio_min) * r.random())
        side = side_for_ratio(ratio, H, W)
        top = int(r.integers(0, H - side + 1))
        left = int(r.integers(0, W - side + 1))
        boxes.append((top, left, side))
    return boxes


def fill_vec(name: str, device) -> torch.Tensor:
    """FILLS 이름 → (1,1,3,1,1) 텐서."""
    if name not in FILLS:
        raise ValueError(f"fill={name!r} 없음. 가능: {sorted(FILLS)}")
    return torch.tensor(FILLS[name], device=device).view(1, 1, 3, 1, 1)
