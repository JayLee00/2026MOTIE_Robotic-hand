#!/usr/bin/env python3
"""모달리티별 인코더.

**불변식**: 모든 인코더는 `(B, n_tokens, d_model)` 을 낸다. 예외 없다.
이 규약 하나 덕분에 fusion 이 인코더 종류를 몰라도 되고, 조합이 곱집합으로 성립한다.

인코더는 `.out_dim`(= d_model) 과 `.n_tokens` 를 속성으로 노출해야 한다.
fusion 이 토큰 수를 미리 알아야 pos-emb 등을 잡을 수 있기 때문이다.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vtdp.registry import register


# ══════════════════════════════════════════════════════════════════════════
# 공통 블록
# ══════════════════════════════════════════════════════════════════════════
def mlp(sizes: list[int], act=nn.Mish, dropout: float = 0.0, norm=True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(act())
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SpatialSoftmax(nn.Module):
    """feature map → 채널별 기대 좌표 (Levine et al. 2016).

    Diffusion Policy(Chi et al.) 의 vision 레시피에서 global average pooling 대신 쓰는 것.
    평균 풀링은 "어디에" 정보를 버리는데, 조작에서는 그게 핵심이라 남긴다.
    """

    def __init__(self, num_kp: int | None = None, in_ch: int = 512):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, num_kp, 1) if num_kp else None
        self.num_kp = num_kp or in_ch

    def forward(self, x):                            # (N, C, H, W)
        if self.proj is not None:
            x = self.proj(x)
        N, C, H, W = x.shape
        # 좌표 그리드를 [-1,1] 로
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype),
            indexing="ij")
        attn = F.softmax(x.reshape(N, C, H * W), dim=-1)
        exp_x = (attn * pos_x.reshape(1, 1, -1)).sum(-1)
        exp_y = (attn * pos_y.reshape(1, 1, -1)).sum(-1)
        return torch.cat([exp_x, exp_y], dim=-1)     # (N, 2C)


def replace_bn_with_gn(module: nn.Module, ch_per_group: int = 16) -> nn.Module:
    """BatchNorm → GroupNorm 재귀 치환.

    Diffusion Policy 논문이 명시한 요구사항이다: BatchNorm 의 running stat 이
    EMA 가중치와 상호작용하면 학습이 망가진다. GroupNorm 은 배치 통계가 없어 안전하다.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            n_ch = child.num_features
            n_groups = max(1, n_ch // ch_per_group)
            while n_ch % n_groups != 0:
                n_groups -= 1
            setattr(module, name, nn.GroupNorm(n_groups, n_ch))
        else:
            replace_bn_with_gn(child, ch_per_group)
    return module


# ══════════════════════════════════════════════════════════════════════════
# state (proprioception)
# ══════════════════════════════════════════════════════════════════════════
@register("state", "mlp")
class StateMLP(nn.Module):
    """(B,T,D) → flatten → MLP → 토큰 1개. refer/ 의 MLPObsEncoder 와 동일한 구조."""

    def __init__(self, in_dim: int, horizon: int, d_model: int = 256,
                 hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        h = hidden or max(64, d_model)
        self.net = mlp([in_dim * horizon, h, d_model], dropout=dropout)
        self.out_dim, self.n_tokens = d_model, 1

    def forward(self, x):                            # (B,T,D)
        return self.net(x.flatten(1)).unsqueeze(1)   # (B,1,d)


# ══════════════════════════════════════════════════════════════════════════
# tactile
# ══════════════════════════════════════════════════════════════════════════
@register("tactile", "mlp")
class TactileMLP(nn.Module):
    """flatten → MLP → 토큰 1개.

    ⚠️ 이것이 **현재 baseline 과 동등한 처리**다(모달리티만 분리됨). 비교 기준선으로 남긴다.
    Beyond Binary 의 'raw taxel' arm 이 이 처리였고 1-bit 접촉 신호보다도 낮았다.
    """

    def __init__(self, in_dim: int, horizon: int, d_model: int = 256,
                 hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        h = hidden or max(64, d_model)
        self.net = mlp([in_dim * horizon, h, d_model], dropout=dropout)
        self.out_dim, self.n_tokens = d_model, 1

    def forward(self, x):
        return self.net(x.flatten(1)).unsqueeze(1)


@register("tactile", "conv1d")
class TactileConv1d(nn.Module):
    """시간축 1D conv + GroupNorm + 학습형 α residual. **부위축·시간축 보존.**

    ManipForce `FTEmbed` 이식(`diffusion_policy/model/vision/utils/ft_embed.py`).
    α 를 1e-2 로 시작하는 게 핵심 — 학습 초기에는 사실상 선형 투영이고, 필요한 만큼만
    conv 분기를 켠다. 소량 데이터에서 촉각 분기가 노이즈에 과적합하는 걸 늦춘다.
    그리고 k=3 시간 conv 는 **국소 차분을 학습으로 배울 수 있다** — `data.transform: hp`
    (인과적 고역통과)의 학습형 대응물이다.

    ⚠️ 원본과의 의도적 차이 두 개:
      1. **`n_part` 로 쪼개 부위를 배치로 접는다.** 원본은 `Conv1d(ft_dim → hidden)` 이라
         첫 conv 에서 6축을 섞는데(손목 F/T 하나라 맞다), 우리는 손가락 4개라
         섞으면 루트 `CLAUDE.md` 절대 방향 1 위반이다. `TactileLSTM` 과 같은 처리.
      2. **`time_embed`** — 원본 `use_temporal_embed: True` 에 해당하는 학습형 시간
         positional embedding. 이 저장소에는 부위 임베딩만 있었고 시간축은
         **아무 표식 없이** flatten 되고 있었다(T3 시간셔플 +0.10%±0.12% = 안 쓴다).

    토큰 수는 `n_part × horizon` — `TactileLSTM(return_seq=True)` 와 같다(융합 용량 불변).
    """

    def __init__(self, in_dim: int, horizon: int, d_model: int = 256,
                 kernel_size: int = 3, alpha_init: float = 1e-2, dropout: float = 0.0,
                 n_part: int = 1, time_embed: bool = False):
        super().__init__()
        ch = _split_parts(in_dim, n_part, "tactile/conv1d")
        self.n_part, self.horizon = n_part, horizon
        self.conv = nn.Conv1d(ch, d_model, kernel_size, padding=kernel_size // 2)
        n_groups = max(1, d_model // 64)
        while d_model % n_groups != 0:
            n_groups -= 1
        self.norm = nn.GroupNorm(n_groups, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.res_proj = nn.Linear(ch, d_model) if ch != d_model else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.part = nn.Parameter(torch.zeros(1, n_part, 1, d_model)) if n_part > 1 else None
        self.time = nn.Parameter(torch.zeros(1, 1, horizon, d_model)) if time_embed else None
        for p in (self.part, self.time):
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)
        self.out_norm = nn.LayerNorm(d_model)     # 원본의 `ft_norm` 자리
        self.out_dim, self.n_tokens = d_model, n_part * horizon

    def forward(self, x):                            # (B,T,in_dim)
        B, T, _ = x.shape
        P = self.n_part
        z = x.reshape(B, T, P, -1).permute(0, 2, 1, 3).reshape(B * P, T, -1)   # (B*P,T,ch)
        h = self.act(self.norm(self.conv(z.transpose(1, 2)))).transpose(1, 2)  # (B*P,T,d)
        out = (self.res_proj(z) + self.alpha.to(h.dtype) * self.drop(h)).reshape(B, P, T, -1)
        if self.part is not None:
            out = out + self.part
        if self.time is not None:
            out = out + self.time
        return self.out_norm(out.reshape(B, P * T, self.out_dim))


def _split_parts(in_dim: int, n_part: int, where: str) -> int:
    """`in_dim` 을 부위(손가락) `n_part` 개로 쪼갠다. 부위당 채널 수를 돌려준다.

    ⚠️ **레이아웃 가정**: 한 스텝의 벡터가 부위-major 로 이어져 있다고 본다 —
    `x[..., p*ch + c]` 가 부위 p 의 채널 c. Paxini 는 둘 다 이 형태다 —
    `10_hand_paxini_ft`(12) = 4손가락 × 3축, `11_hand_paxini_raw`(1524) = 4손가락 × 381
    (= 127탁셀 × xyz). 근거는 드라이버 `tools/paxini/tactile_uart.py` 의 (4,127,3) 디코드.
    실제 배열이 다르면(축-major 등) **조용히 엉뚱한 손가락을 학습한다.**
    """
    if n_part < 1:
        raise ValueError(f"{where}: n_part 는 1 이상 — 받은 값 {n_part}")
    if in_dim % n_part:
        raise ValueError(
            f"{where}: in_dim={in_dim} 이 n_part={n_part} 로 안 나뉜다. "
            f"10_hand_paxini_ft(12)=4손가락×3축 이면 n_part=4, "
            f"11_hand_paxini_raw(1524)=4손가락×381 이면 n_part=4.")
    return in_dim // n_part


@register("tactile", "lstm")
class TactileLSTM(nn.Module):
    """부위별 LSTM (가중치 공유) + 부위 임베딩.

    `n_part>1` 이면 입력을 (B,T,부위,채널) 로 쪼개 **부위를 배치로 접어** 같은 LSTM 에
    통과시킨다. 부위 임베딩이 "몇 번째 손가락인가"를 남기므로 첫 Linear 에서
    손가락이 섞이지 않는다 — 프로젝트 문서가 경고하는 naive flatten 을 피하는 지점.

    토큰 수: `return_seq=True` → `n_part × horizon` (시간축 보존, **기본**)
             `return_seq=False` → `n_part` (부위별 1토큰, 시간은 LSTM 이 접는다)
    """

    def __init__(self, in_dim: int, horizon: int, d_model: int = 256,
                 n_layer: int = 1, return_seq: bool = True, n_part: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        ch = _split_parts(in_dim, n_part, "tactile/lstm")
        self.n_part, self.horizon = n_part, horizon
        self.inp = nn.Linear(ch, d_model)
        self.part = nn.Parameter(torch.zeros(1, n_part, 1, d_model)) if n_part > 1 else None
        if self.part is not None:
            nn.init.trunc_normal_(self.part, std=0.02)
        self.lstm = nn.LSTM(d_model, d_model, n_layer, batch_first=True,
                            dropout=dropout if n_layer > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)
        self.return_seq = return_seq
        self.out_dim = d_model
        self.n_tokens = n_part * (horizon if return_seq else 1)

    def forward(self, x):                            # (B,T,in_dim)
        B, T, _ = x.shape
        P = self.n_part
        z = self.inp(x.reshape(B, T, P, -1).permute(0, 2, 1, 3))    # (B,P,T,d)
        if self.part is not None:
            z = z + self.part
        out, (h, _) = self.lstm(z.reshape(B * P, T, -1))
        z = out if self.return_seq else h[-1].unsqueeze(1)          # (B*P, T|1, d)
        return self.norm(z.reshape(B, -1, self.out_dim))


@register("tactile", "transformer")
class TactileTransformer(nn.Module):
    """부위 × 시간 토큰에 self-attention. 임베딩을 **부위축과 시간축으로 분리**한다.

    토큰 하나 = (부위 p, 시각 t). 부위 임베딩과 시간 임베딩을 따로 더하므로
    "어느 손가락의 언제"가 attention 안에서 살아 있다. 접촉이 손가락을 옮겨 가는
    낙하 회복 태스크에서 필요한 정보가 정확히 이것이다.

    토큰 수: `use_cls=False` → `n_part × horizon` (**기본**), `True` → 1
    """

    def __init__(self, in_dim: int, horizon: int, d_model: int = 256,
                 n_head: int = 4, n_layer: int = 2, use_cls: bool = False,
                 n_part: int = 1, dropout: float = 0.0):
        super().__init__()
        ch = _split_parts(in_dim, n_part, "tactile/transformer")
        self.n_part, self.horizon, self.use_cls = n_part, horizon, use_cls
        self.inp = nn.Linear(ch, d_model)
        self.t_pos = nn.Parameter(torch.zeros(1, horizon, 1, d_model))
        self.p_pos = nn.Parameter(torch.zeros(1, 1, n_part, d_model))
        nn.init.trunc_normal_(self.t_pos, std=0.02)
        nn.init.trunc_normal_(self.p_pos, std=0.02)
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model
        self.n_tokens = 1 if use_cls else n_part * horizon

    def forward(self, x):                            # (B,T,in_dim)
        B, T, _ = x.shape
        z = self.inp(x.reshape(B, T, self.n_part, -1)) + self.t_pos + self.p_pos
        z = z.reshape(B, T * self.n_part, -1)        # 토큰 순서 = (t0p0, t0p1, ..., t1p0, ...)
        if self.use_cls:
            z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1)
        z = self.enc(z)
        return self.norm(z[:, :1] if self.use_cls else z)


# ══════════════════════════════════════════════════════════════════════════
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _VisionBase(nn.Module):
    """(B,T,3,H,W) 을 (B*T,...) 로 펴서 backbone 에 넣고 다시 접는 공통부."""

    def __init__(self, imagenet_norm: bool = True):
        super().__init__()
        if imagenet_norm:
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        else:
            self.mean = self.std = None

    def _pre(self, x):
        B, T = x.shape[:2]
        z = x.flatten(0, 1)
        if self.mean is not None:
            z = (z - self.mean) / self.std
        return z, B, T

    def train(self, mode: bool = True):
        """frozen backbone 은 항상 eval 로 둔다.

        dropout·stochastic depth·BatchNorm running stat 갱신을 모두 막는다.
        BN 을 남긴 채 backbone 을 얼릴 때 이게 없으면 running stat 이 계속 움직여
        "얼렸다"는 말이 거짓이 된다.
        """
        super().train(mode)
        if getattr(self, "frozen", False) and hasattr(self, "backbone"):
            self.backbone.eval()
        return self


class LoRALinear(nn.Module):
    """`Linear` 에 저랭크 보정 branch 를 덧댄다 (Hu et al. 2021).

    `B` 를 0 으로 초기화하므로 **학습 시작 시점의 출력이 원본과 정확히 같다.**
    22 데모에 21M ViT 를 통째로 파인튜닝하는 건 위험하지만 완전히 얼리면
    도메인(손·레몬 클로즈업)이 ImageNet 과 멀어서 적응할 여지가 없다 — LoRA 가 그 중간이다.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_b(self.lora_a(self.lora_drop(x)))


def inject_lora(root: nn.Module, r: int, alpha: float,
                targets: tuple[str, ...], dropout: float = 0.0) -> int:
    """`targets` 에 이름이 맞는 `Linear` 를 전부 `LoRALinear` 로 바꾼다. 바꾼 개수 반환."""
    n = 0
    for name, child in root.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(root, name, LoRALinear(child, r, alpha, dropout))
            n += 1
        else:
            n += inject_lora(child, r, alpha, targets, dropout)
    return n


class _GradCkpt(nn.Module):
    """블록 하나를 gradient checkpointing 으로 감싼다 (메모리↔연산 교환).

    LoRA 를 켜면 base 가 frozen 이어도 **활성값 전체를 들고 있어야 한다**(보정 branch 에
    gradient 를 흘려야 하니까). frozen+no_grad 때는 0 이던 비용이라 그냥 켜면 터진다 —
    batch 64 × 프레임 2 = ViT forward 128회면 ViT-S/14 만으로 13GB 가 넘는다.
    checkpointing 은 블록 경계의 입력만 남기고 나머지는 backward 때 다시 계산한다.
    """

    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        # `self.training` 으로 판단하면 안 된다 — frozen backbone 은 항상 eval 이라
        # (dropout/BN 차단 목적) checkpointing 이 조용히 꺼진다. 기준은 **grad 가
        # 켜져 있는가**다: 켜져 있으면 backward 가 활성값을 요구하고, `sample()` 처럼
        # no_grad 안이면 애초에 저장할 것이 없으니 그냥 통과시킨다.
        if torch.is_grad_enabled() and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self.mod, x, use_reentrant=False)
        return self.mod(x)


@register("vision", "resnet18")
class VisionResNet18(_VisionBase):
    """ResNet-18 + GroupNorm 치환 (+ spatial softmax 또는 패치 토큰).

    Diffusion Policy(Chi et al., RSS 2023) 의 vision 레시피 그대로:
      · BatchNorm → GroupNorm  (EMA 와 충돌 방지, 논문이 명시한 요구사항)
      · global average pooling → spatial softmax  ("어디에" 를 버리지 않기 위해)

    `tokens="patch"` 면 마지막 feature map 을 토큰 시퀀스로 낸다 (cross-attn fusion 용).

    ⚠️ **`norm` 은 `frozen` 과 짝지어야 한다.** DP 논문이 BN→GN 을 요구한 이유는
    backbone 을 **학습할 때** running stat 이 EMA 와 충돌하기 때문이다. 얼린 backbone 은
    stat 이 갱신되지 않아 충돌 자체가 없고, GN 으로 바꾸면 사전학습 가중치가 전제한
    정규화 통계만 버린다 — 실측으로 이미지-의존 성분이 72.8% → 15.3% 로 떨어져
    **서로 다른 그림이 거의 같은 feature 로 나온다**(이미지간 코사인 0.976).
    그래서 `norm="auto"` 는 파인튜닝이면 GN, frozen 이면 BN(eval 고정) 을 고른다.
    """

    def __init__(self, in_shape: tuple[int, int, int], horizon: int, d_model: int = 256,
                 pretrained: bool = True, frozen: bool = False, norm: str = "auto",
                 tokens: str = "pool", num_kp: int = 32,
                 imagenet_norm: bool = True, dropout: float = 0.0):
        super().__init__(imagenet_norm)
        try:
            import torchvision
        except ImportError as e:
            raise ImportError(
                "vision/resnet18 은 torchvision 이 필요하다. "
                "`pip install torchvision` 하거나 config 에서 vision.smallcnn 을 쓸 것.") from e

        if norm not in ("auto", "gn", "bn"):
            raise ValueError(f"norm 은 'auto'|'gn'|'bn' — 받은 값 {norm!r}")
        weights = "IMAGENET1K_V1" if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.norm_kind = ("bn" if frozen else "gn") if norm == "auto" else norm
        if self.norm_kind == "gn":
            replace_bn_with_gn(self.backbone)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        self.frozen = frozen
        feat_ch = 512
        self.tokens = tokens
        if tokens == "pool":
            self.pool = SpatialSoftmax(num_kp=num_kp, in_ch=feat_ch)
            self.proj = mlp([num_kp * 2, d_model], dropout=dropout)
            self.n_tokens = horizon
        elif tokens == "patch":
            h, w = in_shape[1] // 32, in_shape[2] // 32
            self.proj = nn.Linear(feat_ch, d_model)
            self.n_tokens = horizon * h * w
        else:
            raise ValueError(f"tokens 는 'pool' 또는 'patch' — 받은 값 {tokens!r}")
        self.out_dim = d_model

    def forward(self, x):
        z, B, T = self._pre(x)
        if self.frozen:
            with torch.no_grad():
                f = self.backbone(z)
        else:
            f = self.backbone(z)                     # (B*T, 512, h, w)
        if self.tokens == "pool":
            return self.proj(self.pool(f)).reshape(B, T, -1)
        f = f.flatten(2).transpose(1, 2)             # (B*T, h*w, 512)
        return self.proj(f).reshape(B, -1, self.out_dim)   # (B, T*h*w, d)


@register("vision", "dinov2")
class VisionDINOv2(_VisionBase):
    """DINOv2 ViT. **기본은 frozen** — 22 데모에 21M 파라미터를 파인튜닝하는 건 자살이다.

    `torch.hub.load('facebookresearch/dinov2', ...)` 로 받는다.
    ⚠️ 첫 실행에 인터넷이 필요하다(가중치 다운로드). 연구실이 오프라인이면
    `~/.cache/torch/hub` 를 미리 채워 가거나 ManipForce 의 `checkpoints/prepare_dinov2.py`
    로 받아둔 체크포인트를 쓸 것.

    ⚠️ 입력 H,W 는 patch_size(14) 의 배수여야 한다. 224 = 14x16 → 패치 256개.

    tokens:
      "cls"   → 프레임당 토큰 1개 (가볍다. 기본)
      "patch" → 프레임당 (H/14)x(W/14) 토큰 (cross-attn fusion 용. 224면 256개/프레임)
      "pool"  → 패치 토큰 평균 → 프레임당 1개

    `lora_rank>0` 이면 base 가중치는 얼린 채 attention 의 qkv/proj 에만 저랭크 보정을
    학습한다(수백만 대신 수십만 파라미터). **LoRA 를 쓰면 backbone forward 에
    `no_grad` 를 걸 수 없다** — 걸면 LoRA 가 gradient 를 못 받고 조용히 학습되지 않는다.
    """

    NAMES = {"vits14": "dinov2_vits14", "vitb14": "dinov2_vitb14",
             "vitl14": "dinov2_vitl14", "vitg14": "dinov2_vitg14"}

    def __init__(self, in_shape: tuple[int, int, int], horizon: int, d_model: int = 256,
                 variant: str = "vits14", frozen: bool = True, tokens: str = "cls",
                 lora_rank: int = 0, lora_alpha: float | None = None,
                 lora_targets: tuple[str, ...] = ("qkv", "proj"),
                 lora_dropout: float = 0.0, grad_ckpt: bool = False,
                 imagenet_norm: bool = True, dropout: float = 0.0):
        super().__init__(imagenet_norm)
        if variant not in self.NAMES:
            raise ValueError(f"dinov2 variant 는 {sorted(self.NAMES)} 중 하나 — 받은 값 {variant!r}")
        c, h, w = in_shape
        if h % 14 or w % 14:
            raise ValueError(
                f"dinov2 는 입력이 14의 배수여야 한다 — 받은 값 {h}x{w}. "
                f"224(=14x16) 또는 252(=14x18) 를 쓸 것.")
        try:
            self.backbone = torch.hub.load("facebookresearch/dinov2", self.NAMES[variant],
                                           pretrained=True, verbose=False)
        except Exception as e:
            raise RuntimeError(
                f"DINOv2({variant}) 로드 실패: {e}\n"
                f"  오프라인이면 ~/.cache/torch/hub 를 미리 채워 가거나 "
                f"vision.resnet18 을 쓸 것.") from e

        self.frozen = frozen
        if frozen:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.lora_rank = int(lora_rank)
        if self.lora_rank > 0:
            if not frozen:
                raise ValueError("lora_rank>0 은 frozen=true 와 함께 쓴다 "
                                 "(base 를 얼리고 저랭크 보정만 학습하는 것이 LoRA 다)")
            n_inj = inject_lora(self.backbone, self.lora_rank,
                                float(lora_alpha or 2 * self.lora_rank),
                                tuple(lora_targets), lora_dropout)
            if n_inj == 0:
                raise ValueError(
                    f"lora_targets={tuple(lora_targets)} 에 맞는 Linear 가 없다. "
                    f"DINOv2 블록의 이름은 attn.qkv / attn.proj / mlp.fc1 / mlp.fc2 다.")
            self.n_lora_layers = n_inj

        self.grad_ckpt = bool(grad_ckpt)
        if self.grad_ckpt:
            blocks = getattr(self.backbone, "blocks", None)
            if not isinstance(blocks, nn.ModuleList):
                raise RuntimeError("grad_ckpt: backbone.blocks(ModuleList) 를 못 찾았다")
            for i, blk in enumerate(blocks):
                blocks[i] = _GradCkpt(blk)

        feat = self.backbone.embed_dim
        self.tokens = tokens
        self.proj = nn.Sequential(nn.Linear(feat, d_model), nn.LayerNorm(d_model))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        if tokens == "patch":
            self.n_tokens = horizon * (h // 14) * (w // 14)
        elif tokens in ("cls", "pool"):
            self.n_tokens = horizon
        else:
            raise ValueError(f"tokens 는 'cls'|'pool'|'patch' — 받은 값 {tokens!r}")
        self.out_dim = d_model

    def forward(self, x):
        z, B, T = self._pre(x)
        # LoRA 가 붙어 있으면 no_grad 를 걸 수 없다 — 걸면 보정 branch 가 학습되지 않는다.
        ctx = (torch.no_grad() if (self.frozen and self.lora_rank == 0)
               else contextlib.nullcontext())
        with ctx:
            out = self.backbone.forward_features(z)
        if self.tokens == "cls":
            f = out["x_norm_clstoken"].unsqueeze(1)          # (B*T, 1, feat)
        elif self.tokens == "pool":
            f = out["x_norm_patchtokens"].mean(1, keepdim=True)
        else:
            f = out["x_norm_patchtokens"]                    # (B*T, L, feat)
        f = self.drop(self.proj(f))
        return f.reshape(B, -1, self.out_dim)
