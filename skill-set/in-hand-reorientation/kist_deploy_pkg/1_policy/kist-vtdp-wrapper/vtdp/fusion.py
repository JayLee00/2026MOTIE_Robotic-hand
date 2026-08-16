#!/usr/bin/env python3
"""fusion — 모달리티별 토큰들을 하나의 조건으로 합친다.

**불변식**: `forward(feats) -> (tokens (B,N,D), pooled (B,C))`.
두 형태를 항상 낸다. head 가 필요한 쪽만 쓴다(FiLM/adaLN 은 pooled, cross-attn 은 tokens).

`feats` 는 `{modality_name: (B, n_k, D)}` dict 이고, **키가 없으면 그 모달리티는 없는 것**이다.
따라서 `only tactile` / `rgb+tactile` 이 코드 분기 없이 config 만으로 성립한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vtdp.registry import register


class _FusionBase(nn.Module):
    """modality embedding + 순서 고정을 공통 처리.

    modality embedding 은 ManipForce 의 `use_modal_embed` 와 같은 역할 —
    합쳐진 토큰 시퀀스에서 "이 토큰이 어느 감각에서 왔는지"를 남긴다.
    """

    def __init__(self, token_spec: dict[str, int], d_model: int,
                 modal_embed: bool = True):
        super().__init__()
        self.names = sorted(token_spec)              # 순서 고정 — 재현성
        self.token_spec = {k: token_spec[k] for k in self.names}
        self.d_model = d_model
        self.n_total = sum(self.token_spec.values())
        if modal_embed:
            self.modal = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1, 1, d_model)) for k in self.names})
            for p in self.modal.values():
                nn.init.trunc_normal_(p, std=0.02)
        else:
            self.modal = None

    def _stack(self, feats: dict[str, torch.Tensor], mask: dict | None = None):
        """dict → (B, N_total, D). modal embed 를 더하고, mask 된 모달리티는 0 으로 지운다."""
        parts = []
        for k in self.names:
            if k not in feats:
                raise KeyError(
                    f"fusion 이 모달리티 {k!r} 를 기대했는데 feats 에 없다. "
                    f"기대: {self.names}, 받음: {sorted(feats)}")
            z = feats[k]
            if self.modal is not None:
                z = z + self.modal[k]
            if mask is not None and k in mask:
                # mask[k]: (B,) bool, False = 가림
                z = z * mask[k].to(z.dtype).view(-1, 1, 1)
            parts.append(z)
        return torch.cat(parts, dim=1)


@register("fusion", "concat")
class ConcatFusion(_FusionBase):
    """토큰을 그냥 이어붙이고 pooled 는 flatten→MLP.

    ⚠️ **S0 baseline.** vault 에 이 방식이 해롭다는 반례가 5건 모여 있다
    (T-Rex, ForceVLA2, GeoProp, FTP-1, CoorDex) — 하지만 그 반례들은 전부
    *이미 시각으로 사전학습된 큰 backbone 에 저차원 센서를 뒤늦게 붙인* late-fusion 상황이다.
    우리는 처음부터 함께 학습하는 소형 정책이라 해당하지 않을 수 있다.
    **그래서 기준선으로 반드시 남긴다 — 이게 실제로 지는지가 첫 번째 실험이다.**
    """

    def __init__(self, token_spec: dict[str, int], d_model: int, cond_dim: int = 256,
                 modal_embed: bool = True, pool: str = "flatten", dropout: float = 0.0):
        super().__init__(token_spec, d_model, modal_embed)
        self.pool = pool
        if pool == "flatten":
            self.head = nn.Sequential(nn.Linear(self.n_total * d_model, cond_dim),
                                      nn.LayerNorm(cond_dim), nn.Mish())
        elif pool == "mean":
            self.head = nn.Sequential(nn.Linear(d_model, cond_dim),
                                      nn.LayerNorm(cond_dim), nn.Mish())
        else:
            raise ValueError(f"pool 은 'flatten'|'mean' — 받은 값 {pool!r}")
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.cond_dim, self.n_cond_tokens = cond_dim, self.n_total

    def forward(self, feats, mask=None):
        tok = self._stack(feats, mask)
        z = tok.flatten(1) if self.pool == "flatten" else tok.mean(1)
        return tok, self.drop(self.head(z))


@register("fusion", "passthrough")
class PassthroughFusion(_FusionBase):
    """추가 파라미터 0. 토큰을 이어붙이고 pooled 는 평균만 낸다.

    **refer/ 기준선을 정확히 재현하기 위한 것.** refer/ 에는 fusion 단계가 아예 없어서
    (인코더 → 디노이저 직결) 어떤 fusion 이든 파라미터가 붙으면 비교가 오염된다.
    `state.mlp + passthrough + unet1d + diffusion` 이 refer/dp_unet 과 파라미터까지 같다.

    modal embed 도 기본으로 끈다 — 기준선에 없던 것이니까.
    """

    def __init__(self, token_spec: dict[str, int], d_model: int, cond_dim: int = 256,
                 modal_embed: bool = False):
        super().__init__(token_spec, d_model, modal_embed)
        if cond_dim != d_model:
            raise ValueError(
                f"passthrough 는 투영층이 없어 cond_dim({cond_dim}) == d_model({d_model}) 이어야 한다")
        self.cond_dim, self.n_cond_tokens = d_model, self.n_total

    def forward(self, feats, mask=None):
        tok = self._stack(feats, mask)
        return tok, tok.mean(1)


class _CrossAttnBlock(nn.Module):
    """query 를 key/value 로 갱신. ManipForce `CrossAttention` 과 동형(pre-norm 으로 정리)."""

    def __init__(self, d_model: int, n_head: int = 4, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.nk = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        h = d_model * mlp_ratio
        self.ffn = nn.Sequential(nn.Linear(d_model, h), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(h, d_model))

    def forward(self, q, kv):
        a, _ = self.attn(self.n1(q), self.nk(kv), self.nk(kv), need_weights=False)
        q = q + a
        return q + self.ffn(self.n2(q))


@register("fusion", "cross_attn")
class CrossAttnFusion(_FusionBase):
    """모달리티 간 양방향 cross-attention (ManipForce 방식).

    각 모달리티가 query 가 되어 **나머지 전부**를 key/value 로 본다.
    GelFusion 의 ablation 에서 cross-attention 이 naive concat 과 self-attention 을 모두 이겼다.

    ⚠️ `pool` 은 **pooled 를 쓰는 head(unet1d·dit) 에서만** 의미가 있다. 기본이 `mean` 이면
    부위·시간 토큰을 평균으로 지워서 cross-attention 으로 만든 구조가 그대로 사라진다
    (`gated`/`concat` 은 원래 flatten 이었다). 그래서 기본을 `flatten` 으로 둔다.
    """

    def __init__(self, token_spec: dict[str, int], d_model: int, cond_dim: int = 256,
                 n_head: int = 4, n_layer: int = 1, modal_embed: bool = True,
                 pool: str = "flatten", dropout: float = 0.0):
        super().__init__(token_spec, d_model, modal_embed)
        # 모달리티가 하나면 상대가 없어 self-attention 으로 축퇴된다(구조는 동일).
        self.blocks = nn.ModuleDict({
            k: nn.ModuleList([_CrossAttnBlock(d_model, n_head, dropout=dropout)
                              for _ in range(n_layer)]) for k in self.names})
        if pool not in ("flatten", "mean"):
            raise ValueError(f"pool 은 'flatten'|'mean' — 받은 값 {pool!r}")
        self.pool = pool
        in_dim = self.n_total * d_model if pool == "flatten" else d_model
        self.head = nn.Sequential(nn.Linear(in_dim, cond_dim),
                                  nn.LayerNorm(cond_dim), nn.Mish())
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.cond_dim, self.n_cond_tokens = cond_dim, self.n_total

    def forward(self, feats, mask=None):
        embedded = {}
        for k in self.names:
            z = feats[k]
            if self.modal is not None:
                z = z + self.modal[k]
            if mask is not None and k in mask:
                z = z * mask[k].to(z.dtype).view(-1, 1, 1)
            embedded[k] = z

        out = {}
        for k in self.names:
            others = [embedded[o] for o in self.names if o != k]
            kv = torch.cat(others, dim=1) if others else embedded[k]
            q = embedded[k]
            for blk in self.blocks[k]:
                q = blk(q, kv)
            out[k] = q

        tok = torch.cat([out[k] for k in self.names], dim=1)
        z = tok.flatten(1) if self.pool == "flatten" else tok.mean(1)
        return tok, self.drop(self.head(z))


@register("fusion", "gated")
class GatedFusion(_FusionBase):
    """접촉 세기로 촉각 분기를 gating 한다.

    M2-ResiPolicy(이진 threshold) · Dream-Tac CASA(촉각 시간차분에 sigmoid) · FoAR(학습형
    접촉 예측기) 세 논문이 독립적으로 수렴한 메커니즘. 촉각 노름을 게이트로 쓰면
    **접촉이 없는 구간에 촉각 분기가 노이즈로 과적합하는 걸 막는다** —
    9분짜리 데이터에서 가장 값싼 보험이다.

    gate = sigmoid(w · ‖tactile 토큰‖ + b) 를 [floor, 1] 로 매핑해 곱한다.
    floor=0.15 는 Dream-Tac 의 값. 완전히 0 으로 죽이지 않아 gradient 가 살아 있다.
    """

    def __init__(self, token_spec: dict[str, int], d_model: int, cond_dim: int = 256,
                 gate_on: str = "tactile", floor: float = 0.15,
                 modal_embed: bool = True, dropout: float = 0.0):
        super().__init__(token_spec, d_model, modal_embed)
        if gate_on not in self.token_spec:
            raise KeyError(f"gated fusion 의 gate_on={gate_on!r} 가 없다. 가능: {self.names}")
        self.gate_on, self.floor = gate_on, floor
        self.gate = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(),
                                  nn.Linear(d_model // 4, 1))
        self.head = nn.Sequential(nn.Linear(self.n_total * d_model, cond_dim),
                                  nn.LayerNorm(cond_dim), nn.Mish())
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.cond_dim, self.n_cond_tokens = cond_dim, self.n_total
        self.last_gate = None                        # 로깅용 — 게이트가 실제로 움직이는지 본다

    def forward(self, feats, mask=None):
        embedded = {}
        for k in self.names:
            z = feats[k]
            if self.modal is not None:
                z = z + self.modal[k]
            if mask is not None and k in mask:
                z = z * mask[k].to(z.dtype).view(-1, 1, 1)
            embedded[k] = z

        g = torch.sigmoid(self.gate(embedded[self.gate_on].mean(1)))   # (B,1)
        g = self.floor + (1.0 - self.floor) * g
        self.last_gate = g.detach()
        embedded[self.gate_on] = embedded[self.gate_on] * g.unsqueeze(1)

        tok = torch.cat([embedded[k] for k in self.names], dim=1)
        return tok, self.drop(self.head(tok.flatten(1)))
