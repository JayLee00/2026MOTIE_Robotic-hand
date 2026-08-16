#!/usr/bin/env python3
"""policy — 인코더 · fusion · head 를 config 하나로 조립한다.

    obs dict ──▶ 모달리티별 encoder ──▶ {k: (B,n_k,D)} ──▶ fusion ──▶ (tokens, pooled)
                                                                          │
                                                            action head ◀─┘

config 에서 모달리티 키를 빼면 인코더도 안 만들어진다 →
`only rgb` / `only tactile` / `rgb+tactile` 이 **코드 분기 없이** 나온다.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from vtdp import encoders as _enc      # noqa: F401  (레지스트리 등록 부수효과)
from vtdp import fusion as _fus        # noqa: F401
from vtdp import heads as _heads       # noqa: F401
from vtdp.registry import available, build


class VTDPolicy(nn.Module):
    """visuo-tactile diffusion policy.

    Parameters
    ----------
    obs_spec : {name: {"kind": "state"|"tactile"|"vision", "shape": ..., "horizon": int,
                       "encoder": str, "encoder_kwargs": dict}}
        `shape` 는 state/tactile 이면 int(차원), vision 이면 (C,H,W).
    """

    def __init__(self, obs_spec: dict, action_dim: int, pred_horizon: int,
                 d_model: int = 256, cond_dim: int = 256,
                 fusion: str = "concat", fusion_kwargs: dict | None = None,
                 denoiser: str = "unet1d", denoiser_kwargs: dict | None = None,
                 head: str = "diffusion", head_kwargs: dict | None = None,
                 modality_dropout: float = 0.0, modality_dropout_on: list | None = None):
        super().__init__()
        if not obs_spec:
            raise ValueError("obs_spec 이 비었다 — 모달리티가 최소 하나는 있어야 한다")

        self.obs_spec = copy.deepcopy(obs_spec)
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.d_model = d_model
        self.modality_dropout = modality_dropout
        self.modality_dropout_on = list(modality_dropout_on) if modality_dropout_on else None

        # ── 1. 모달리티별 인코더 ────────────────────────────────────────────
        self.encoders = nn.ModuleDict()
        token_spec: dict[str, int] = {}
        for name, spec in sorted(self.obs_spec.items()):
            kind = spec["kind"]
            if kind not in ("state", "tactile", "vision"):
                raise ValueError(f"{name}: kind 는 state|tactile|vision — 받은 값 {kind!r}")
            kw = dict(spec.get("encoder_kwargs") or {})
            kw.update(horizon=spec["horizon"], d_model=d_model)
            if kind == "vision":
                kw["in_shape"] = tuple(spec["shape"])
            else:
                kw["in_dim"] = int(spec["shape"])
            enc = build(kind, spec["encoder"], **kw)
            if enc.out_dim != d_model:
                raise ValueError(
                    f"{name}/{spec['encoder']} 의 out_dim={enc.out_dim} != d_model={d_model}")
            self.encoders[name] = enc
            token_spec[name] = enc.n_tokens

        self.token_spec = token_spec

        # ── 2. fusion ──────────────────────────────────────────────────────
        self.fusion = build("fusion", fusion,
                            token_spec=token_spec, d_model=d_model, cond_dim=cond_dim,
                            **(fusion_kwargs or {}))

        # ── 3. head (+ denoiser) ───────────────────────────────────────────
        head_kwargs = dict(head_kwargs or {})
        dkw = dict(denoiser_kwargs or {})
        # cross-attn 디노이저는 토큰 차원을, 나머지는 pooled 차원을 받는다
        probe = available("denoiser")
        if denoiser not in probe:
            raise KeyError(f"denoiser/{denoiser} 없음. 가능: {probe}")
        needs_tokens = denoiser == "transformer"
        if denoiser in ("dit", "transformer"):
            dkw.setdefault("d_model", d_model)
        den = build("denoiser", denoiser,
                    action_dim=action_dim, pred_horizon=pred_horizon,
                    cond_dim=(d_model if needs_tokens else self.fusion.cond_dim), **dkw)
        self.head = build("head", head, denoiser=den,
                          action_dim=action_dim, pred_horizon=pred_horizon, **head_kwargs)

        self.needs = self.head.needs

    # ──────────────────────────────────────────────────────────────────────
    def _encode(self, obs: dict, mask: dict | None = None):
        missing = [k for k in self.obs_spec if k not in obs]
        if missing:
            raise KeyError(f"obs 에 {missing} 가 없다. 필요: {sorted(self.obs_spec)}")
        feats = {name: self.encoders[name](obs[name]) for name in self.obs_spec}
        return self.fusion(feats, mask)

    def _sample_dropout_mask(self, obs: dict) -> dict | None:
        """학습 중 모달리티를 확률적으로 가린다 (DIPOLE 레시피, p≈0.2).

        효과 둘: ① 한 모달리티에만 의존하는 걸 막고 ② 배포 때 센서 하나가 죽어도
        정책이 무너지지 않는다. 9분 데이터에서는 정규화 효과도 크다.
        **모든 모달리티가 동시에 가려지는 경우는 배제한다** — 조건이 사라지면 학습 신호가 없다.

        `modality_dropout_on` 이 주어지면 **그 키만** 가린다. 왜 필요한가: 전부 독립으로
        가리면 V arm(state+rgb, 2개)과 VT arm(state+tactile+rgb, 3개)의 **총 교란량이
        달라져** 짝 비교가 오염된다(plan_v6 M2). 가림 대조에서 움직이는 축은 rgb 하나다.
        """
        if not self.training or self.modality_dropout <= 0 or len(self.obs_spec) < 2:
            return None
        names = sorted(self.obs_spec)
        tgt = set(self.modality_dropout_on or names)
        any_key = obs[names[0]]
        B, dev = any_key.shape[0], any_key.device
        keep = {k: (torch.rand(B, device=dev) >= self.modality_dropout if k in tgt
                    else torch.ones(B, dtype=torch.bool, device=dev)) for k in names}
        stacked = torch.stack([keep[k] for k in names], dim=0)      # (M,B)
        all_gone = ~stacked.any(dim=0)                              # (B,)
        if all_gone.any():
            keep[names[0]] = keep[names[0]] | all_gone              # 최소 하나는 살린다
        return keep

    def forward(self, obs: dict, mask: dict | None = None):
        return self._encode(obs, mask)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        obs, action = batch["obs"], batch["action"]
        mask = batch.get("mask") or self._sample_dropout_mask(obs)
        tokens, pooled = self._encode(obs, mask)
        return self.head.compute_loss(action, tokens, pooled)

    @torch.no_grad()
    def sample(self, obs: dict, mask: dict | None = None, generator=None,
               infer_steps: int | None = None) -> torch.Tensor:
        tokens, pooled = self._encode(obs, mask)
        return self.head.sample(tokens, pooled, generator=generator, infer_steps=infer_steps)

    # ──────────────────────────────────────────────────────────────────────
    def n_params(self, trainable_only: bool = False) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)

    def summary(self) -> str:
        lines = ["VTDPolicy"]
        for name in sorted(self.obs_spec):
            spec, enc = self.obs_spec[name], self.encoders[name]
            n = sum(p.numel() for p in enc.parameters())
            lines.append(f"  {name:9s} {spec['kind']:8s} {str(spec['shape']):16s} "
                         f"T={spec['horizon']:<3d} {spec['encoder']:12s} "
                         f"→ {enc.n_tokens:>4d} tok  ({n/1e6:.2f}M)")
        nf = sum(p.numel() for p in self.fusion.parameters())
        nh = sum(p.numel() for p in self.head.parameters())
        lines.append(f"  fusion    {type(self.fusion).__name__:22s} "
                     f"{self.fusion.n_cond_tokens:>4d} tok, cond={self.fusion.cond_dim} ({nf/1e6:.2f}M)")
        lines.append(f"  head      {type(self.head).__name__:22s} needs={self.needs} ({nh/1e6:.2f}M)")
        lines.append(f"  총 파라미터 {self.n_params()/1e6:.2f}M "
                     f"(학습 대상 {self.n_params(True)/1e6:.2f}M)")
        return "\n".join(lines)


def build_policy(cfg: dict) -> VTDPolicy:
    """config dict → policy. `vtdp.config.load_config` 가 넘겨주는 형태를 받는다."""
    m = cfg["model"]
    return VTDPolicy(
        obs_spec=cfg["obs_spec"],
        action_dim=cfg["action"]["dim"],
        pred_horizon=cfg["action"]["pred_horizon"],
        d_model=m.get("d_model", 256),
        cond_dim=m.get("cond_dim", 256),
        fusion=m["fusion"], fusion_kwargs=m.get("fusion_kwargs"),
        denoiser=m.get("denoiser", "unet1d"), denoiser_kwargs=m.get("denoiser_kwargs"),
        head=m.get("head", "diffusion"), head_kwargs=m.get("head_kwargs"),
        modality_dropout=m.get("modality_dropout", 0.0),
        modality_dropout_on=m.get("modality_dropout_on"))
