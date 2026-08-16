#!/usr/bin/env python3
"""action head — denoiser backbone × 목적함수.

두 축이 **직교**한다:
  · denoiser  : unet1d(FiLM) / dit(adaLN-Zero) / transformer(cross-attn)
  · objective : diffusion(cosine DDPM 학습 + DDIM 추론) / flow(rectified flow) / bc(회귀)

그래서 `dit + flow`, `unet1d + diffusion` 같은 조합이 곱집합으로 나온다.

⚠️ flow matching 을 기본으로 삼지 말 것. ManiFeel 벤치마크가 Diffusion Policy·Equivariant DP·
Flow Matching 을 같은 조건에서 비교했고 **flow 가 둘 다에 뒤졌다.** 옵션으로만 둔다.

unet1d / dit 은 refer/diffusion_policy/dp_model.py 의 구현을 옮긴 것이다 —
기준선과 수치가 갈리지 않게 하려면 여기가 달라지면 안 된다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vtdp.registry import register


# ══════════════════════════════════════════════════════════════════════════
# 공통
# ══════════════════════════════════════════════════════════════════════════
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000)
                         * torch.arange(half, device=t.device, dtype=torch.float32)
                         / (half - 1))
        emb = t.float()[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ══════════════════════════════════════════════════════════════════════════
# denoiser — pooled 조건 (FiLM / adaLN)
# ══════════════════════════════════════════════════════════════════════════
class ConvResBlock(nn.Module):
    """1D 잔차 블록 + FiLM(scale/shift) 조건화."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        ng = 8 if out_ch % 8 == 0 else (4 if out_ch % 4 == 0 else 1)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(ng, out_ch)
        self.norm2 = nn.GroupNorm(ng, out_ch)
        self.act = nn.Mish()
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.act(self.norm1(self.conv1(x)))
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


@register("denoiser", "unet1d")
class ConditionalUnet1D(nn.Module):
    """FiLM 조건화 1D U-Net. `needs = "pooled"`."""

    needs = "pooled"

    def __init__(self, action_dim: int, cond_dim: int, pred_horizon: int,
                 time_emb_dim: int = 128, channels=(64, 128, 256), kernel_size: int = 3):
        super().__init__()
        channels = tuple(channels)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim))
        c = time_emb_dim + cond_dim
        self.stem = nn.Conv1d(action_dim, channels[0], kernel_size, padding=kernel_size // 2)
        n_down = len(channels) - 1
        enc_in = [channels[0]] + list(channels[:-2])
        enc_out = list(channels[:-1])
        self.enc_blocks = nn.ModuleList(
            [ConvResBlock(enc_in[i], enc_out[i], c, kernel_size) for i in range(n_down)])
        self.enc_ds = nn.ModuleList(
            [nn.Conv1d(enc_out[i], enc_out[i], 4, stride=2, padding=1) for i in range(n_down)])
        self.mid_blocks = nn.ModuleList([
            ConvResBlock(channels[-2], channels[-1], c, kernel_size),
            ConvResBlock(channels[-1], channels[-1], c, kernel_size)])
        dec_us_in = list(reversed(channels[1:]))
        dec_skip = list(reversed(channels[:-1]))
        self.dec_us = nn.ModuleList(
            [nn.ConvTranspose1d(dec_us_in[i], dec_us_in[i], 4, stride=2, padding=1)
             for i in range(n_down)])
        self.dec_blocks = nn.ModuleList(
            [ConvResBlock(dec_us_in[i] + dec_skip[i], dec_skip[i], c, kernel_size)
             for i in range(n_down)])
        self.head = nn.Conv1d(channels[0], action_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, noisy_actions, timestep, cond):
        x = noisy_actions.permute(0, 2, 1)
        c = torch.cat([self.time_mlp(timestep), cond], dim=-1)
        x = self.stem(x)
        skips = []
        for block, ds in zip(self.enc_blocks, self.enc_ds):
            x = block(x, c)
            skips.append(x)
            x = ds(x)
        for block in self.mid_blocks:
            x = block(x, c)
        for us, block in zip(self.dec_us, self.dec_blocks):
            x = us(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = block(torch.cat([x, skip], dim=1), c)
        return self.head(x).permute(0, 2, 1)


class DiTBlock(nn.Module):
    """self-attention + MLP, adaLN-Zero 조건화 (Peebles & Xie 2023)."""

    def __init__(self, d_model, n_head, mlp_ratio=4.0, cond_dim=384, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, cond):
        s1, b1, g1, s2, b2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.n1(x) * (1 + s1[:, None]) + b1[:, None]
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None] * a
        h = self.n2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.mlp(h)


@register("denoiser", "dit")
class DiTDenoiser(nn.Module):
    """액션 스텝을 토큰으로 보는 transformer 디노이저. `needs = "pooled"`."""

    needs = "pooled"

    def __init__(self, action_dim: int, cond_dim: int, pred_horizon: int,
                 time_emb_dim: int = 128, d_model: int = 256, n_head: int = 4,
                 n_layer: int = 4, dropout: float = 0.0, max_len: int | None = None):
        super().__init__()
        # max_len 은 pos-emb 슬롯 수. 기본은 pred_horizon(남는 슬롯 = 죽은 파라미터라서).
        # refer/dp_model.py 는 max_len=64 로 고정이라 48슬롯(12,288 param)이 미사용으로 남는다.
        # 그 체크포인트와 shape 을 맞춰야 하면 max_len=64 로 준다.
        max_len = max_len or pred_horizon
        if max_len < pred_horizon:
            raise ValueError(f"max_len({max_len}) < pred_horizon({pred_horizon})")
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim))
        c = time_emb_dim + cond_dim
        self.inp = nn.Linear(action_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_head, 4.0, c, dropout) for _ in range(n_layer)])
        self.nf = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * d_model))
        nn.init.zeros_(self.ada_f[1].weight); nn.init.zeros_(self.ada_f[1].bias)
        self.head = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, noisy_actions, timestep, cond):
        T = noisy_actions.shape[1]
        c = torch.cat([self.time_mlp(timestep), cond], dim=-1)
        x = self.inp(noisy_actions) + self.pos[:, :T]
        for blk in self.blocks:
            x = blk(x, c)
        s, b = self.ada_f(c).chunk(2, dim=-1)
        return self.head(self.nf(x) * (1 + s[:, None]) + b[:, None])


# ══════════════════════════════════════════════════════════════════════════
# denoiser — 토큰 조건 (cross-attention)
# ══════════════════════════════════════════════════════════════════════════
class XBlock(nn.Module):
    """self-attn(액션 토큰) → cross-attn(관측 토큰) → MLP. 전부 pre-norm + adaLN 시간조건."""

    def __init__(self, d_model, n_head, cond_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.sa = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.nk = nn.LayerNorm(d_model)
        self.xa = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d_model)
        h = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, h), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(h, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * d_model))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, obs_tokens, t_emb):
        g1, g2, g3 = self.ada(t_emb).chunk(3, dim=-1)
        h = self.n1(x)
        a, _ = self.sa(h, h, h, need_weights=False)
        x = x + g1[:, None] * a
        kv = self.nk(obs_tokens)
        a, _ = self.xa(self.n2(x), kv, kv, need_weights=False)
        x = x + g2[:, None] * a
        return x + g3[:, None] * self.mlp(self.n3(x))


@register("denoiser", "transformer")
class TransformerDenoiser(nn.Module):
    """액션 토큰이 관측 토큰을 cross-attention 으로 본다. `needs = "tokens"`.

    Diffusion Policy 의 transformer 변형과 같은 골격. pooled 로 뭉개지 않으므로
    "어느 관측 토큰이 이 액션 스텝에 중요한가"를 모델이 고를 수 있다 —
    모달리티가 늘어날수록 pooled 대비 이점이 커진다.
    """

    needs = "tokens"

    def __init__(self, action_dim: int, cond_dim: int, pred_horizon: int,
                 d_model: int = 256, n_head: int = 4, n_layer: int = 4,
                 time_emb_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim))
        self.obs_proj = nn.Linear(cond_dim, d_model) if cond_dim != d_model else nn.Identity()
        self.inp = nn.Linear(action_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, pred_horizon, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [XBlock(d_model, n_head, time_emb_dim, dropout=dropout) for _ in range(n_layer)])
        self.nf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, noisy_actions, timestep, cond):
        # cond: (B, N, D_tok)
        obs = self.obs_proj(cond)
        t = self.time_mlp(timestep)
        x = self.inp(noisy_actions) + self.pos[:, :noisy_actions.shape[1]]
        for blk in self.blocks:
            x = blk(x, obs, t)
        return self.head(self.nf(x))


# ══════════════════════════════════════════════════════════════════════════
# 스케줄러
# ══════════════════════════════════════════════════════════════════════════
class CosineDDPM(nn.Module):
    """Cosine 노이즈 스케줄 (Nichol & Dhariwal 2021) + DDIM 결정적 추론.

    `alpha_bar` 는 **non-persistent buffer** 다. `policy.to(device)` 로 같이 따라가되
    `state_dict` 에는 안 들어간다(config 로 완전히 결정되는 상수라 저장할 값이 아니고,
    EMA 가 state_dict 를 훑으며 건드리는 것도 막는다). plain 속성으로 두면 스텝마다
    CPU→GPU 복사가 일어났다.
    """

    def __init__(self, num_steps: int = 100):
        super().__init__()
        T = num_steps
        s = 0.008
        t = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((t / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
        alpha_bar = (f / f[0]).float()
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        self.T = T
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0),
                             persistent=False)

    def add_noise(self, x0, noise, t):
        ab = self.alpha_bar[t][:, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def ddim_step(self, pred_noise, t_curr: int, t_prev: int, x_t):
        ab = self.alpha_bar[t_curr]
        ab_prev = (self.alpha_bar[t_prev] if t_prev >= 0
                   else torch.ones((), device=self.alpha_bar.device))
        x0 = ((x_t - (1 - ab).sqrt() * pred_noise) / ab.sqrt()).clamp(-3.0, 3.0)
        return ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * pred_noise

    def infer_timesteps(self, n_steps: int) -> list[int]:
        step = max(1, self.T // n_steps)
        return list(range(self.T - 1, -1, -step))[:n_steps]


# ══════════════════════════════════════════════════════════════════════════
# head = denoiser + 목적함수
# ══════════════════════════════════════════════════════════════════════════
class _HeadBase(nn.Module):
    def __init__(self, denoiser: nn.Module, action_dim: int, pred_horizon: int):
        super().__init__()
        self.denoiser = denoiser
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.needs = getattr(denoiser, "needs", "pooled")

    def _cond(self, tokens, pooled):
        return tokens if self.needs == "tokens" else pooled


@register("head", "diffusion")
class DiffusionHead(_HeadBase):
    """Cosine DDPM 학습 (eps 예측) + DDIM 결정적 추론."""

    def __init__(self, denoiser: nn.Module, action_dim: int, pred_horizon: int,
                 diff_steps: int = 100, infer_steps: int = 10):
        super().__init__(denoiser, action_dim, pred_horizon)
        self.sched = CosineDDPM(diff_steps)
        self.diff_steps, self.infer_steps = diff_steps, infer_steps

    def compute_loss(self, action, tokens, pooled):
        B = action.shape[0]
        t = torch.randint(0, self.diff_steps, (B,), device=action.device)
        noise = torch.randn_like(action)
        pred = self.denoiser(self.sched.add_noise(action, noise, t), t,
                             self._cond(tokens, pooled))
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, tokens, pooled, generator=None, infer_steps: int | None = None):
        cond = self._cond(tokens, pooled)
        B, dev = cond.shape[0], cond.device
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=dev, generator=generator)
        ts = self.sched.infer_timesteps(infer_steps or self.infer_steps)
        for i, t_curr in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            t_b = torch.full((B,), t_curr, device=dev, dtype=torch.long)
            x = self.sched.ddim_step(self.denoiser(x, t_b, cond), t_curr, t_prev, x)
        return x


@register("head", "flow")
class FlowHead(_HeadBase):
    """Rectified flow matching. 속도장을 예측하고 forward Euler 로 적분.

    x_τ = (1-τ)·noise + τ·x0,  목표 v = x0 - noise.
    τ 는 Beta(1.5,1) 에서 뽑는다(GR00T N1 레시피 — τ=1 근처를 더 많이 본다).

    ⚠️ ManiFeel 이 같은 조건에서 diffusion 에 뒤진다고 보고했다. 비교용으로만.
    """

    def __init__(self, denoiser: nn.Module, action_dim: int, pred_horizon: int,
                 infer_steps: int = 10, beta_a: float = 1.5, beta_b: float = 1.0,
                 time_scale: float = 100.0):
        super().__init__(denoiser, action_dim, pred_horizon)
        self.infer_steps = infer_steps
        self.beta_a, self.beta_b = beta_a, beta_b
        # denoiser 의 시간 임베딩이 [0,T) 정수 스케일을 가정하므로 τ∈[0,1] 을 맞춰 띄운다
        self.time_scale = time_scale

    def _tau(self, B, device):
        d = torch.distributions.Beta(self.beta_a, self.beta_b)
        return d.sample((B,)).to(device)

    def compute_loss(self, action, tokens, pooled):
        B = action.shape[0]
        tau = self._tau(B, action.device)
        noise = torch.randn_like(action)
        x_tau = (1 - tau[:, None, None]) * noise + tau[:, None, None] * action
        v_pred = self.denoiser(x_tau, tau * self.time_scale, self._cond(tokens, pooled))
        return F.mse_loss(v_pred, action - noise)

    @torch.no_grad()
    def sample(self, tokens, pooled, generator=None, infer_steps: int | None = None):
        cond = self._cond(tokens, pooled)
        B, dev = cond.shape[0], cond.device
        n = infer_steps or self.infer_steps
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=dev, generator=generator)
        dt = 1.0 / n
        for i in range(n):
            tau = torch.full((B,), i * dt, device=dev)
            x = x + dt * self.denoiser(x, tau * self.time_scale, cond)
        return x
