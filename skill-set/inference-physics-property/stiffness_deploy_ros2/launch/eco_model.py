# eco_model.py — deep_ws/src/ecoflex2fruit/model.py 의 vendored 사본 (deploy 자립용).
#   출처 커밋: deep_ws@07e9f9d (2026-08-14). deep_ws 쪽 model.py 가 바뀌면 이 사본도
#   갱신해야 한다 — ckpt 구조와 클래스 정의가 어긋나면 load_state_dict 가 실패한다.
"""물성 회귀 모델 (mass / size / stif 동시 출력).

    입력 (B, L, C) ─┬─ [1×1 사영] ─ stem conv ─ ResBlock1D ──┐
                    └─ 스칼라 (B, S) ─ LayerNorm ────────┐   │
                                                        │   ▼
                                    시간축 인코더 (transformer | GRU/LSTM)
                                                        │   │
                                                        ▼   ▼
                                        풀링 (mean+max | 타깃별 어텐션)
                                                            │
                                                            ▼
                                        헤드 → sigmoid → (B, n_targets)
                                        + 보조 헤드 (cone · 순서회귀)
"""
import torch
from torch import nn


# ═══════════════════════════════════════════════════════════════════════════
# 유틸
# ═══════════════════════════════════════════════════════════════════════════
def _sinusoid(max_len: int, d: int) -> torch.Tensor:
    """고정 사인/코사인 위치임베딩 (1, max_len, d)."""
    pos = torch.arange(max_len, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)[None, :]
    ang = pos / torch.pow(10000.0, i / d)

    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)[:, : pe[:, 1::2].shape[1]]      # d 가 홀수면 잘라 맞춘다
    return pe[None]


def _mlp_head(feat_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """전 헤드 공통 형태 — Dropout → Linear(64) → GELU → Dropout → Linear(out)."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(feat_dim, 64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, out_dim),
    )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# 블록
# ═══════════════════════════════════════════════════════════════════════════
class ResBlock1D(nn.Module):
    """conv-bn-act ×2 + shortcut. 길이는 그대로 두고 채널만 바꾼다."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.short = (
            nn.Sequential() if in_ch == out_ch else
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + self.short(x))
        return self.drop(out)


# ═══════════════════════════════════════════════════════════════════════════
# 본 모델
# ═══════════════════════════════════════════════════════════════════════════
class MultiTargetRegressorAux(nn.Module):
    """CNN stem + 시간축 인코더 + 풀링 + 멀티타깃 헤드.

    선택 손잡이
        temporal   "transformer" | "gru" | "lstm"      시간축 인코더
        pool       "meanmax" | "meanmax_heads" | "attn" | "attn_prior" | "pqa"
                   풀링. attn 계열은 헤드 분리·Kendall 가중이 **묶음**으로 켜지고,
                   `meanmax_heads` 는 그중 **헤드 분리만** 켠다 (§10-5 귀속 실험)
        in_proj    채널이 많을 때 앞에 붙이는 1×1 사영 폭 (None 이면 없음)
        cone_head  fruit 도메인용 보조 헤드
        ord_mode   "sord" | "coral" | "pinball"        stif 를 순서회귀로 대체
    """

    def __init__(self, in_channels, n_scalars=0, hidden_dim=64, n_targets=3,
                 dropout=0.3, temporal="gru", in_proj=None, pool="meanmax",
                 prior_channels=None, prior_target=2, cone_head=False,
                 ord_mode="", ord_k=16,
                 #   transformer 구조
                 tf_layers=2, tf_nhead=4, tf_ff_mult=2.0, tf_posemb="learned",
                 #   §12b 앵커 cross-attention — anchor_n=0 이면 완전 무변화
                 anchor_n=0, anchor_dh=48, anchor_tau=1.0):
        super().__init__()

        # ── 설정 ────────────────────────────────────────────────────────────
        self.cone_head = bool(cone_head)
        self.n_scalars = int(n_scalars)
        self.temporal = temporal
        self.pool = pool
        self.n_targets = int(n_targets)

        self.prior_channels = prior_channels
        self.prior_target = int(prior_target)

        # ── 1. 입력단 — [1×1 사영] → stem conv → ResBlock ────────────────────
        self.pre = None
        stem_in = in_channels
        if in_proj:
            self.pre = nn.Sequential(
                nn.Conv1d(in_channels, in_proj, kernel_size=1, bias=False),
                nn.BatchNorm1d(in_proj),
                nn.GELU(),
            )
            stem_in = in_proj

        self.stem = nn.Sequential(
            nn.Conv1d(stem_in, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.res = ResBlock1D(32, 48, dropout=dropout)

        # ── 2. 시간축 인코더 ────────────────────────────────────────────────
        if temporal == "transformer":
            d = hidden_dim
            self.tf_layers = int(tf_layers)
            self.tf_nhead = int(tf_nhead)
            self.tf_ff_mult = float(tf_ff_mult)
            self.tf_posemb = str(tf_posemb)
            if d % self.tf_nhead:
                raise ValueError(f"hidden({d}) 이 nhead({self.tf_nhead}) 로 안 나눠떨어진다")

            self.proj = nn.Linear(48, d)

            if self.tf_posemb == "learned":
                self.pos = nn.Parameter(torch.zeros(1, 512, d))
            elif self.tf_posemb == "sin":
                self.register_buffer("pos", _sinusoid(512, d), persistent=False)
            elif self.tf_posemb == "none":
                self.pos = None
            else:
                raise ValueError(f"모르는 TF_POSEMB: {self.tf_posemb}")

            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=self.tf_nhead,
                dim_feedforward=max(1, int(round(d * self.tf_ff_mult))),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=self.tf_layers)
            feat_dim = d * 2 + self.n_scalars            # mean+max 풀링 → 2d
        else:
            rnn = nn.LSTM if temporal == "lstm" else nn.GRU
            self.gru = rnn(48, hidden_dim, num_layers=1, batch_first=True,
                           bidirectional=True)
            feat_dim = hidden_dim * 4 + self.n_scalars   # 양방향 2H → mean+max 4H

        # ── 3. 스칼라 ───────────────────────────────────────────────────────
        self.scalar_norm = nn.LayerNorm(self.n_scalars) if self.n_scalars else None

        # ── 4. 풀링 · 주 헤드 ───────────────────────────────────────────────
        if pool in ("attn", "attn_prior"):
            # 타깃마다 시점 가중을 따로 학습하고 헤드도 따로 둔다
            D = feat_dim - self.n_scalars            # 시간축 특징 폭 (2H 또는 d) × 2
            self.attn_w = nn.Parameter(torch.zeros(self.n_targets, D // 2))
            self.heads = nn.ModuleList(
                _mlp_head(feat_dim, 1, dropout) for _ in range(self.n_targets))
            self.log_vars = nn.Parameter(torch.zeros(self.n_targets))   # Kendall 손실 가중
            self.head = None
            if pool == "attn_prior":
                self.prior_lambda = nn.Parameter(torch.ones(1))
        elif pool == "meanmax_heads":
            #   §10-5 — 타깃별 **헤드만** 분리한다. 풀링은 meanmax 그대로다.
            #   `attn_w`(시점 가중)·`log_vars`(Kendall 가중)를 **만들지 않는 것**이
            #   이 조건의 정의다 — train.py 가 Kendall 을 `log_vars` 존재 여부로
            #   켜므로, 안 만들면 손실이 그대로 MSE 다(학습 루프 수정 불필요).
            self.heads = nn.ModuleList(
                _mlp_head(feat_dim, 1, dropout) for _ in range(self.n_targets))
            self.head = None
        elif pool == "pqa":
            #   §12 — 속성(타깃) 쿼리 cross-attention 풀링 (docs/crossattention.md 용도1).
            #   `attn` 과의 차이는 시점 가중의 파라미터화뿐이다: 직접 가중(h @ attn_w[k])
            #   대신 학습 쿼리 q 와 키 사영 Wk 의 내적으로 만든다(+값 사영 Wv).
            #   dq=48 = 시간축 특징 폭과 같게 → 헤드 입력 폭(ctx+max+scalars)이
            #   attn 경로의 feat_dim 과 같아져 **헤드 파라미터가 attn 조건과 동일**하다
            #   (차이는 쿼리·사영 기구뿐 — §12-2 사다리의 귀속 논리).
            #   `log_vars` 를 만들지 않는다 — 손실은 MSE 그대로 (§10-5 규약).
            #   ⚠ 기존 pool 분기들의 모듈 생성 순서는 건드리지 않는다(§10-5 경고) —
            #     이 분기는 pool="pqa" 일 때만 실행된다.
            hid_t = (feat_dim - self.n_scalars) // 2     # 시간축 특징 폭 (d 또는 2H)
            self.pqa_dq = hid_t
            self.pqa_q = nn.Parameter(
                torch.randn(self.n_targets, self.pqa_dq) * 0.02)
            self.pqa_k = nn.Linear(hid_t, self.pqa_dq)
            self.pqa_v = nn.Linear(hid_t, self.pqa_dq)
            self.heads = nn.ModuleList(
                _mlp_head(self.pqa_dq + hid_t + self.n_scalars, 1, dropout)
                for _ in range(self.n_targets))
            self.head = None
        else:
            self.head = _mlp_head(feat_dim, n_targets, dropout)

        # ── 5. 보조 헤드 ────────────────────────────────────────────────────
        if self.cone_head:
            self.head_cone = _mlp_head(feat_dim, 1, dropout)

        self.ord_mode = (ord_mode or "").strip()
        self.ord_k = int(ord_k)
        self.stif_idx = int(prior_target)      # TARGETS 안의 stif 위치(기본 2)
        if self.ord_mode:
            self.head_ord = _mlp_head(feat_dim, self.ord_k, dropout)
            self.register_buffer("ord_centers",
                                 (torch.arange(self.ord_k).float() + 0.5) / self.ord_k)

        # ── 6. §12b 앵커 cross-attention (docs/crossattention.md 용도2) ─────
        #   Q = 시간 풀링 표현(현 물체), KV = train 개체 앵커 은행 → stif 를
        #   "신뢰 앵커 대비 위치"(어텐션 가중합, 라벨 정규화 공간)로도 예측한다.
        #   은행(anchor_feat·anchor_stif)은 train.py 가 epoch 마다 채운다 —
        #   버퍼라 체크포인트에 저장돼 배포에서도 그대로 쓴다(원문 §3-3).
        #   ⚠ 기존 모듈 생성 순서 불변 — anchor_n=0(기본)이면 아무것도 안 만든다.
        self.anchor_n = int(anchor_n)
        if self.anchor_n > 0:
            D = feat_dim - self.n_scalars          # 시간 풀링 표현 폭 (mean+max)
            self.anchor_dh = int(anchor_dh)
            self.anc_q = nn.Linear(D, self.anchor_dh)
            self.anc_k = nn.Linear(D, self.anchor_dh)
            #   tau 는 log 공간 학습 파라미터(양수 보장) — 원문 §3-2 검증: 낮을수록
            #   최근접 앵커에 집중, 높을수록 보간.
            self.anc_log_tau = nn.Parameter(
                torch.tensor(float(anchor_tau)).log())
            self.register_buffer("anchor_feat", torch.zeros(self.anchor_n, D))
            self.register_buffer("anchor_stif", torch.zeros(self.anchor_n))
            self.register_buffer("anchor_ready", torch.zeros(1))

    # ───────────────────────────────────────────────────────────────────────
    def forward(self, x, s=None):
        """x: (B, L, C) 시퀀스 · s: (B, n_scalars) 스칼라 → (B, n_targets) 0~1."""

        # ── 1. 입력단 ───────────────────────────────────────────────────────
        h = x.transpose(1, 2)                               # (B,C,L)
        if self.pre is not None:
            h = self.pre(h)
        h = self.res(self.stem(h)).transpose(1, 2)          # (B,L,48)

        # ── 2. 시간축 인코더 ────────────────────────────────────────────────
        if self.temporal == "transformer":
            z = self.proj(h)
            if self.pos is not None:
                z = z + self.pos[:, :z.size(1)]
            h = self.encoder(z)                             # (B,L,d)
        else:
            h, _ = self.gru(h)                              # (B,L,2H)

        # ── 3. 스칼라 ───────────────────────────────────────────────────────
        if self.n_scalars:
            if s is None:
                s = x.new_zeros(x.size(0), self.n_scalars)
            s = self.scalar_norm(s)

        # ── 3′. §12b 앵커 경로 — 풀링 방식과 무관하게 시간 풀링 표현으로 계산 ─
        self.last_anchor = None
        if self.anchor_n > 0:
            tp = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)  # (B,D)
            self.last_anchor_in = tp
            if bool(self.anchor_ready.item()):
                q = self.anc_q(tp)                          # (B,dh)
                k = self.anc_k(self.anchor_feat)            # (A,dh)
                tau = self.anc_log_tau.exp()
                sc = (q @ k.t()) / (self.anchor_dh ** 0.5) / tau
                a = torch.softmax(sc, dim=-1)               # (B,A) 앵커 유사도
                self.last_anchor = a @ self.anchor_stif     # (B,) 라벨 정규화 공간
                self.last_anchor_attn = a

        # ── 4-a. 어텐션 풀링 경로 — 타깃별 시점 가중 + 타깃별 헤드 ──────────
        if self.pool in ("attn", "attn_prior"):
            mx = h.max(dim=1).values                        # 피크 정보는 공유

            prior = None
            if self.pool == "attn_prior" and self.prior_channels:
                cols = torch.cat([torch.arange(a0, a1) for a0, a1 in self.prior_channels])
                xf = x[:, :, cols.to(x.device)]
                d = (xf[:, 1:] - xf[:, :-1]).abs().mean(dim=2)
                a_ = torch.cat([d[:, :1], d], dim=1)        # (B,L)
                prior = a_ / (a_.amax(dim=1, keepdim=True) + 1e-6)

            outs, attn = [], []
            for k, head in enumerate(self.heads):
                logit = h @ self.attn_w[k]
                if prior is not None and k == self.prior_target:
                    logit = logit + self.prior_lambda * prior
                a = torch.softmax(logit, dim=1)             # (B,L) 타깃별 시점 가중
                pk = (a.unsqueeze(-1) * h).sum(dim=1)
                f = torch.cat([pk, mx] + ([s] if self.n_scalars else []), dim=1)
                outs.append(head(f))
                attn.append(a)

            self.last_attn = torch.stack(attn, dim=1)       # (B, n_targets, L) — 해석용
            return torch.sigmoid(torch.cat(outs, dim=1))

        # ── 4-a′. 속성 쿼리 풀링 경로 (§12 pqa) — 쿼리·키 내적 시점 가중 ────
        if self.pool == "pqa":
            mx = h.max(dim=1).values                        # 피크 정보 공유 (attn 과 동일)
            K = self.pqa_k(h)                               # (B,L,dq)
            V = self.pqa_v(h)
            scores = torch.einsum("pd,bld->bpl", self.pqa_q, K) / (self.pqa_dq ** 0.5)
            a = torch.softmax(scores, dim=-1)               # (B, n_targets, L)
            ctx = torch.einsum("bpl,bld->bpd", a, V)        # (B, n_targets, dq)
            outs = []
            for k, head in enumerate(self.heads):
                f = torch.cat([ctx[:, k], mx]
                              + ([s] if self.n_scalars else []), dim=1)
                outs.append(head(f))
            self.last_attn = a                              # §10 해석 도구 재사용 형상
            return torch.sigmoid(torch.cat(outs, dim=1))

        # ── 4-b. mean+max 풀링 경로 ─────────────────────────────────────────
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        if self.n_scalars:
            pooled = torch.cat([pooled, s], dim=1)
        self.last_feat = pooled

        if self.cone_head:
            self.last_cone = torch.sigmoid(self.head_cone(pooled)).squeeze(-1)  # (B,)

        if self.head is None:                   # meanmax_heads — 타깃별 헤드 (§10-5)
            out = torch.sigmoid(torch.cat([hd(pooled) for hd in self.heads], dim=1))
        else:
            out = torch.sigmoid(self.head(pooled))

        # ── 5. 순서회귀 헤드가 있으면 stif 자리를 갈아끼운다 ────────────────
        if self.ord_mode:
            z = self.head_ord(pooled)                       # (B,K) 로짓
            self.last_ord = z
            if self.ord_mode == "sord":
                v = (torch.softmax(z, dim=1) * self.ord_centers).sum(dim=1)
            elif self.ord_mode == "coral":
                v = torch.sigmoid(z).mean(dim=1)            # 넘은 임계값 비율 = 0~1
            else:                                           # pinball → 중앙분위
                v = torch.sigmoid(z[:, self.ord_k // 2])
            k = self.stif_idx
            out = torch.cat([out[:, :k], v.unsqueeze(1), out[:, k + 1:]], dim=1)

        return out


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 스모크: data_preprocessing 이 만드는 형상(seq (L,56) + scalars (44,)) 기준
    import config as C

    m = MultiTargetRegressorAux(in_channels=56, n_scalars=44,
                                hidden_dim=C.HIDDEN, dropout=C.DROPOUT)
    x = torch.randn(4, C.RESAMPLE_LEN, 56)
    s = torch.randn(4, 44)
    y = m(x, s)
    print(f"params={count_params(m):,}  out={tuple(y.shape)} "
          f"range=[{y.min():.3f},{y.max():.3f}]")
