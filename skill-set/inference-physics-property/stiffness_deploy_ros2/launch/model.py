"""
model.py  (회귀용 / 소규모 데이터 대응)
=======================================
기존 분류 모델(StiffnessClassifier 25-class)을 회귀로 전환 + 축소.

왜 바꿨나:
  - 회귀 우선 전략: 출력 1개(정규화 강성). 분류는 후처리(경계)로.
  - 데이터가 작음(136 데모, 6개체, 시퀀스 ~20). 기존 모델은 수십만+ 파라미터라
    과적합/개체암기 위험. 채널·hidden 축소, 공격적 다운샘플 제거.
  - 짧은 시퀀스에 stride=2 누적은 시퀀스를 거의 없앰 -> 약한 다운샘플만.

엄지 강조(옵션):
  - use_finger_attention=True 면 입력 채널에 학습가능한 가중(attention)을 부여.
    엄지 채널이 중요하면 모델이 그 가중을 키우게 됨(데이터가 결정).
  - 기본 False: 마스킹 O/X 비교 때 변수 안 섞이게. 비교 끝나면 켜서 추가실험.

출력:
  forward -> (B,) 회귀값. use_sigmoid=True 면 0~1 (타깃이 정규화강성이라 권장).
"""
import torch
import torch.nn.functional as F
from torch import nn


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + self.shortcut(x))
        return self.drop(out)


class ChannelAttention(nn.Module):
    """입력 채널별 학습가능 가중. 엄지 채널이 중요하면 자연히 가중↑.
       (명시적 '엄지=중요' 대신 데이터가 결정 -> 6개체에 안전)"""
    def __init__(self, n_channels):
        super().__init__()
        # 채널별 게이트: 시퀀스 평균 -> FC -> sigmoid 가중
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.GELU(),
            nn.Linear(n_channels, n_channels), nn.Sigmoid(),
        )

    def forward(self, x):  # x: (B,T,C)
        w = self.fc(x.mean(dim=1))          # (B,C) 채널 가중
        return x * w.unsqueeze(1)           # 채널별 스케일


class StiffnessRegressor(nn.Module):
    """소규모 데이터용 회귀 모델. CNN(약한 다운샘플) + 1층 LSTM + 회귀헤드.
       in_channels=40 (joint16+ft12+resultant12) 기준.

    heteroscedastic=True 면 출력이 (μ, log σ²) 튜플 — 강성 예측(μ)과 그 예측의
    불확실성(σ²)을 함께 냄. log-variance 로 내는 이유: σ²>0 제약을 자동 만족
    시키면서 학습 초반 수치 폭주를 막음(exp 로 복원). 학습은 β-NLL 손실
    (train.py beta_nll_loss) 사용 — 순수 NLL 의 mean-fit 악화 문제 회피
    (Seitzer et al. 2022, ICLR)."""

    def __init__(self, in_channels=40, hidden_dim=48, num_layers=1, dropout=0.3,
                 use_sigmoid=True, use_finger_attention=False, heteroscedastic=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.use_attn = use_finger_attention
        self.heteroscedastic = heteroscedastic

        if use_finger_attention:
            self.attn = ChannelAttention(in_channels)

        # 약한 다운샘플: stride=1 유지(짧은 시퀀스 보존), 채널 작게
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
        )
        self.res1 = ResBlock1D(32, 48, stride=1, dropout=dropout)   # 다운샘플 안 함

        # LSTM: num_layers>1 이면 층 사이에 dropout 적용(1층이면 dropout 인자 무시됨).
        #   큰 데이터셋(예 전체 2662)에서 2층 등 깊은 LSTM 시도용.
        self.lstm = nn.LSTM(input_size=48, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0))

        out_dim = 2 if heteroscedastic else 1   # [μ, log σ²] or [μ]
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
        )

    def forward(self, x, lengths=None, mask=None):
        # x: (B,T,C)
        if self.use_attn:
            x = self.attn(x)

        h = x.transpose(1, 2)                # (B,C,T)
        h = self.init_conv(h)
        h = self.res1(h)
        h = h.transpose(1, 2)                # (B,T,C')

        # 유효 길이로 LSTM 마지막 스텝 선택 (stride=1 이라 길이 유지)
        t_new = h.shape[1]
        if mask is not None:
            mask_f = mask.unsqueeze(1).float()
            mask_d = F.adaptive_max_pool1d(mask_f, output_size=t_new).squeeze(1).bool()
            valid_len = mask_d.sum(dim=1).long().clamp(min=1)
        elif lengths is not None:
            valid_len = lengths.to(h.device).clamp(min=1, max=t_new)
        else:
            valid_len = torch.full((h.size(0),), t_new, dtype=torch.long, device=h.device)

        lstm_out, _ = self.lstm(h)
        idx = torch.arange(h.size(0), device=h.device)
        last = lstm_out[idx, valid_len - 1, :]    # (B,hidden)

        out = self.head(last)                     # (B, out_dim)
        if self.heteroscedastic:
            mu, log_var = out[:, 0], out[:, 1]
            if self.use_sigmoid:
                mu = torch.sigmoid(mu)             # μ만 0~1 (정규화강성). log_var는 제약 없음.
            return mu, log_var
        out = out.squeeze(-1)                      # (B,)
        if self.use_sigmoid:
            out = torch.sigmoid(out)              # 0~1 (정규화강성)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import math as _math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-_math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class StiffnessTransformerRegressor(nn.Module):
    """CNN(약한 다운샘플) + Transformer 회귀. 소규모 데이터용으로 축소.
       StiffnessRegressor 와 동일 인터페이스: forward(x,lengths,mask)->(B,)."""

    def __init__(self, in_channels=40, hidden_dim=64, num_layers=2, nhead=4,
                 max_len=256, dropout=0.3, use_sigmoid=True,
                 use_finger_attention=False, apply_input_conv=True,
                 heteroscedastic=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.use_attn = use_finger_attention
        self.apply_input_conv = apply_input_conv
        self.heteroscedastic = heteroscedastic

        if use_finger_attention:
            self.attn = ChannelAttention(in_channels)

        if apply_input_conv:
            self.init_conv = nn.Sequential(
                nn.Conv1d(in_channels, 32, 5, stride=1, padding=2, bias=False),
                nn.BatchNorm1d(32), nn.GELU(),
            )
            self.res1 = ResBlock1D(32, 48, stride=1, dropout=dropout)
            feat_dim = 48
        else:
            feat_dim = in_channels

        self.input_layer = nn.Linear(feat_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, dropout, max_len)
        self.norm = nn.LayerNorm(hidden_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers,
                                             enable_nested_tensor=False)
        out_dim = 2 if heteroscedastic else 1   # [μ, log σ²] or [μ]
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim//2, out_dim),
        )

    def forward(self, x, lengths=None, mask=None):
        if self.use_attn:
            x = self.attn(x)

        if self.apply_input_conv:
            h = x.transpose(1, 2)
            h = self.init_conv(h)
            h = self.res1(h)
            h = h.transpose(1, 2)               # (B,T,feat)
            t_new = h.shape[1]
            if mask is not None:
                mf = mask.unsqueeze(1).float()
                md = F.adaptive_max_pool1d(mf, output_size=t_new).squeeze(1).bool()
                valid_len = md.sum(1).long().clamp(min=1)
            elif lengths is not None:
                valid_len = lengths.clamp(min=1, max=t_new)
            else:
                valid_len = torch.full((h.size(0),), t_new, dtype=torch.long, device=h.device)
        else:
            h = x
            t_new = h.shape[1]
            valid_len = (lengths.clamp(min=1, max=t_new) if lengths is not None
                         else torch.full((h.size(0),), t_new, dtype=torch.long, device=h.device))

        h = F.gelu(self.input_layer(h))
        h = self.pos(h)
        h = self.norm(h)

        pad_mask = torch.arange(h.size(1), device=h.device).expand(len(valid_len), h.size(1)) >= valid_len.unsqueeze(1)
        h = h.masked_fill(pad_mask.unsqueeze(2), 0.0)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        valid = ~pad_mask
        pooled = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        out = self.head(pooled)                 # (B, out_dim)
        if self.heteroscedastic:
            mu, log_var = out[:, 0], out[:, 1]
            if self.use_sigmoid:
                mu = torch.sigmoid(mu)
            return mu, log_var
        out = out.squeeze(-1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


# 모델 레지스트리 (train.py 에서 이름으로 선택)
MODEL_REGISTRY = {
    "lstm": StiffnessRegressor,
    "transformer": StiffnessTransformerRegressor,
}


if __name__ == "__main__":
    x = torch.randn(4, 20, 40); lengths = torch.tensor([20, 18, 15, 20])
    for name, cls in MODEL_REGISTRY.items():
        m = cls(in_channels=40)
        y = m(x, lengths=lengths)
        print(f"{name}: params={count_params(m):,}, out={tuple(y.shape)}, "
              f"range=[{y.min():.3f},{y.max():.3f}]")
