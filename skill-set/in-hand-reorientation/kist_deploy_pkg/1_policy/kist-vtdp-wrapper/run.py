#!/usr/bin/env python3
"""실기 배포 루프 — 마일스톤 0 (`docs/DEPLOY.md`).

    # 로봇 없이: 학습 h5 를 100Hz 로 재생해 배포 경로 전체를 검증
    python3 run.py --run runs/loop/r6_x_tacdrop_vt --replay auto --dry_run

    # 리그가 올라오면 (아직 미구현 — §입력 백엔드)
    python3 run.py --run runs/loop/r6_x_tacdrop_vt --source ros2 --dry_run

`refer/diffusion_policy/run.py` 는 `OBS_DIM=46` 순수 state 정책이라 **이식이 아니라 신규 구축**
이다(DEPLOY §6). 이 파일이 지키는 계약은 전부 `docs/DEPLOY.md` 에 있고, 여기서는 그 계약을
**코드로** 고정한다:

  §1 링버퍼 깊이가 모달마다 다르다 — 촉각 21 / rgb·state 7 프레임. 차기 전 추론 ❌
  §3 `hp` EMA 는 **에피소드 하나 동안 끊김 없이** 굴린다. 윈도우마다 초기화 ❌ (τ=100 프레임)
  §4 정규화는 `ckpt['obs_norm']`·`act_norm` 그대로. 재계산 ❌ (상수 채널 std=1)
  §5 액션은 역정규화 후 count 단위로 전송

⚠️ **실로봇 명령 전송은 사용자 confirm 없이 ❌** (프로젝트 규약). `--dry_run` 이 기본값이고,
발행 백엔드는 아직 없다 — 이 파일은 지금 **계산·주기까지만** 검증한다.
"""
import argparse, json, sys, time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_weights                                        # noqa: E402
from vtdp.config import load_config                                   # noqa: E402
from vtdp.data import (HP_TAU_DEFAULT, Normalizer, apply_tactile_transform,  # noqa: E402
                       build_datasets)
from vtdp.policy import build_policy                                  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# DEPLOY §3-1 — 인과적 EMA 고역통과의 **스트리밍** 판.
# `vtdp.data.causal_highpass` 와 같은 점화식이어야 한다(아래 --self_test 가 대조한다).
# ══════════════════════════════════════════════════════════════════════════
# ══ 모델 레지스트리 ═══════════════════════════════════════════════════════════
# 실기에서 모델을 갈아 끼우며 실험하려면 run 경로를 외우고 있어야 했다 → 짧은 id 로 준다.
# **이 딕트가 정본이다** — 리그 쪽 `run_kist_vtdp.py` 가 이 모듈을 import 해서 같은 표를 쓴다
# (계약이 두 벌이 되면 어긋난다: LEARNINGS 2026-08-11).
# J 는 홀드아웃 액션 MAE[count], **낮을수록 좋다**. 000·1xx 은 레몬 8데모, 002·003 은
# 레몬 8 + 복숭아 7 을 못박은 홀드아웃이라 **레몬 칸끼리만** 직접 비교된다(done_v7 §2B P5).
MODEL_TYPES: dict[str, tuple[str, str]] = {
    # ── 데모용 4종 ────────────────────────────────────────────────────────────
    "000": ("runs/loop/r6_x_tacdrop_vt",
            "레몬 · 시각+촉각 · J 405.45 · ⭐ 유일하게 실기 rollout 검증됨 (기본값)"),
    "001": ("runs/loop/v7_lemon_aug_vt",
            "레몬 · 시각+촉각 + 가림 aug · J 416.36 · 가림 열화 +0.14% (내성 최고)"),
    "002": ("runs/loop/v7_both_v",
            "레몬+복숭아 · 시각만 · 레몬 398.83 / 복숭아 407.84 · ⭐ 오프라인 최고, 한 모델 두 과일"),
    "003": ("runs/loop/v7_both_aug_v",
            "레몬+복숭아 · 시각만 + 가림 aug · 레몬 403.10 / 복숭아 399.75 (복숭아 우세)"),
    # ── 대조 arm: 실기에서 V vs VT 를 가른다 (결정 큐 후보 E) ─────────────────
    "100": ("runs/loop/trial_5_xattn_v",
            "레몬 · 시각만 · J 416.62 · = 000 의 짝 (촉각만 뺐다)"),
    "101": ("runs/loop/m2_mdrop_v",
            "레몬 · 시각만 · J 401.15 · 레몬 전용 시각-only 최고"),
    "102": ("runs/loop/v7_lemon_aug_v",
            "레몬 · 시각만 + 가림 aug · J 415.67 · = 001 의 짝"),
    # ── v8 (가림 aug 결함 2건 수정) ───────────────────────────────────────────
    "010": ("runs/loop/v8_lemon_aug2_vt", "레몬 · 시각+촉각 + 가림 aug v2 (shared mask + 전면가림)"),
    "110": ("runs/loop/v8_lemon_aug2_v",  "레몬 · 시각만 + 가림 aug v2 · = 010 의 짝"),
    # ── 2xx · 단일 과일 전용 (v2 데이터 세대, 구세대 아키텍처) ────────────────
    # ⚠️ 0xx·1xx 과 **아키텍처가 다르다**: concat + unet1d + resnet18 **파인튜닝**
    #    (`configs/_base_v2.yaml`). 0xx 는 cross_attn + frozen resnet 이다.
    #    그래서 200 과 002 의 차이에는 데이터 축과 아키텍처 축이 같이 섞여 있다.
    # J 는 **복숭아 홀드아웃 7 데모** 기준이라 002 의 복숭아 칸(407.84)과만 같은 자다
    #    (`tools/subset_delta.py runs/23_peach_v runs/loop/v7_both_v --contains peach`:
    #     Δ +4.20% ±4.51% = 판정선 안 = 합본이 복숭아를 손해 봤다는 근거는 없다).
    "200": ("runs/23_peach_v",
            "복숭아 전용 · 시각만 · J 391.42 (복숭아 7데모) · 상수 674.3 대비 −44.8%"),
    # 짝 VT(`runs/22_peach_vt`, J 403.14)는 **일부러 안 올린다** — 복숭아 촉각이 물리적으로
    # 죽은 채 수집돼(part1) Δ +2.99%±2.79 로 유의하게 나쁘다. 재수집 후 다시 만든다(README §🍑).
}


def resolve_model_type(spec: str, repo: Path | None = None) -> str:
    """`--model_type` → run 상대경로. 없는 id 면 표를 보여주고 종료한다."""
    if spec not in MODEL_TYPES:
        raise SystemExit(f"❌ --model_type {spec!r} 없음.\n{format_model_types(repo)}")
    rel = MODEL_TYPES[spec][0]
    if repo is not None:
        d = Path(repo) / rel
        missing = [f for f in ("config.yaml", "best.pt") if not (d / f).exists()]
        if missing:
            raise SystemExit(f"❌ --model_type {spec} → {rel} 에 {missing} 가 없다 "
                             f"(아직 학습 안 끝났거나 경로가 옮겨졌다)")
    return rel


def format_model_types(repo: Path | None = None) -> str:
    out = ["  사용 가능한 --model_type:"]
    for k, (rel, desc) in MODEL_TYPES.items():
        ok = ""
        if repo is not None:
            d = Path(repo) / rel
            ok = "" if (d / "best.pt").exists() else "  ⚠️ best.pt 없음"
        out.append(f"    {k}  {rel:32s} {desc}{ok}")
    return "\n".join(out)


class StreamingHighpass:
    def __init__(self, tau: float = HP_TAU_DEFAULT):
        self.a = 1.0 / float(tau)
        self.m = None
        self.n = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.m is None:
            self.m = x.copy()                       # m[0] = x[0]
        else:
            self.m = self.m + self.a * (x - self.m)
        self.n += 1
        return x - self.m


class RingBuffer:
    """100 Hz 프레임 링버퍼. `offsets` 는 `[-(h-1)s, …, -s, 0]`."""

    def __init__(self, horizon: int, stride: int):
        self.offsets = [-(horizon - 1 - j) * stride for j in range(horizon)]
        self.depth = (horizon - 1) * stride + 1
        self.buf = deque(maxlen=self.depth)

    def push(self, frame):
        self.buf.append(np.asarray(frame, dtype=np.float32))

    @property
    def ready(self) -> bool:
        return len(self.buf) == self.depth        # DEPLOY §1: 차기 전 추론 ❌

    def window(self) -> np.ndarray:
        b = list(self.buf)
        return np.stack([b[self.depth - 1 + o] for o in self.offsets])


# ══════════════════════════════════════════════════════════════════════════
class Deployer:
    def __init__(self, run_dir: Path, ckpt_name: str, device: str,
                 exec_horizon: int | None = None):
        self.cfg = load_config(str(run_dir / "config.yaml"))
        self.dev = torch.device(device)
        ck = torch.load(run_dir / ckpt_name, map_location=self.dev, weights_only=False)

        self.policy = build_policy(self.cfg).to(self.dev)
        self.prefer = load_weights(self.policy, ck)
        self.policy.eval()                         # DEPLOY §2: frozen BN 은 eval 필수

        # DEPLOY §4 — 정규화는 ckpt 것을 그대로. 재계산 ❌
        self.obs_norm = {k: Normalizer(**v) for k, v in ck["obs_norm"].items()}
        self.act_norm = Normalizer(**ck["act_norm"])

        self.ospec = self.cfg["obs_spec"]
        act = self.cfg["action"]
        self.stride = int(act.get("stride", 5))
        self.exec_h = int(exec_horizon or act["exec_horizon"])
        self.pred_h = int(act["pred_horizon"])
        self.hz = 100.0 / self.stride
        self.tick_budget_ms = 1000.0 / self.hz

        self.rings, self.hp = {}, {}
        for name, s in self.ospec.items():
            self.rings[name] = RingBuffer(int(s["horizon"]), int(s["stride"]))
            if name == "tactile" and "hp" in _as_list(s.get("transform")):
                kw = s.get("transform_kwargs") or {}
                self.hp[name] = StreamingHighpass(float(kw.get("tau", HP_TAU_DEFAULT)))
        self.warmup_frames = max(r.depth for r in self.rings.values())

    # ── 100 Hz 프레임 하나 ────────────────────────────────────────────
    def push(self, frame: dict):
        """frame = {모달명: 원시 100Hz 벡터}. `hp` 는 여기서 **매 프레임** 굴린다."""
        for name, ring in self.rings.items():
            x = frame[name]
            if name in self.hp:
                x = self.hp[name](x)               # DEPLOY §3-1: 끊김 없이
            ring.push(x)

    @property
    def ready(self) -> bool:
        return all(r.ready for r in self.rings.values())

    # ── 20 Hz 제어 틱에서의 추론 ──────────────────────────────────────
    @torch.no_grad()
    def infer(self) -> np.ndarray:
        obs = {}
        for name, ring in self.rings.items():
            w = ring.window()
            s = self.ospec[name]
            if name == "tactile":
                t = _as_list(s.get("transform"))
                rest = [x for x in t if x != "hp"]  # hp 는 스트리밍으로 이미 적용됐다
                if rest:
                    w = apply_tactile_transform(w, rest, s.get("transform_kwargs"))
            if name in self.obs_norm:               # rgb 는 대상 아님 (DEPLOY §4)
                w = self.obs_norm[name].normalize(w)
            obs[name] = torch.as_tensor(np.asarray(w, dtype=np.float32),
                                        device=self.dev).unsqueeze(0)
        a = self.policy.sample(obs)[0].cpu().numpy()
        return a * self.act_norm.std + self.act_norm.mean   # DEPLOY §5: count 단위


def _as_list(t):
    return [] if t is None else ([t] if isinstance(t, str) else list(t))


# ══════════════════════════════════════════════════════════════════════════
def replay_frames(cfg, which: str):
    """홀드아웃 데모 하나를 100 Hz 프레임 스트림으로.

    RGB 디코드는 **학습과 같은 경로**(`VTWindowDataset._decode` — crop→224² BILINEAR→/255)를
    그대로 쓴다. 두 번째 전처리 경로를 만들면 조용히 어긋난다(LEARNINGS 2026-08-10).
    `rgb_index == -1`(카메라 늦게 켜짐) 프레임은 건너뛴다 — DEPLOY §1.
    """
    _, val_ds, _, _, _ = build_datasets(cfg, verbose=False)
    di = 0 if which == "auto" else next(i for i, d in enumerate(val_ds.demos)
                                        if which in d.name)
    dem = val_ds.demos[di]
    for t in range(dem.n):
        if dem.rgb_index is not None and int(dem.rgb_index[t]) < 0:
            continue                               # 프레임이 아직 없다 → 추론을 미룬다
        f = {}
        for name in cfg["obs_spec"]:
            f[name] = (val_ds._decode(di, int(dem.rgb_index[t])) if name == "rgb"
                       else dem.lowdim[name][t])
        yield f


def self_test():
    """스트리밍 hp 가 배치 `causal_highpass` 와 **같은 값**인지 (LEARNINGS: 두 경로 대조)."""
    from vtdp.data import causal_highpass
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 12)).astype(np.float32)
    x[:, 3] = 1.234                                    # 상수 채널 → 정확히 0 이어야
    batch = causal_highpass(x, HP_TAU_DEFAULT)
    st = StreamingHighpass(HP_TAU_DEFAULT)
    stream = np.stack([st(x[t]) for t in range(len(x))])
    err = np.abs(batch - stream).max()
    zero = np.abs(stream[:, 3]).max()
    print(f"  스트리밍 hp vs 배치 hp : max|Δ| = {err:.3e}   {'✅' if err < 1e-5 else '❌'}")
    print(f"  상수 채널이 정확히 0    : max|y| = {zero:.3e}   {'✅' if zero == 0.0 else '❌'}")
    return err < 1e-5 and zero == 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run 디렉터리 (config.yaml + best.pt)")
    ap.add_argument("--model_type", default=None,
                    help="run 대신 짧은 id 로 고른다 (예: 000). --list_models 로 목록")
    ap.add_argument("--list_models", action="store_true", help="--model_type 표를 찍고 종료")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--source", default="replay", choices=["replay", "ros2"])
    ap.add_argument("--replay", default="auto", help="홀드아웃 데모 선택 ('auto' 또는 이름 일부)")
    ap.add_argument("--exec_horizon", type=int, default=None, help="추론 주기 override (스트레스)")
    ap.add_argument("--realtime", action="store_true", help="100Hz 벽시계로 페이싱")
    ap.add_argument("--max_ticks", type=int, default=200)
    ap.add_argument("--dry_run", action="store_true", default=True)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    repo = Path(__file__).resolve().parent
    if a.list_models:
        print(format_model_types(repo)); sys.exit(0)
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if (a.run is None) == (a.model_type is None):
        sys.exit("❌ --run 과 --model_type 중 **정확히 하나**를 줄 것.\n"
                 + format_model_types(repo))
    if a.model_type is not None:
        a.run = str(repo / resolve_model_type(a.model_type, repo))
        print(f"\n  [model_type {a.model_type}] {MODEL_TYPES[a.model_type][1]}")
    if a.source == "ros2":
        sys.exit("❌ ros2 백엔드 미구현 — 리그 토픽(/hand/*, 카메라, paxini)이 올라온 뒤 붙인다.\n"
                 "   현재 이 머신에 뜬 토픽은 Isaac Sim 것뿐이다 (plan_v5 §4 분기 3).")

    dep = Deployer(Path(a.run), a.ckpt, a.device, a.exec_horizon)
    print(f"\n  {a.run}  ({dep.prefer} 가중치)")
    print(f"  제어 {dep.hz:.0f} Hz (틱 {dep.tick_budget_ms:.0f} ms) · "
          f"exec_horizon {dep.exec_h} → 추론 {dep.exec_h * dep.tick_budget_ms:.0f} ms 마다")
    print(f"  링버퍼: " + " · ".join(f"{k} {r.depth}프레임" for k, r in dep.rings.items())
          + f"  → 예열 {dep.warmup_frames}프레임 ({dep.warmup_frames * 10} ms)")
    print(f"  hp EMA: " + (", ".join(dep.hp) if dep.hp else "없음")
          + "  (에피소드 동안 연속)")
    print(f"  발행: {'DRY-RUN (발행 안 함)' if a.dry_run else '❌ 미구현'}\n")

    tick_ms, infer_ms, misses, n_tick, n_infer, warm = [], [], 0, 0, 0, False
    plan = None
    t_next = time.perf_counter()
    for i, f in enumerate(replay_frames(dep.cfg, a.replay)):
        dep.push(f)
        if a.realtime:
            t_next += 0.01
            d = t_next - time.perf_counter()
            if d > 0:
                time.sleep(d)
        if i % dep.stride or not dep.ready:
            continue
        if not warm:
            # 🔴 첫 추론은 CUDA 콜드스타트로 ~265 ms 다(실측) = 예산의 5배.
            # 버퍼가 찬 직후, **잡기 시작 전에** 몇 번 굴려 두는 것이 배포 계약이다.
            for _ in range(3):
                dep.infer()
            warm = True
        t0 = time.perf_counter()
        if n_tick % dep.exec_h == 0:               # exec_horizon 스텝마다 한 번 추론
            plan = dep.infer()
            infer_ms.append((time.perf_counter() - t0) * 1000)
            n_infer += 1
        _ = plan[n_tick % dep.exec_h]              # 이번 틱에 나갈 16-D 타겟 [count]
        dt = (time.perf_counter() - t0) * 1000
        tick_ms.append(dt)
        misses += dt > dep.tick_budget_ms
        n_tick += 1
        if n_tick >= a.max_ticks:
            break

    tick = np.array(tick_ms); inf = np.array(infer_ms)
    keep = 100.0 * (1 - misses / max(n_tick, 1))
    print(f"  제어 틱 {n_tick}회 (추론 {n_infer}회)")
    print(f"  틱 소요   p50 {np.percentile(tick,50):6.2f}  p90 {np.percentile(tick,90):6.2f}"
          f"  max {tick.max():6.2f} ms   (예산 {dep.tick_budget_ms:.0f} ms)")
    print(f"  추론 소요 p50 {np.percentile(inf,50):6.2f}  p90 {np.percentile(inf,90):6.2f}"
          f"  max {inf.max():6.2f} ms")
    print(f"  ▸ 주기 유지율 {keep:.2f}%  (초과 {misses}틱)  "
          f"{'✅ 마일스톤 0 통과' if keep >= 99.0 else '❌'}\n")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"run": a.run, "source": a.source, "realtime": a.realtime,
             "hz": dep.hz, "exec_horizon": dep.exec_h, "n_tick": n_tick, "n_infer": n_infer,
             "tick_ms": {"p50": float(np.percentile(tick, 50)),
                         "p90": float(np.percentile(tick, 90)), "max": float(tick.max())},
             "infer_ms": {"p50": float(np.percentile(inf, 50)),
                          "p90": float(np.percentile(inf, 90)), "max": float(inf.max())},
             "misses": int(misses), "keep_pct": keep}, indent=2), "utf-8")
        print(f"  → {a.json}")


if __name__ == "__main__":
    main()
