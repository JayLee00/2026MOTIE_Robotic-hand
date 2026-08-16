#!/usr/bin/env python3
"""YAML config 로드 · 검증 · 병합.

설계 원칙 하나: **오류는 학습 시작 전에, 사람이 읽을 수 있는 문장으로.**
config 오타 때문에 30분 학습 돌리고 나서 죽는 상황을 만들지 않는다.

    cfg = load_config("configs/01_separated.yaml")
    cfg = load_config("configs/01_separated.yaml", overrides=["model.fusion=gated"])
    policy = build_policy(cfg)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from vtdp.registry import available

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

_MISSING = object()          # "키가 아예 없다" 와 "None 으로 줬다" 를 구분하는 표식


# ══════════════════════════════════════════════════════════════════════════
def _deep_merge(base: dict, over: dict) -> dict:
    """재귀 병합. 단 `*_kwargs` 로 끝나는 키는 **통째 교체**한다.

    kwargs 는 자기 모듈 선택에 종속된 값이다. 자식이 `fusion: cross_attn` → `concat` 으로
    바꿨는데 부모의 `fusion_kwargs: {n_head: 4}` 가 merge 되어 남으면 concat 생성자가
    모르는 인자를 받고 터진다. 그리고 `fusion_kwargs: {}` 로 지우려 해도
    빈 dict 를 merge 하면 부모가 그대로 남아 **지울 방법이 없다.**
    → 모듈을 바꾸면 그 kwargs 도 함께 갈리는 게 유일하게 안전한 규칙이다.
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.endswith("_kwargs"):
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(s: str) -> Any:
    """CLI override 값 문자열을 파이썬 값으로. YAML 파서를 그대로 쓴다."""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def _apply_override(cfg: dict, expr: str) -> None:
    """`a.b.c=value` 형태 하나를 적용. 없는 경로는 만들지 않고 에러를 낸다."""
    if "=" not in expr:
        raise ValueError(f"override 는 'key.path=value' 형식이어야 한다 — 받은 값: {expr!r}")
    path, raw = expr.split("=", 1)
    keys = path.split(".")
    node = cfg
    for i, k in enumerate(keys[:-1]):
        if k not in node or not isinstance(node[k], dict):
            here = ".".join(keys[:i + 1])
            raise KeyError(f"override 경로 {path!r} 중 {here!r} 가 config 에 없다")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(
            f"override 경로 {path!r} 가 config 에 없다. "
            f"'{'.'.join(keys[:-1])}' 아래 가능한 키: {sorted(node)}")
    node[keys[-1]] = _parse_scalar(raw)


# ══════════════════════════════════════════════════════════════════════════
def _check_kwargs(kind: str, name: str | None, kwargs: dict | None,
                  errs: list[str], where: str) -> None:
    """`kwargs` 의 키가 해당 모듈 생성자에 실제로 있는지 확인한다."""
    if not kwargs or not name:
        return
    from vtdp.registry import _REGISTRY
    ctor = _REGISTRY.get(kind, {}).get(name)
    if ctor is None:
        return                       # 이름 오류는 다른 검사가 이미 잡는다
    import inspect
    sig = inspect.signature(ctor)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return                       # **kwargs 를 받으면 뭐든 통과
    known = set(sig.parameters)
    unknown = [k for k in kwargs if k not in known]
    if unknown:
        # policy 가 주입하는 인자는 사용자가 쓸 일이 없으니 후보에서 뺀다
        injected = {"token_spec", "d_model", "cond_dim", "in_dim", "in_shape",
                    "horizon", "action_dim", "pred_horizon", "denoiser", "self"}
        opts = sorted(known - injected)
        errs.append(f"{where}: {kind}/{name} 가 모르는 인자 {unknown}. "
                    f"가능: {opts}  "
                    f"(상속 체인에서 {kind} 를 바꿨는데 부모 kwargs 가 남지 않았는지 확인)")


def validate(cfg: dict) -> dict:
    """config 를 학습 전에 전수 검사한다. 통과하면 build_policy 가 실패할 이유가 거의 없다."""
    errs: list[str] = []

    # ── 필수 최상위 키 ──
    for k in ("obs_spec", "action", "model"):
        if k not in cfg:
            errs.append(f"최상위 키 {k!r} 가 없다")
    if errs:
        raise ValueError("config 오류:\n  - " + "\n  - ".join(errs))

    # ── obs_spec ──
    if not cfg["obs_spec"]:
        errs.append("obs_spec 이 비었다 — 모달리티가 최소 하나는 있어야 한다")
    for name, spec in cfg["obs_spec"].items():
        for k in ("kind", "shape", "horizon", "encoder"):
            if k not in spec:
                errs.append(f"obs_spec.{name}: {k!r} 가 없다")
        kind = spec.get("kind")
        if kind not in ("state", "tactile", "vision"):
            errs.append(f"obs_spec.{name}.kind={kind!r} — state|tactile|vision 중 하나여야 한다")
            continue
        enc = spec.get("encoder")
        if enc not in available(kind):
            errs.append(f"obs_spec.{name}.encoder={enc!r} 없음. "
                        f"가능한 {kind} 인코더: {available(kind)}")
        shape = spec.get("shape")
        if kind == "vision":
            if not (isinstance(shape, (list, tuple)) and len(shape) == 3):
                errs.append(f"obs_spec.{name}.shape 는 vision 이면 [C,H,W] — 받은 값 {shape!r}")
        elif not isinstance(shape, int):
            errs.append(f"obs_spec.{name}.shape 는 {kind} 이면 정수 차원 — 받은 값 {shape!r}")
        h = spec.get("horizon")
        if not isinstance(h, int) or h < 1:
            errs.append(f"obs_spec.{name}.horizon 은 1 이상 정수 — 받은 값 {h!r}")
        s = spec.get("stride")
        if s is not None and (not isinstance(s, int) or s < 1):
            errs.append(f"obs_spec.{name}.stride 는 1 이상 정수 — 받은 값 {s!r}")

        # ── keys: 이 모달리티를 구성하는 HDF5 키 ──
        # 여기가 틀리면 학습이 아니라 **조용히 엉뚱한 신호를 학습**하므로 엄격하게 본다.
        from vtdp.data import ALWAYS_ZERO_KEYS, KEY_DIMS, RGB_INDEX_KEY
        ks = spec.get("keys")
        if not isinstance(ks, (list, tuple)) or not ks:
            errs.append(f"obs_spec.{name}.keys 가 없거나 비었다 — HDF5 키 목록이 필요하다")
        elif kind == "vision":
            if list(ks) != [RGB_INDEX_KEY]:
                errs.append(f"obs_spec.{name}.keys 는 vision 이면 ['{RGB_INDEX_KEY}'] "
                            f"하나여야 한다 — 받은 값 {list(ks)}")
        else:
            unknown = [k for k in ks if k not in KEY_DIMS]
            if unknown:
                errs.append(f"obs_spec.{name}.keys 에 모르는 HDF5 키 {unknown}. "
                            f"레코더가 쓰는 키만 가능하다(vtdp/data.py KEY_DIMS 참조)")
            zero = [k for k in ks if k in ALWAYS_ZERO_KEYS]
            if zero:
                errs.append(f"obs_spec.{name}.keys 의 {zero} 는 실데이터에서 전 구간 0 이다 "
                            f"— 관측으로 쓸 수 없다(vtdp/data.py ALWAYS_ZERO_KEYS 참조)")
            elif isinstance(shape, int):
                from vtdp.data import transform_out_dim
                total = sum(KEY_DIMS[k] for k in ks)
                detail = " + ".join(f"{k}({KEY_DIMS[k]})" for k in ks)
                tf = spec.get("transform")
                if tf:
                    n_part = int((spec.get("transform_kwargs") or {}).get("n_part", 4))
                    try:
                        total = transform_out_dim(total, tf, n_part)
                        detail += f" --{tf}--> {total}"
                    except ValueError as e:
                        errs.append(f"obs_spec.{name}.transform: {e}")
                        total = shape          # 아래 중복 오류 억제
                if total != shape:
                    errs.append(f"obs_spec.{name}: keys 합 {total} != shape {shape}   "
                                f"[{detail}]")

    # ── action ──
    a = cfg.get("action", {})
    from vtdp.data import KEY_DIMS as _KD
    ak = a.get("key")
    if not ak:
        errs.append("action.key 가 없다 (예: 04_hand_j_tar)")
    elif ak not in _KD:
        errs.append(f"action.key={ak!r} 는 레코더가 쓰는 키가 아니다")
    elif isinstance(a.get("dim"), int) and _KD[ak] != a["dim"]:
        errs.append(f"action.dim({a['dim']}) != {ak} 의 실제 차원 {_KD[ak]}")
    if not (cfg.get("data") or {}).get("root"):
        errs.append("data.root 가 없다 — HDF5 폴더 경로가 필요하다 (--data 로도 줄 수 있다)")
    for k in ("dim", "pred_horizon"):
        if not isinstance(a.get(k), int) or a[k] < 1:
            errs.append(f"action.{k} 는 1 이상 정수 — 받은 값 {a.get(k)!r}")
    if isinstance(a.get("exec_horizon"), int) and isinstance(a.get("pred_horizon"), int):
        if a["exec_horizon"] > a["pred_horizon"]:
            errs.append(f"action.exec_horizon({a['exec_horizon']}) > "
                        f"pred_horizon({a['pred_horizon']}) — 예측한 것보다 많이 실행할 수 없다")

    # ── model ──
    m = cfg.get("model", {})
    if m.get("fusion") not in available("fusion"):
        errs.append(f"model.fusion={m.get('fusion')!r} 없음. 가능: {available('fusion')}")
    head = m.get("head", "diffusion")
    if head not in available("head"):
        errs.append(f"model.head={head!r} 없음. 가능: {available('head')}")
    if m.get("denoiser", "unet1d") not in available("denoiser"):
        errs.append(f"model.denoiser={m.get('denoiser')!r} 없음. 가능: {available('denoiser')}")

    d_model, cond_dim = m.get("d_model", 256), m.get("cond_dim", 256)
    if m.get("fusion") == "passthrough" and cond_dim != d_model:
        errs.append(f"fusion=passthrough 는 cond_dim({cond_dim}) == d_model({d_model}) 이어야 한다")
    if m.get("fusion") == "gated":
        g = (m.get("fusion_kwargs") or {}).get("gate_on", "tactile")
        if g not in cfg["obs_spec"]:
            errs.append(f"fusion=gated 의 gate_on={g!r} 가 obs_spec 에 없다. "
                        f"가능: {sorted(cfg['obs_spec'])}")
    # ── kwargs 가 실제 생성자 시그니처와 맞는지 ──
    # 상속 체인에서 fusion 을 바꿨는데 부모의 fusion_kwargs 가 남아 따라오는 사고가
    # 가장 흔하다(예: cross_attn 의 n_head 가 concat 으로 넘어옴). build 까지 가면
    # 오류가 늦고 메시지도 멀어서, 여기서 잡는다.
    _check_kwargs("fusion", m.get("fusion"), m.get("fusion_kwargs"), errs, "model.fusion_kwargs")
    _check_kwargs("denoiser", m.get("denoiser", "unet1d"), m.get("denoiser_kwargs"),
                  errs, "model.denoiser_kwargs")
    _check_kwargs("head", head, m.get("head_kwargs"), errs, "model.head_kwargs")
    for name, spec in cfg["obs_spec"].items():
        if spec.get("kind") in ("state", "tactile", "vision"):
            _check_kwargs(spec["kind"], spec.get("encoder"), spec.get("encoder_kwargs"),
                          errs, f"obs_spec.{name}.encoder_kwargs")

    md = m.get("modality_dropout", 0.0)
    if not (0.0 <= float(md) <= 1.0):
        errs.append(f"model.modality_dropout 은 [0,1] — 받은 값 {md!r}")
    if md > 0 and len(cfg["obs_spec"]) < 2:
        errs.append("modality_dropout > 0 인데 모달리티가 1개다 — 가릴 게 없다")
    on = m.get("modality_dropout_on")
    if on is not None:
        if not isinstance(on, list) or not on:
            errs.append(f"model.modality_dropout_on 은 비어 있지 않은 리스트여야 한다 — 받은 값 {on!r}")
        elif [k for k in on if k not in cfg["obs_spec"]]:
            errs.append(f"model.modality_dropout_on 의 {[k for k in on if k not in cfg['obs_spec']]} "
                        f"가 obs_spec 에 없다. 가능: {sorted(cfg['obs_spec'])}")

    if errs:
        raise ValueError("config 오류 %d건:\n  - %s" % (len(errs), "\n  - ".join(errs)))
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """YAML 을 읽고 `_base_` 상속을 풀고 override 를 적용한 뒤 검증한다."""
    path = Path(path)
    if not path.exists():
        cand = CONFIG_DIR / path.name
        if cand.exists():
            path = cand
        else:
            avail = sorted(p.name for p in CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.exists() else []
            raise FileNotFoundError(f"config 없음: {path}. configs/ 에 있는 것: {avail}")

    # `_base_: other.yaml` 체인 상속. 순환은 위 `seen` 이 즉시 잡고, 깊이는 runaway 방지로 제한한다.
    # ⚠️ 8 → 12 (plan_v7, 2026-08-11): 루프 사다리가 누적되며 `r6_x_tacdrop_vt` 가 이미 **7단계**다
    #    (r6 → xattn_vt → ftconv_vt → fthp_vt → frozen_vt → 20_lemon_vt → _base_v2 → _base).
    #    즉 상한 8 에서는 **정본 arm 에서 파생하는 것 자체가 불가능**했다 — 그러면 새 config 이
    #    obs_spec 을 재기술하게 되고, 그건 LEARNINGS 2026-08-07("대조군의 다른 줄을 세어 볼 것")이
    #    경고한 divergence 경로다. 상속으로 동일성을 보장하는 쪽이 안전해서 상한을 올렸다.
    chain: list[Path] = []
    seen: set[Path] = set()
    cur = path
    while True:
        rp = cur.resolve()
        if rp in seen:
            names = " → ".join(p.name for p in chain + [cur])
            raise ValueError(f"_base_ 순환 참조: {names}")
        seen.add(rp)
        chain.append(cur)
        with open(cur, encoding="utf-8") as f:
            node = yaml.safe_load(f) or {}
        base_name = node.get("_base_")
        if not base_name:
            break
        if len(chain) >= 12:
            raise ValueError(f"_base_ 상속이 12단계를 넘는다: "
                             f"{' → '.join(p.name for p in chain)}")
        nxt = cur.parent / base_name
        if not nxt.exists():
            raise FileNotFoundError(f"{cur.name} 의 _base_ 가 가리키는 {base_name} 이 없다")
        cur = nxt

    # 루트(가장 바깥 base)부터 덮어쓰며 병합.
    #
    # ⚠️ `obs_spec` 만은 **deep merge 가 아니라 통째 교체**다.
    #    모달리티 집합이 곧 config 의 정체성이라, merge 하면 자식이 부모의 모달리티를
    #    **뺄 수 없다** (예: vision-only 대조군이 부모의 tactile 을 물려받아버린다).
    #    대조군을 못 만드는 config 시스템은 ablation 도구로 쓸 수 없다.
    #
    # ⚠️ `data.clamp_spec` 도 같은 이유로 **통째 교체**다.
    #    데이터셋 정체성이 파일 목록이라, merge 하면 자식이 부모의 **파일을 뺄 수 없다**
    #    (예: 복숭아 config 이 부모의 레몬 h5 를 물려받아 두 과일이 조용히 섞인다).
    cfg: dict = {}
    for p in reversed(chain):
        with open(p, encoding="utf-8") as f:
            node = yaml.safe_load(f) or {}
        node.pop("_base_", None)
        own_obs = node.pop("obs_spec", None)
        own_clamp = (node.get("data") or {}).pop("clamp_spec", _MISSING)
        cfg = _deep_merge(cfg, node)
        if own_obs is not None:
            cfg["obs_spec"] = copy.deepcopy(own_obs)
        if own_clamp is not _MISSING:                  # None(전부 사용) 도 유효한 값이다
            cfg.setdefault("data", {})["clamp_spec"] = copy.deepcopy(own_clamp)
    cfg["_base_chain"] = [p.name for p in reversed(chain)]

    for expr in overrides or []:
        _apply_override(cfg, expr)

    cfg["_config_path"] = str(path)
    return validate(cfg)


def timing_table(cfg: dict, src_hz: float = 100.0) -> str:
    """모달리티별 샘플링 격자를 사람이 읽을 수 있게 펼친다.

    **주파수 비율 축이 실제로 의도대로 잡혔는지 눈으로 확인하는 용도.**
    stride 와 horizon 은 곱해져야 의미가 나오는데 config 만 보면 안 보인다.
    `lookback` 이 모달리티마다 다르면 대개 실수다(비율이 아니라 히스토리 길이를 바꾼 것).
    """
    default_stride = int(cfg.get("data", {}).get("ds_stride", 5))
    rows, spans = [], {}
    for name in sorted(cfg["obs_spec"]):
        sp = cfg["obs_spec"][name]
        s = int(sp.get("stride") or default_stride)
        h = int(sp["horizon"])
        hz = src_hz / s
        span_ms = (h - 1) * s / src_hz * 1000.0
        spans[name] = round(span_ms, 3)
        offs = [-(h - 1 - k) * s for k in range(h)]
        rows.append(f"  {name:9s} stride={s:<3d} horizon={h:<3d} "
                    f"{hz:6.1f}Hz  lookback={span_ms:6.1f}ms  offsets={offs}")

    a = cfg["action"]
    a_stride = int(a.get("stride") or default_stride)
    a_hz = src_hz / a_stride
    a_span = (a["pred_horizon"] - 1) * a_stride / src_hz * 1000.0
    rows.append(f"  {'action':9s} stride={a_stride:<3d} horizon={a['pred_horizon']:<3d} "
                f"{a_hz:6.1f}Hz  span    ={a_span:6.1f}ms  "
                f"exec={a.get('exec_horizon','?')} ({a.get('exec_horizon',0)*a_stride/src_hz*1000:.0f}ms)")

    out = ["샘플링 격자 (원본 %gHz 기준)" % src_hz] + rows
    uniq = set(spans.values())
    if len(uniq) > 1:
        out.append(f"  ⚠️ lookback 이 모달리티마다 다르다: {spans}")
        out.append("     주파수 비율만 바꾸려 했다면 lookback 은 같아야 한다 "
                   "(span = (horizon-1) x stride).")
    if len(spans) > 1:
        base = min(int(cfg["obs_spec"][n].get("stride") or default_stride) for n in spans)
        ratios = {n: (int(cfg["obs_spec"][n].get("stride") or default_stride)) for n in spans}
        mx = max(ratios.values())
        out.append("  비율(느린 쪽 기준): " + ", ".join(
            f"{n} 1:{mx / v:g}" for n, v in sorted(ratios.items())))
    return "\n".join(out)


def dump_config(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return yaml.safe_dump(c, allow_unicode=True, sort_keys=False, default_flow_style=False)
