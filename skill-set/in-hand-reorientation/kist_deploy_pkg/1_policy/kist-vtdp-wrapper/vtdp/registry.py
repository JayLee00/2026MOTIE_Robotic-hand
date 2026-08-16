#!/usr/bin/env python3
"""이름 → 생성자 레지스트리.

Hydra `_target_` 대신 이걸 쓴다. 이유는 하나 — **오류 메시지**다.
`_target_` 오타는 import 실패로 뜨지만, 여기서는 "그런 이름 없다, 있는 건 이것들이다"가 뜬다.
연구실에서 시간에 쫓기며 디버깅할 때 이 차이가 크다.

사용:
    @register("tactile", "conv1d")
    class TactileConv1d(nn.Module): ...

    enc = build("tactile", "conv1d", in_dim=12, horizon=8, d_model=256)
"""
from __future__ import annotations

from typing import Any, Callable

# kind -> {name -> ctor}
_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}

KINDS = ("state", "tactile", "vision", "fusion", "denoiser", "head")


def register(kind: str, name: str):
    """데코레이터. 같은 (kind, name) 을 두 번 등록하면 즉시 실패한다(조용한 덮어쓰기 방지)."""
    if kind not in KINDS:
        raise ValueError(f"모르는 kind={kind!r}. 가능: {KINDS}")

    def deco(ctor):
        table = _REGISTRY.setdefault(kind, {})
        if name in table:
            raise KeyError(f"{kind}/{name} 이 이미 등록돼 있다 ({table[name]})")
        table[name] = ctor
        return ctor

    return deco


def build(kind: str, name: str, **kwargs):
    table = _REGISTRY.get(kind, {})
    if name not in table:
        avail = ", ".join(sorted(table)) or "(없음)"
        raise KeyError(f"{kind}/{name} 없음. 등록된 {kind}: {avail}")
    try:
        return table[name](**kwargs)
    except TypeError as e:
        # 생성자 인자 불일치는 config 오타에서 가장 흔하다. 뭘 받는지 같이 보여준다.
        import inspect
        sig = inspect.signature(table[name])
        raise TypeError(f"{kind}/{name} 생성 실패: {e}\n  받는 인자: {sig}") from e


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def describe() -> str:
    lines = []
    for kind in KINDS:
        names = available(kind)
        lines.append(f"  {kind:9s}: {', '.join(names) if names else '(없음)'}")
    return "\n".join(lines)
