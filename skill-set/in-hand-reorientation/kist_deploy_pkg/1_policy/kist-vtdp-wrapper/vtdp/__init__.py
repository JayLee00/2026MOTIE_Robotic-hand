"""vtdp — KIST visuo-tactile diffusion policy wrapper.

config 로 모듈을 갈아끼우는 정책 프레임워크. 텐서 계약은 docs/SHAPES.md 참조.
"""
from vtdp import encoders, fusion, heads      # noqa: F401  (레지스트리 등록)
from vtdp.policy import VTDPolicy, build_policy
from vtdp.registry import available, build, describe, register

__all__ = ["VTDPolicy", "build_policy", "register", "build", "available", "describe"]
