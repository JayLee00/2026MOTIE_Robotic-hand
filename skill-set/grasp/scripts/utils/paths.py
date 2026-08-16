#!/usr/bin/env python3
"""
머신별 경로/환경 상수 — configs/paths.yaml 을 읽어 제공.

내보내는 상수:
  CONDA_BASE        (Path)  conda 설치 루트
  CONDA_ENV         (str)   사용할 conda 환경 이름
  DOCKER_CONTAINER  (str)   ROS2 Docker 컨테이너 이름
  ROS_DOMAIN_ID     (int)   ROS_DOMAIN_ID
  KISTAR_WS         (str)   kistar_ws 절대경로 (호스트)
  MOUNT_MAP         (list)  [(host_prefix, container_prefix), ...]
"""

from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).resolve().parents[2] / "configs" / "paths.yaml"


def _load() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load()

CONDA_BASE:       Path = Path(_cfg["__CONDA_BASE__"])
CONDA_ENV:        str  = str(_cfg["__CONDA_ENV__"])
DOCKER_CONTAINER: str  = str(_cfg["__DOCKER_CONTAINER__"])
ROS_DOMAIN_ID:    int  = int(_cfg["__ROS_DOMAIN_ID__"])
KISTAR_WS:        str  = str(_cfg["__KISTAR_WS__"])
MOUNT_MAP:        list = [tuple(pair) for pair in _cfg["__MOUNT_MAP__"]]
