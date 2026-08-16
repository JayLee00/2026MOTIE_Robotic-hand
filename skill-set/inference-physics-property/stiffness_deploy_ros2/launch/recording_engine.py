#!/usr/bin/env python3
"""recording_engine.py — 배포와 '동일한 프레임'을 데모별 HDF5 로 기록하는 부품.

목적(하이브리드 수집의 Option 1):
  배포(deploy_ros2 / deploy_task3_ros2)가 스퀴즈 구간에서 모델에 넣는 바로 그 프레임을,
  그 순간·그 주기·그 정책(causal ZOH)으로 파일에도 흘린다. 저장 시퀀스 == 모델 입력 시퀀스.

parity 보장(핵심):
  - 프레임 소스가 배포와 '같은 함수' real_deploy_inference_final.read_live_sample 이다.
  - valid!=1 프레임은 배포 엔진(add_sample)과 '똑같이' 스킵한다.
  - 매 tick 브리지를 '한 번만' read → 버퍼링과 기록이 동일 프레임을 본다(이중 read 드리프트 없음).
  ⇒ 저장되는 (joint, ft, resultant, tactile) 은 배포 엔진이 self.buf 에 쌓는 값과 동일.
     (build_sensor 의 Δjoint·JOINT_SCALE·마스킹·FACTOR 다운샘플은 '추론 시' 변환이므로 여기 저장 X.
      학습/변환기가 동일 변환을 적용 → 전처리를 데이터에 baked 하지 않는다.)

RecordingEngine 은 '추론 없는' 경량 엔진이다(모델 로딩 불필요):
  deploy 의 move_hand_to_squeeze 가 engine 에 요구하는 건 reset() 와 add_sample(shm, paxini) 뿐.
  → 아직 학습 모델이 없는 과일도 수집할 수 있다.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import h5py

# 배포와 동일한 프레임 읽기/상수. (launch/ 가 sys.path 에 있어야 import 됨 — 엔트리포인트가 보장)
import real_deploy_inference_final as RE


class HDF5DemoWriter:
    """세션 1파일 + 구간별 그룹으로 저장 (이름 규칙 = bag_to_hdf5 와 통일: '{segment}__run{NNN}').

    각 그룹 = 스퀴즈 1회 = 학습 시퀀스 1개 (데모 분할이 파일 구조로 내장).
    데모 종료 시점에 그룹을 flush 하므로, 중간 크래시에도 '완료된 데모'는 보존된다
    (진행 중 데모의 메모리 버퍼만 유실 — 동시 기록한 rosbag 이 raw 보험).
    """

    def __init__(self, path, session_attrs: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "w")
        self.f.attrs["schema_version"] = "collect_v1"
        for k, v in (session_attrs or {}).items():
            self.f.attrs[k] = "None" if v is None else v
        self._g = None
        self._buf = None
        self._n = 0
        self.n_demos = 0

    # ── 데모 경계 (엔트리포인트가 스퀴즈 앞뒤로 호출) ──────────────────────
    def start_demo(self, demo_id: int, attrs: dict | None = None, name: str | None = None):
        # name 을 주면 그 이름으로(예: 'squeeze_A__run000' — bag_to_hdf5 와 규칙 통일),
        # 없으면 기존 'demo_{id}'. demo_id 는 전역 그룹 인덱스로 attr 에 남긴다.
        name = name or f"demo_{demo_id:03d}"
        if name in self.f:          # 재실행/중복 방지
            del self.f[name]
        self._g = self.f.create_group(name)
        self._g.attrs["demo_id"] = int(demo_id)
        for k, v in (attrs or {}).items():
            self._g.attrs[k] = "None" if v is None else v
        self._buf = {"joint": [], "ft": [], "resultant": [], "tactile": [],
                     "valid": [], "squeeze_on": [], "t_mono_ns": []}
        self._n = 0

    @property
    def demo_open(self) -> bool:
        return self._g is not None

    @property
    def n_current(self) -> int:
        return self._n

    # ── 프레임 적재 (RecordingEngine.add_sample 이 호출) ────────────────────
    def append(self, *, joint, ft, resultant, tactile, t_mono_ns,
               squeeze_on: int = 1, valid: int = 1):
        if self._g is None:         # 데모 미개시면 조용히 무시(안전)
            return
        b = self._buf
        b["joint"].append(np.asarray(joint, np.float32).reshape(-1))       # (16,)
        b["ft"].append(np.asarray(ft, np.float32).reshape(-1))             # (12,)
        b["resultant"].append(np.asarray(resultant, np.float32).reshape(-1))  # (12,)
        b["tactile"].append(np.asarray(tactile, np.float32))               # (4,127,3)
        b["valid"].append(np.int8(valid))
        b["squeeze_on"].append(np.int8(squeeze_on))
        b["t_mono_ns"].append(np.int64(t_mono_ns))
        self._n += 1

    def end_demo(self, attrs: dict | None = None) -> int:
        if self._g is None:
            return 0
        b, n = self._buf, self._n
        if n > 0:
            gz = dict(compression="gzip")
            self._g.create_dataset("joint", data=np.stack(b["joint"]), **gz)            # (n,16)
            self._g.create_dataset("ft", data=np.stack(b["ft"]), **gz)                  # (n,12)
            self._g.create_dataset("resultant",
                                   data=np.stack(b["resultant"]).reshape(n, 4, 3), **gz)  # (n,4,3)
            self._g.create_dataset("tactile", data=np.stack(b["tactile"]), **gz)        # (n,4,127,3)
            self._g.create_dataset("valid", data=np.asarray(b["valid"], np.int8))
            self._g.create_dataset("squeeze_on", data=np.asarray(b["squeeze_on"], np.int8))
            self._g.create_dataset("t_mono_ns", data=np.asarray(b["t_mono_ns"], np.int64))
        self._g.attrs["n_samples"] = int(n)
        for k, v in (attrs or {}).items():
            self._g.attrs[k] = "None" if v is None else v
        if n > 0:
            self.n_demos += 1
        self.f.flush()              # 완료 데모 즉시 디스크로
        self._g = None
        self._buf = None
        self._n = 0
        return n

    def set_group_attr(self, name: str, key: str, value):
        """이미 닫힌 그룹에 attr 를 사후 기록(예: 데모 판정 outcome). 파일은 세션 내내 열려 있음."""
        if name in self.f:
            self.f[name].attrs[key] = "None" if value is None else value

    def close(self):
        self.f.attrs["n_demos"] = int(self.n_demos)
        self.f.close()


class RecordingEngine:
    """deploy 의 move_hand_to_squeeze 가 쓰는 engine 인터페이스(reset/add_sample) 를
    구현하되, 모델 대신 '기록'을 한다. (추론 없음 — 모델 불필요)

    writer          : HDF5DemoWriter (열려 있는 현재 데모 그룹에 append)
    on_squeeze_start: 스퀴즈의 '첫 유효 프레임' 시점에 1회 호출되는 콜백
                      (엔트리포인트가 /collect/squeeze_on=1 발행에 사용 — rosbag 데모 분할용)
    """

    def __init__(self, writer: HDF5DemoWriter | None = None, on_squeeze_start=None):
        self._writer = writer
        self._on_squeeze_start = on_squeeze_start
        self._sq_started = False

    def reset(self):
        """스퀴즈 시작(move_hand_to_squeeze 초입)마다 호출 — 배포 엔진과 동일 타이밍."""
        self._sq_started = False

    def add_sample(self, shm, paxini) -> bool:
        """스퀴즈+hold 구간 매 제어주기 호출. 배포 엔진 add_sample 과 '동일 규칙'으로 기록."""
        s = RE.read_live_sample(shm, paxini)     # 배포와 같은 프레임 (단일 read)
        if s["valid"] != 1:
            return False                          # 무효 프레임 스킵 (배포와 동일)
        if not self._sq_started:                  # 스퀴즈 첫 유효 프레임 = rising edge
            self._sq_started = True
            if self._on_squeeze_start is not None:
                self._on_squeeze_start()
        if self._writer is not None:
            self._writer.append(
                joint=s["joint"], ft=s["ft"], resultant=s["resultant"],
                tactile=s["tactile"], t_mono_ns=time.monotonic_ns(),
                squeeze_on=1, valid=1)
        return True
