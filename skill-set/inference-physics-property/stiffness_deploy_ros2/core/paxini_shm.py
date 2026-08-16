#!/usr/bin/env python3
"""PaXini tactile distribution published into a dedicated System V SHM segment.

WHY a separate segment (not the C++ 0x3931 SHMmsgs):
  The Franka/KISTAR C++ owns the SHMmsgs struct and we do not modify/recompile
  C++. So PaXini gets its OWN segment, written by the PaXini side-channel
  process and read by the logger/viewer. This replaces the JSON file hop with a
  true shared-memory hop (lower latency, no file mtime polling), while the JSON
  side-channel stays available as a fallback.

WHAT is stored: the full tactile distribution ``(4, 127, 3)`` =
  (finger, tactile point, xyz), unit N. The resultant per-finger FT (4, 3) is
  NOT stored separately — consumers that want it reduce the distribution
  (sum over the 127 points).

Concurrency: a single writer / many readers seqlock. ``write_seq`` (a uint64 at
offset 0) is made ODD while a write is in progress and EVEN when committed. A
reader samples write_seq, reads the payload, samples write_seq again, and retries
if it changed or was odd -> it never returns a torn (half-updated) frame.

Layout (little-endian):
    offset 0  : uint64 write_seq         (seqlock; odd = writing)
    offset 8  : int64  seq               (PaXini sample sequence)
              : int64  t_mono_ns
              : int64  t_wall_ns
              : int32  valid              (1 valid / 0 invalid)
              : int32  error_code         (-1 unknown)
              : float32 tactile[1524]     (4 x 127 x 3, unit N, row-major)
              
최종 출력 변수
- tactile(4,127,3)
- t_mono_ns: monotonic timestamp (ns), 샘플 시간
- valid: tactile 정보가 유효한지 여부 (1: 유효, 0: 무효)
- seq: tactile 샘플링 시퀀스 번호 

"""
from __future__ import annotations

import struct
import time
from typing import Optional

import numpy as np

PAXINI_SHM_KEY = 0x3934           # distinct from C++ 0x3931 / 0x3932
_SEQLOCK_FMT = "<Q"
_SEQLOCK_SIZE = struct.calcsize(_SEQLOCK_FMT)          # 8
_HEADER_FMT = "<qqqii"                                  # seq,t_mono,t_wall,valid,err
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)            # 32
_TACTILE_SHAPE = (4, 127, 3)
_TACTILE_COUNT = int(np.prod(_TACTILE_SHAPE))         # 1524
_TACTILE_BYTES = _TACTILE_COUNT * 4                    # 6096 (float32)
_PAYLOAD_SIZE = _HEADER_SIZE + _TACTILE_BYTES         # 6128
_SEG_SIZE = _SEQLOCK_SIZE + _PAYLOAD_SIZE             # 6136
DEFAULT_MAX_AGE_SEC = 1.0


def _get_sysv():
    import sysv_ipc
    return sysv_ipc


class PaxiniShmWriter:
    """Single-writer publisher for the PaXini tactile shared-memory segment."""

    def __init__(self, key: int = PAXINI_SHM_KEY):
        self.key = key
        self._shm = None
        self._wseq = 0
        self.created = False

    def attach_create(self) -> None:
        sysv = _get_sysv()
        try:
            self._shm = sysv.SharedMemory(self.key, flags=sysv.IPC_CREAT, size=_SEG_SIZE)
            self.created = True
        except sysv.ExistentialError:
            self._shm = sysv.SharedMemory(self.key)
            self.created = False
        # initialise seqlock to a clean even value
        self._wseq = 0
        self._shm.write(struct.pack(_SEQLOCK_FMT, self._wseq), 0)

    def publish(self, *, seq: int, t_mono_ns: int, t_wall_ns: int,
                valid: bool, error_code: Optional[int], tactile) -> None:
        """Publish one tactile frame atomically (seqlock begin/commit).

        ``tactile`` is any array-like reducible to ``(4, 127, 3)`` (finger, point,
        xyz) in N; missing entries are stored as NaN.
        """
        if self._shm is None:
            raise RuntimeError("attach_create() first")
        tac = np.full(_TACTILE_SHAPE, np.nan, dtype="<f4")
        if tactile is not None:
            try:
                # fast path: clean rectangular numeric array
                a = np.asarray(tactile, dtype="<f4")
                if a.ndim == 3:
                    f, p, c = (min(x, y) for x, y in zip(a.shape, _TACTILE_SHAPE))
                    tac[:f, :p, :c] = a[:f, :p, :c]
            except (ValueError, TypeError):
                # ragged / contains None (예: 일부 센서 블록 미수신) → 요소별 복사,
                # None 은 NaN 으로 둔다. 이렇게 해야 publish 가 죽지 않고 디코딩된
                # 센서만이라도 SHM 에 실린다.
                F, P, C = _TACTILE_SHAPE
                for si, sensor in enumerate(tactile[:F]):
                    if not sensor:
                        continue
                    for pi, pt in enumerate(sensor[:P]):
                        if not pt:
                            continue
                        for ci, v in enumerate(pt[:C]):
                            if v is not None:
                                tac[si, pi, ci] = v
        header = struct.pack(
            _HEADER_FMT,
            int(seq), int(t_mono_ns), int(t_wall_ns),
            1 if valid else 0,
            -1 if error_code is None else int(error_code),
        )
        payload = header + tac.tobytes()                 # contiguous payload
        self._wseq += 1                                  # -> odd: write in progress
        self._shm.write(struct.pack(_SEQLOCK_FMT, self._wseq), 0)
        self._shm.write(payload, _SEQLOCK_SIZE)
        self._wseq += 1                                  # -> even: committed
        self._shm.write(struct.pack(_SEQLOCK_FMT, self._wseq), 0)

    def publish_invalid(self, error_code: Optional[int] = None) -> None:
        self.publish(seq=-1, t_mono_ns=time.monotonic_ns(), t_wall_ns=time.time_ns(),
                     valid=False, error_code=error_code, tactile=None)

    def close(self, remove: bool = True) -> None:
        if self._shm is None:
            return
        try:
            if remove and self.created:
                self._shm.remove()
        except Exception:
            pass
        self._shm = None


class PaxiniShmReader:
    """Many-reader consumer. Returns
    (tactile(4,127,3) float32, t_mono_ns, valid, seq).
    """

    def __init__(self, key: int = PAXINI_SHM_KEY, max_age_sec: float = DEFAULT_MAX_AGE_SEC):
        self.key = key
        self.max_age_ns = int(max_age_sec * 1e9)
        self._shm = None

    def attach(self) -> bool:
        sysv = _get_sysv()
        try:
            self._shm = sysv.SharedMemory(self.key)
            return True
        except sysv.ExistentialError:
            self._shm = None
            return False

    @staticmethod
    def _invalid():
        return (np.full(_TACTILE_SHAPE, np.nan, dtype=np.float32),
                np.array(0, np.int64), np.array(0, np.int8), np.array(-1, np.int64))

    def read(self, retries: int = 8):
        if self._shm is None:
            if not self.attach():
                return self._invalid()
        for _ in range(retries):
            s0 = struct.unpack(_SEQLOCK_FMT, self._shm.read(_SEQLOCK_SIZE, 0))[0]
            if s0 & 1:                                   # writer mid-update
                continue
            payload = self._shm.read(_PAYLOAD_SIZE, _SEQLOCK_SIZE)
            s1 = struct.unpack(_SEQLOCK_FMT, self._shm.read(_SEQLOCK_SIZE, 0))[0]
            if s0 != s1 or (s1 & 1):                      # torn read -> retry
                continue
            seq, t_mono_ns, _t_wall, valid, _err = struct.unpack(
                _HEADER_FMT, payload[:_HEADER_SIZE])
            tactile = np.frombuffer(
                payload[_HEADER_SIZE:], dtype="<f4").astype(np.float32).reshape(_TACTILE_SHAPE)
            is_valid = bool(valid) and not np.all(np.isnan(tactile))
            if is_valid and self.max_age_ns > 0:
                if time.monotonic_ns() - int(t_mono_ns) > self.max_age_ns:
                    is_valid = False                     # stale (writer died?)
            if not is_valid:
                return (tactile, np.array(0, np.int64), np.array(0, np.int8), np.array(-1, np.int64))
            return (tactile, np.array(int(t_mono_ns), np.int64),
                    np.array(1, np.int8), np.array(int(seq), np.int64))
        return self._invalid()


def read_paxini_shm(reader: PaxiniShmReader):
    """Convenience drop-in returning (tactile(4,127,3), t_mono_ns, valid, seq)."""
    return reader.read()
