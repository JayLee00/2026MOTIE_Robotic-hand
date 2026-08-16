"""KISTAR 손 포즈 파일(rad) ↔ 엔코더 count 변환 + 로더.

kistar_hand_gui 가 저장하는 포즈 파일(예: launch/kiwi.txt)은 관절값을 **rad** 로 담는다
(GUI 의 Position 모드 SHM 은 float rad 를 직접 쓰기 때문). 반면 motion_sequence_A_*.py 는
core.shm_common 의 SHM(j_tar = int16 **count**)으로 손을 제어한다. 그래서 GUI 포즈 파일을
그대로 쓰려면 rad → count 변환이 필요하다.

변환 규약은 kistar_hand_gui/hand_config.py 의 단일 소스(``POSITION_TICK`` = 8192 ↔ ±π rad)를
그대로 사용한다:  count = round(rad · 8192/π),  rad = count · π/8192.

포즈 파일 형식(kiwi.txt) 예::

    # KISTAR hand pose
    name: kiwi
    mode: Position (0x0001)
    unit: rad
    # control input that reproduces this pose under the above mode:
    tar:    +1.575000 -1.571000 ... (16개 rad)
    # measured proprioception at save time:
    q_meas: +1.552389 -1.430821 ... (16개 rad)
    cur_meas: ...

``tar`` (제어 입력)을 기본으로 읽는다. 측정값을 쓰고 싶으면 field="q_meas".
"""

import math
import os
import sys
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
if project_root not in sys.path:
    sys.path.append(project_root)

# 변환 규약(POSITION_TICK)은 GUI 와 동일 소스를 쓴다. import 실패 시에도 안전하게 동작.
try:
    from kistar_hand_gui.hand_config import POSITION_TICK
except Exception:  # pragma: no cover - GUI 패키지 미존재 환경 fallback
    POSITION_TICK = 8192  # ±8192 ↔ ±π rad (Kistar_Spec.txt)

HAND_DOF = 16


# ── 스칼라/벡터 변환 ──────────────────────────────────────────────────────────
def rad_to_count(rad: float) -> int:
    """관절 rad → 엔코더 count (round, int16 의도)."""
    return int(round(float(rad) * POSITION_TICK / math.pi))


def count_to_rad(count: float) -> float:
    """엔코더 count → 관절 rad."""
    return float(count) * math.pi / POSITION_TICK


def rads_to_counts(rads) -> List[int]:
    """rad 리스트(16) → count 리스트(16)."""
    return [rad_to_count(r) for r in rads]


def counts_to_rads(counts) -> List[float]:
    """count 리스트(16) → rad 리스트(16)."""
    return [count_to_rad(c) for c in counts]


# ── 포즈 파일 파싱 ────────────────────────────────────────────────────────────
def parse_pose_rads(path: str, field: str = "tar") -> List[float]:
    """kiwi.txt 형식 파일에서 ``field`` 줄의 16개 rad 값을 읽어 반환.

    Args:
        path:  포즈 파일 경로.
        field: 읽을 줄 키워드. "tar"(제어 입력, 기본) | "q_meas"(측정값).

    Raises:
        FileNotFoundError: 파일 없음.
        ValueError: field 줄이 없거나 값이 16개가 아니거나 unit 이 rad 가 아님.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"포즈 파일을 찾을 수 없음: {path}")

    unit = None
    values: List[float] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(":")
            key = key.strip()
            if key == "unit":
                unit = rest.strip().lower()
            elif key == field:
                values = [float(tok) for tok in rest.split()]

    if not values:
        raise ValueError(f"'{path}' 에 '{field}:' 줄이 없습니다.")
    if len(values) != HAND_DOF:
        raise ValueError(
            f"'{path}' 의 '{field}' 값이 {len(values)}개입니다 — {HAND_DOF}개여야 합니다."
        )
    if unit is not None and unit != "rad":
        raise ValueError(f"'{path}' 의 unit 이 'rad' 가 아님: {unit!r} — 변환 규약 불일치.")
    return values


def load_pose_counts(path: str, field: str = "tar") -> List[int]:
    """kiwi.txt 형식 포즈 파일 → 16개 엔코더 count (motion_sequence j_tar 용)."""
    return rads_to_counts(parse_pose_rads(path, field))


if __name__ == "__main__":
    # 간단한 CLI: 포즈 파일을 count 로 출력. (예: python hand_pose_io.py kiwi.txt)
    import argparse

    ap = argparse.ArgumentParser(description="kiwi.txt(rad) → 엔코더 count 변환 확인")
    ap.add_argument("path", help="포즈 파일 경로 (kiwi.txt 형식)")
    ap.add_argument("--field", default="tar", help="읽을 줄 (tar|q_meas, 기본 tar)")
    a = ap.parse_args()
    rads = parse_pose_rads(a.path, a.field)
    counts = rads_to_counts(rads)
    print(f"POSITION_TICK = {POSITION_TICK} (±{POSITION_TICK} ↔ ±π rad)")
    print(f"rad   ({a.field}): " + " ".join(f"{r:+.4f}" for r in rads))
    print("count       : [" + ", ".join(str(c) for c in counts) + "]")
