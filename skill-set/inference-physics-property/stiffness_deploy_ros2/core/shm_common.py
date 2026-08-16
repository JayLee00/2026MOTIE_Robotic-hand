"""
공유 메모리(SHM) 접근 모듈.
C++ shm.h의 SHMmsgs 구조체와 동일한 레이아웃으로 읽기/쓰기.
(shm.cpp / shm.h 코드는 수정하지 않음)
"""
from ctypes import (
    Structure, Union,
    c_uint8, c_int16, c_int32, c_float, c_double,
    sizeof, addressof, memmove,
    POINTER, cast,
)
from typing import Optional, Tuple
import os

# shm.h 상수 (동일)
Hand_Num = 1
Hand_DOF = 16
Kinesthetic_Sensor_Num = 4
Kinesthetic_Sensor_DOF = 3
Tactile_Sensor_Num = 60
Finger_Num = 4
Cartesian_DOF = 3
Quaternion_DOF = 4
Arm_Num = 1
Arm_DOF = 7
Glove_Num = 1
Glove_DOF = 1
Glove_Tactile_Sensor_Num = 16

# C++ shm_msg_key = 0x3931
SHM_MSG_KEY = 0x3931


class SHMmsgs(Structure):
    """shm.h SHMmsgs와 동일한 필드 순서·타입 (패딩은 ctypes 기본 정렬)"""
    _fields_ = [
        ("hand_mode", c_uint8 * Hand_Num),
        ("servo_on", c_uint8 * Hand_Num),
        ("j_pos", (c_int16 * Hand_DOF) * Hand_Num),
        ("j_tar", (c_int16 * Hand_DOF) * Hand_Num),
        ("j_cur", (c_int16 * Hand_DOF) * Hand_Num),
        ("j_kin", ((c_int16 * Kinesthetic_Sensor_DOF) * Kinesthetic_Sensor_Num) * Hand_Num),
        ("j_tac", (c_int16 * Tactile_Sensor_Num) * Hand_Num),
        # C: [Hand_Num][Finger_Num][Cartesian_DOF] → 안쪽이 Cartesian 먼저여야 [h][f][c] 인덱싱 가능
        ("tip_pos", ((c_float * Cartesian_DOF) * Finger_Num) * Hand_Num),
        ("tip_quat", ((c_float * Quaternion_DOF) * Finger_Num) * Hand_Num),
        ("Arm_j_pos", (c_double * Arm_DOF) * Arm_Num),
        ("Arm_j_tar", (c_double * Arm_DOF) * Arm_Num),
        ("Arm_j_vel", (c_double * Arm_DOF) * Arm_Num),
        ("Arm_C_Pos", (c_double * 16) * Arm_Num),
        ("Arm_j_tq", (c_double * Arm_DOF) * Arm_Num),
        ("Arm_Speed_Factor", (c_double * Arm_Num)),
        ("g_pos", (c_int16 * Glove_DOF) * Glove_Num),
        ("g_tac", (c_int16 * Glove_Tactile_Sensor_Num) * Glove_Num),
        ("process_num", c_int32),
    ]


SHMMSGS_SIZE = sizeof(SHMmsgs)


def _get_shm_module():
    """sysv_ipc 사용 가능 시 반환, 없으면 None."""
    try:
        import sysv_ipc
        return sysv_ipc
    except ImportError:
        return None


class ShmAccess:
    """SHM 세그먼트 attach 후 읽기/쓰기."""
    def __init__(self, key: int = SHM_MSG_KEY):
        self._key = key
        self._shm = None
        self._buf = None
        self._view = None  # SHMmsgs 뷰
        self._actual_size = SHMMSGS_SIZE

    def _read_compat(self) -> bytes:
        """구버전 SHM(예: 760바이트)도 0 패딩으로 호환 읽기."""
        raw = bytes(self._shm.read(SHMMSGS_SIZE))
        self._actual_size = len(raw)
        if self._actual_size < SHMMSGS_SIZE:
            raw += b"\x00" * (SHMMSGS_SIZE - self._actual_size)
        return raw

    def attach(self) -> bool:
        """기존 SHM에 attach. C++ 프로세스가 먼저 생성한 경우 사용."""
        mod = _get_shm_module()
        if mod is None:
            raise RuntimeError("pip install sysv_ipc 필요")
        try:
            self._shm = mod.SharedMemory(self._key)
        except mod.ExistentialError:
            return False
        self._buf = (c_uint8 * SHMMSGS_SIZE).from_buffer_copy(self._read_compat())
        # 매번 최신 메모리에서 SHMmsgs 뷰 생성 (read 시 복사 후 뷰)
        return True

    def attach_create(self, size: Optional[int] = None) -> bool:
        """SHM이 없으면 생성 후 attach (테스트용)."""
        mod = _get_shm_module()
        if mod is None:
            raise RuntimeError("pip install sysv_ipc 필요")
        size = size or SHMMSGS_SIZE
        try:
            self._shm = mod.SharedMemory(self._key, flags=mod.IPC_CREAT, size=size)
        except mod.ExistentialError:
            self._shm = mod.SharedMemory(self._key)
        self._buf = (c_uint8 * SHMMSGS_SIZE).from_buffer_copy(self._read_compat())
        return True

    def read(self) -> SHMmsgs:
        """현재 SHM 내용을 읽어 SHMmsgs 인스턴스로 반환."""
        if self._shm is None:
            raise RuntimeError("attach() 먼저 호출 필요")
        raw = self._read_compat()
        msg = SHMmsgs.from_buffer_copy(raw)
        return msg

    def read_into(self, msg: SHMmsgs) -> None:
        """SHM 내용을 기존 SHMmsgs 인스턴스에 복사."""
        if self._shm is None:
            raise RuntimeError("attach() 먼저 호출 필요")
        raw = self._read_compat()
        memmove(addressof(msg), raw, SHMMSGS_SIZE)

    def _write_field(self, field_name: str, data: bytes) -> None:
        """Write one SHMmsgs field without overwriting concurrently updated sensor data."""
        offset = getattr(SHMmsgs, field_name).offset
        if offset >= self._actual_size:
            return
        if offset + len(data) > self._actual_size:
            data = data[:self._actual_size - offset]
        self._shm.write(data, offset)

    def write_partial(
        self,
        *,
        hand_mode: Optional[Tuple[int, ...]] = None,
        servo_on: Optional[Tuple[int, ...]] = None,
        j_tar: Optional[Tuple[Tuple[int, ...], ...]] = None,
        Arm_j_tar: Optional[Tuple[Tuple[float, ...], ...]] = None,
        Arm_Speed_Factor: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """일부 명령 필드만 덮어쓰기.

        이전 구현은 SHM 전체를 read-modify-write 해서, C++ 프로세스가 갱신한
        j_kin 같은 센서 필드를 오래된 Python 스냅샷으로 되돌릴 수 있었다.
        여기서는 실제로 제어에 필요한 필드의 byte range만 직접 기록한다.
        """
        if self._shm is None:
            raise RuntimeError("attach() 먼저 호출 필요")
        if hand_mode is not None:
            value = (c_uint8 * Hand_Num)()
            for i, v in enumerate(hand_mode):
                if i < Hand_Num:
                    value[i] = int(v) & 0xFF
            self._write_field("hand_mode", bytes(value))
        if servo_on is not None:
            value = (c_uint8 * Hand_Num)()
            for i, v in enumerate(servo_on):
                if i < Hand_Num:
                    value[i] = int(v) & 0xFF
            self._write_field("servo_on", bytes(value))
        if j_tar is not None:
            value = ((c_int16 * Hand_DOF) * Hand_Num)()
            for h in range(Hand_Num):
                if h < len(j_tar):
                    for j in range(Hand_DOF):
                        if j < len(j_tar[h]):
                            value[h][j] = int(j_tar[h][j])
            self._write_field("j_tar", bytes(value))
        if Arm_j_tar is not None:
            value = ((c_double * Arm_DOF) * Arm_Num)()
            for a in range(Arm_Num):
                if a < len(Arm_j_tar):
                    for j in range(Arm_DOF):
                        if j < len(Arm_j_tar[a]):
                            value[a][j] = float(Arm_j_tar[a][j])
            self._write_field("Arm_j_tar", bytes(value))
        if Arm_Speed_Factor is not None:
            value = (c_double * Arm_Num)()
            for a in range(Arm_Num):
                if a < len(Arm_Speed_Factor):
                    value[a] = float(Arm_Speed_Factor[a])
            self._write_field("Arm_Speed_Factor", bytes(value))

    def detach(self) -> None:
        """attach 해제 (프로세스 종료 시 자동이지만 명시적으로 가능)."""
        self._shm = None
        self._buf = None
        self._view = None


def shmmsgs_to_arrays(msg: SHMmsgs):
    """SHMmsgs ctypes 인스턴스를 numpy-friendly 리스트/배열로 변환 (HDF5 저장용)."""
    import numpy as np
    return {
        "01_hand_mode": np.array([msg.hand_mode[i] for i in range(Hand_Num)], dtype=np.uint8),
        "02_hand_servo_on": np.array([msg.servo_on[i] for i in range(Hand_Num)], dtype=np.uint8),
        "03_hand_j_pos": np.array([[msg.j_pos[h][j] for j in range(Hand_DOF)] for h in range(Hand_Num)], dtype=np.int16),
        "04_hand_j_tar": np.array([[msg.j_tar[h][j] for j in range(Hand_DOF)] for h in range(Hand_Num)], dtype=np.int16),
        "05_hand_j_cur": np.array([[msg.j_cur[h][j] for j in range(Hand_DOF)] for h in range(Hand_Num)], dtype=np.int16),
        "06_hand_j_kin": np.array([[[msg.j_kin[h][i][k] for k in range(Kinesthetic_Sensor_DOF)] for i in range(Kinesthetic_Sensor_Num)] for h in range(Hand_Num)], dtype=np.int16),
        "07_hand_j_tac": np.array([[msg.j_tac[h][i] for i in range(Tactile_Sensor_Num)] for h in range(Hand_Num)], dtype=np.int16),
        "08_hand_tip_pos": np.array([[msg.tip_pos[h][f][c] for f in range(Finger_Num) for c in range(Cartesian_DOF)] for h in range(Hand_Num)], dtype=np.float32),
        "09_hand_tip_quat": np.array([[msg.tip_quat[h][f][q] for f in range(Finger_Num) for q in range(Quaternion_DOF)] for h in range(Hand_Num)], dtype=np.float32),
        "10_franka_Arm_j_pos": np.array([[msg.Arm_j_pos[a][j] for j in range(Arm_DOF)] for a in range(Arm_Num)], dtype=np.float64),
        "11_franka_Arm_j_tar": np.array([[msg.Arm_j_tar[a][j] for j in range(Arm_DOF)] for a in range(Arm_Num)], dtype=np.float64),
        "12_franka_Arm_j_vel": np.array([[msg.Arm_j_vel[a][j] for j in range(Arm_DOF)] for a in range(Arm_Num)], dtype=np.float64),
        "13_franka_Arm_C_pos": np.array([[msg.Arm_C_Pos[a][i] for i in range(16)] for a in range(Arm_Num)], dtype=np.float64),
        "14_franka_Arm_j_tq": np.array([[msg.Arm_j_tq[a][j] for j in range(Arm_DOF)] for a in range(Arm_Num)], dtype=np.float64),
        "15_franka_Arm_speed_factor": np.array([msg.Arm_Speed_Factor[a] for a in range(Arm_Num)], dtype=np.float64),
        "16_glove_g_pos": np.array([[msg.g_pos[h][i] for i in range(Glove_DOF)] for h in range(Glove_Num)], dtype=np.int16),
        "17_glove_g_tac": np.array([[msg.g_tac[h][i] for i in range(Glove_Tactile_Sensor_Num)] for h in range(Glove_Num)], dtype=np.int16),
    }
