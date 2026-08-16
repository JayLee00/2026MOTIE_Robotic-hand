#!/usr/bin/env python3
"""deploy_ros2.py — deploy.py 를 Dual_Arm_Hand_Ctrl 의 ROS2 토픽 기반으로 실행.

deploy.py 의 모션 시퀀스·힘 판정·강성 추론 로직을 **그대로 재사용**하고, SHM 직접
접근(core.shm_common.ShmAccess / core.paxini_shm.PaxiniShmReader)만 Dual_Arm_Hand_Ctrl
의 ROS2 토픽으로 치환한다. (기존 기능 100% 유지 — I/O 계층만 교체)

Dual_Arm_Hand_Ctrl 인터페이스 (해당 워크스페이스는 수정하지 않음):
  ※ 토픽명은 실제 시스템(ros2 topic list, 2026-07-06)의 /<side>/ 규약을 따른다. (side=right)
  명령(publish):
    Arm_j_tar   → /franka/right/q_target  (Float64MultiArray, 7)   [arm q 수신 노드]
    hand j_tar  → /hand/right/q_target    (Float32MultiArray, 16)  [hand 수신 노드]
    servo_on    → /hand/right/cmd_servo   (Bool)
    hand_mode   → /hand/right/cmd_mode    (Int32)
  상태(subscribe):
    /franka/right/joint_states  (JointState) → R팔 position[0:7]      → msg.Arm_j_pos[0]
    /hand/right/joint_states    (JointState) → R손 position[0:16]     → msg.j_pos[0]
    /hand/right/kin             (Float32MultiArray, 4x3)             → msg.j_kin[0]  (추론용)
    /paxini/right/ft            (Float32MultiArray, 4x3, 손가락별 합력) → PaXini (4,127,3) 재구성

전제:
  - Dual_Arm_Hand_Ctrl 의 C++ 컨트롤러 + shm_state_publisher_node
    + arm q 수신 노드 + hand 수신 노드 (+ paxini writer) 가
    함께 실행 중이어야 한다. (SHM 은 그 스택이 채운다)
  - 단일 팔/손(R) 기준(deploy.py 원본과 동일). L 확장은 토픽의 right→left 만 바꾸면 됨.
  - 힘 판정은 /paxini/right/ft (4,3) 을 (4,127,3) 의 point0 에 실어 재구성하므로,
    deploy.calculate_contact_normal_force 의 sum(over 127 points) 이 그대로 성립한다.
    (→ /paxini/right/ft 이 '손가락별 합력'을 담아야 유효. 센서 자체 FT블록이 0이면 힘=0)

실행:
  # (Ctrl 스택 + paxini 가 떠 있는 상태에서)
  source /opt/ros/humble/setup.bash
  source /home/prime/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash
  python3 /home/prime/YS_ws/Gen3/launch/deploy_ros2.py   # 실행 후 과일 번호 입력
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Bool, Int32

# deploy.py 의 로직/상수/엔진 재사용.
#   - launch/ : `import deploy`, `import real_deploy_inference_old`
#   - Gen3(project_root) : deploy 내부의 `from core.*` 해석
_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_LAUNCH_DIR, ".."))  # Gen3
sys.path.insert(0, _LAUNCH_DIR)                       # launch/

# ROS2 의 'launch' 패키지가 sys.path 를 선점하므로, deploy.py 의
# `from launch.real_deploy_inference_old import ...` 가 Gen3/launch 대신 ROS 의 launch 를
# 보게 되어 실패한다. Gen3/launch 의 그 모듈을 top-level 로 로드해 'launch.*' 이름으로
# 미리 등록하면, deploy 의 import 가 이 모듈을 그대로 쓴다. (deploy.py 는 수정하지 않음)
import importlib  # noqa: E402
sys.modules.setdefault(
    "launch.real_deploy_inference_old",
    importlib.import_module("real_deploy_inference_final"))

import real_deploy_inference_final as _RE  # noqa: E402  (SOTA 앙상블 설정 참조)
import deploy as D  # noqa: E402  (module-level: path/const/engine 준비, SHM attach 없음)

# 강성 추론 결과 → GUI 로 연속 발행하는 퍼블리셔 + GUI 자동 실행 (launch/ 에 위치, std_msgs 만 의존).
from stiffness_result_pub import StiffnessResultPublisher, spawn_gui  # noqa: E402

# core 상수 (deploy 와 동일 출처)
from core.shm_common import (  # noqa: E402
    Hand_DOF, Arm_DOF, Kinesthetic_Sensor_Num, Kinesthetic_Sensor_DOF,
)

_PAXINI_POINTS = 127
_KIN = Kinesthetic_Sensor_Num * Kinesthetic_Sensor_DOF  # 12


class _Msg:
    """deploy/추론엔진이 접근하는 shm.read() 결과 흉내:
    .Arm_j_pos[0][j] (7), .j_pos[0][j] (16), .j_kin[0][i][k] (4x3)."""

    def __init__(self, arm_r, hand_r, kin_r):
        self.Arm_j_pos = [arm_r]
        self.j_pos = [hand_r]
        self.j_kin = [kin_r]


class Ros2ShmBridge(Node):
    """ShmAccess 인터페이스(attach/read/write_partial/detach)를 Ctrl ROS2 토픽으로 구현."""

    def __init__(self):
        super().__init__("deploy_ros2_bridge")
        cmd_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        state_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        # 고속 setpoint 스트림(q_target): 수신 노드(hand_target_receiver)가 BEST_EFFORT 라
        # RELIABLE 로 보내면 ack 대기로 write 가 수초 블로킹(→ 끊김). BEST_EFFORT 로 맞춤.
        stream_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)

        self._pub_arm = self.create_publisher(Float64MultiArray, "/franka/right/q_target", stream_qos)
        self._pub_hand = self.create_publisher(Float32MultiArray, "/hand/right/q_target", stream_qos)
        self._pub_servo = self.create_publisher(Bool, "/hand/right/cmd_servo", cmd_qos)   # 단발=RELIABLE 유지
        self._pub_mode = self.create_publisher(Int32, "/hand/right/cmd_mode", cmd_qos)
        
        self._lock = threading.Lock()
        self._arm_pos = None                                   # list[7]
        self._hand_pos = None                                  # list[16]
        self._kin = np.zeros((Kinesthetic_Sensor_Num, Kinesthetic_Sensor_DOF), np.float32)
        
        self._warned_sf = False
        self._last_servo = None      # servo/mode 100Hz 중복 발행 방지 (RELIABLE congestion 감소)
        self._last_mode = None

        self.create_subscription(JointState, "/franka/right/joint_states", self._on_arm, state_qos)
        self.create_subscription(JointState, "/hand/right/joint_states", self._on_hand, state_qos)
        self.create_subscription(Float32MultiArray, "/hand/right/kin", self._on_kin, state_qos)

    # ── 구독 콜백 (Ctrl shm_state_publisher: R 먼저, 그다음 L) ──────────────
    def _on_arm(self, m):
        if len(m.position) >= Arm_DOF:
            with self._lock:
                self._arm_pos = [float(m.position[j]) for j in range(Arm_DOF)]

    def _on_hand(self, m):
        if len(m.position) >= Hand_DOF:
            with self._lock:
                self._hand_pos = [int(round(m.position[j])) for j in range(Hand_DOF)]

    def _on_kin(self, m):
        if len(m.data) >= _KIN:
            with self._lock:
                self._kin = np.array(m.data[:_KIN], np.float32).reshape(
                    Kinesthetic_Sensor_Num, Kinesthetic_Sensor_DOF)

    # ── ShmAccess 인터페이스 ───────────────────────────────────────────────
    def attach(self, timeout_sec: float = 5.0) -> bool:
        """arm/hand 상태 첫 수신까지 대기."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_sec:
            with self._lock:
                if self._arm_pos is not None and self._hand_pos is not None:
                    return True
            time.sleep(0.05)
        with self._lock:
            return self._arm_pos is not None and self._hand_pos is not None

    def read(self) -> _Msg:
        with self._lock:
            arm = list(self._arm_pos) if self._arm_pos is not None else [0.0] * Arm_DOF
            hand = list(self._hand_pos) if self._hand_pos is not None else [0] * Hand_DOF
            kin = self._kin.copy()
        return _Msg(arm, hand, kin)

    def write_partial(self, *, hand_mode=None, servo_on=None, j_tar=None,
                      Arm_j_tar=None, Arm_Speed_Factor=None) -> None:
        if Arm_j_tar is not None:
            msg = Float64MultiArray()
            msg.data = [float(x) for x in Arm_j_tar[0]]
            self._pub_arm.publish(msg)
        if j_tar is not None:
            msg = Float32MultiArray()
            msg.data = [float(x) for x in j_tar[0]]
            self._pub_hand.publish(msg)

        if servo_on is not None:
            v = bool(int(servo_on[0]))
            if v != self._last_servo:            # 바뀔 때만 발행 (매 tick 재발행 X)
                self._pub_servo.publish(Bool(data=v)); self._last_servo = v
        if hand_mode is not None:
            v = int(hand_mode[0])
            if v != self._last_mode:
                self._pub_mode.publish(Int32(data=v)); self._last_mode = v

        if Arm_Speed_Factor is not None and not self._warned_sf:
            # Ctrl 에 speed_factor 명령 토픽이 없음 → 무시(로그 1회).
            self.get_logger().warn(
                "Arm_Speed_Factor 명령 토픽이 Dual_Arm_Hand_Ctrl 에 없어 무시함")
            self._warned_sf = True

    def safe_hand_servo_on(self, mode: int = 1, settle_sec: float = 0.2) -> bool:
        """안전 서보-온(ROS2_TOPIC_GUIDE §2): 현재 손 자세를 q_target 으로 1회 먼저 발행 →
        (정착 대기) → servo_on. 서보를 먼저 켜서 q_tar=0 으로 손가락이 튀는 것을 방지한다.
        시작 시 attach 직후 1회 호출. 손 상태 미수신이면 서보를 켜지 않고 False 반환.
        """
        with self._lock:
            hand = None if self._hand_pos is None else list(self._hand_pos)
        if hand is None:
            self.get_logger().warn("safe_hand_servo_on: 손 상태 미수신 — 서보 on 생략")
            return False
        # 1) 모드 + 현재 자세를 q_target 으로 먼저 발행 (수신 노드가 q_tar 를 현재값으로 잡게)
        self._pub_mode.publish(Int32(data=int(mode)))
        msg = Float32MultiArray()
        msg.data = [float(x) for x in hand]
        self._pub_hand.publish(msg)
        time.sleep(settle_sec)
        # 2) 그 다음 servo_on
        self._pub_servo.publish(Bool(data=True))
        time.sleep(0.05)
        self.get_logger().info(
            f"safe servo-on: 현재 손 자세를 q_target 으로 먼저 발행 후 servo on (mode={mode})")
        return True

    def detach(self) -> None:
        pass


class Ros2PaxiniBridge:
    """PaxiniShmReader.read() 흉내: /paxini/right/ft (4,3) → (4,127,3) 재구성.
    point0 에 손가락별 합력을 실어 deploy 의 sum(over 127 points) 이 그대로 성립."""

    def __init__(self, node: Node, topic: str = "/paxini/right/ft"):
        self._node = node
        self._lock = threading.Lock()
        self._ft = None                                        # (4,3)
        self._seq = 0
        state_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        node.create_subscription(Float32MultiArray, topic, self._on_ft, state_qos)

    def _on_ft(self, m):
        if len(m.data) >= 12:
            with self._lock:
                self._ft = np.array(m.data[:12], np.float32).reshape(4, 3)
                self._seq += 1

    def attach(self, timeout_sec: float = 3.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_sec:
            with self._lock:
                if self._ft is not None:
                    return True
            time.sleep(0.05)
        with self._lock:
            return self._ft is not None

    def read(self):
        with self._lock:
            ft = None if self._ft is None else self._ft.copy()
            seq = self._seq
        if ft is None:
            return (np.zeros((4, _PAXINI_POINTS, 3), np.float32),
                    np.array(0, np.int64), np.array(0, np.int8), np.array(-1, np.int64))
        tac = np.zeros((4, _PAXINI_POINTS, 3), np.float32)
        tac[:, 0, :] = ft                                      # point0 = 합력
        return (tac, np.array(time.monotonic_ns(), np.int64),
                np.array(1, np.int8), np.array(int(seq), np.int64))


def _print_inference(engine):
    """스퀴즈 직후 추론 결과 출력. (stiffness, cls, cname) 또는 None(엔진없음/샘플부족) 반환."""
    if engine is None:
        return None
    stiffness, cls, cname = engine.infer()
    print("\n" + "=" * 48)
    if stiffness is not None:
        print(f"  [추론 결과] {engine.fruit}")
        print(f"    절대강성 = {stiffness:.3f}")
        print(f"    등급     = {cname}  (class {cls})")
    else:
        print("  [추론 결과] 샘플 부족 — 추론 불가")
    print("=" * 48 + "\n")
    return (stiffness, cls, cname) if stiffness is not None else None


def _grip(shm, paxini):
    """안전 위치 → 파지. 파지 도달 position 반환(스퀴즈 복귀 기준)."""
    shm.write_partial(servo_on=(D.HAND_SAFE_SERVO_ON,), hand_mode=(D.HAND_SAFE_MODE,))
    print("=================== 안전 위치 (손만) ===================")
    D.move_hand_to(shm, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)
    print("=================== 파지 위치 (손만) ===================")
    return D.move_hand_to_target_until_force(
        shm, paxini, D.HAND_GRIP_POINT, D.HAND_MOVE_DURATION, D.GRIP_FORCE_THRESHOLD,
        grip_curl=True)


def _squeeze_and_infer(shm, paxini, engine, grip_position, result_pub=None):
    """현재 파지 자세(grip_position)에서 스퀴즈 → 추론. 결과를 GUI(result_pub)로 발행."""
    print("=================== 스퀴즈 모션 ===================")
    if result_pub is not None and engine is not None:
        result_pub.set_measuring(engine.fruit, engine.norm_min, engine.norm_max,
                                 engine.boundaries, engine.class_names)   # GUI: 측정 중
    D.move_hand_to_squeeze(
        shm, paxini, D.HAND_SAVE_POINT, D.HAND_SQUEEZE_DURATION,
        D.SQUEEZE_FORCE_THRESHOLD, grip_position,
        return_duration=D.HAND_SQUEEZE_RETURN_DURATION, engine=engine)
    result = _print_inference(engine)
    if result_pub is not None and engine is not None:      # GUI: 결과(또는 샘플부족)
        if result is not None:
            result_pub.set_result(engine.fruit, result[0], result[1], result[2],
                                  engine.norm_min, engine.norm_max,
                                  engine.boundaries, engine.class_names)
        else:
            result_pub.set_error("샘플 부족 — 추론 불가")
    return result


def _ask_next_action() -> str:
    """추론 후 다음 동작 선택 (과일 선택과 동일한 터미널 입력)."""
    while True:
        c = input("\n다음 동작  [1] 다시 스퀴즈+추론  "
                  "[2] 안전위치 복귀 후 다시 스퀴즈  [3] 안전위치 복귀 후 종료 : ").strip()
        if c in ("1", "2", "3"):
            return c
        print("  1, 2, 3 중에서 입력하세요.")


def main() -> None:
    # --no-gui : 결과 GUI 자동 실행 끄기. D.parse_args(argparse) 가 모르는 인자라 미리 제거.
    want_gui = "--no-gui" not in sys.argv
    if not want_gui:
        sys.argv = [a for a in sys.argv if a != "--no-gui"]
    if want_gui:
        spawn_gui()

    args = D.parse_args()
    # 시작 시 모듈 기본 포즈(kiwi) 출력은 생략 — 과일 선택 후 set_pose_for_fruit 가 실제 포즈를 찍음.
    marker_path = Path(args.marker_file).resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    D._set_squeeze_flag(False)

    rclpy.init()
    bridge = Ros2ShmBridge()
    paxini = Ros2PaxiniBridge(bridge, "/paxini/right/ft")
    result_pub = StiffnessResultPublisher()   # 강성 결과 → GUI 연속 발행
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(result_pub)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if not bridge.attach():
            raise SystemExit(
                "상태 토픽 미수신 — Dual_Arm_Hand_Ctrl 의 shm_state_publisher_node "
                "(및 C++ 컨트롤러)가 실행 중인지, ROS_DOMAIN_ID 가 맞는지 확인하세요.")
        if not paxini.attach():
            print("[deploy_ros2] 경고: /paxini/right/ft 미수신 — 힘=0 으로 진행"
                  "(paxini writer 실행/토픽 확인).")

        # 안전 서보-온: 현재 손 자세를 q_target 으로 먼저 발행 → servo_on (손가락 튐 방지)
        bridge.safe_hand_servo_on(mode=D.HAND_SAFE_MODE)

        # deploy.py 와 동일: 과일 선택 → 모델/포즈/엔진 준비
        fruit = D.ask_fruit()
        model_path, pose_file, force_zero = D.resolve_fruit_config(fruit)
        D.set_pose_for_fruit(pose_file)
        D.set_thresholds_for_fruit(fruit)     # 과일별 파지/스퀴즈 임계값 로드
        # ★ SOTA 앙상블(Phase_SOTA.md §15): USE_SOTA_ENSEMBLE 면 5-seed 리스트로 교체
        #   (재파지·제어기 변경 없이 추론측만 바꿈). False 면 기존 단일모델 그대로.
        _model_arg = _RE.SOTA_ENSEMBLE_PATHS if _RE.USE_SOTA_ENSEMBLE else model_path
        engine = D.StiffnessInferenceEngine(
            model_path=_model_arg, fruit=fruit, label_dir=D.LABEL_DIR, force_zero=force_zero)
        _mdesc = (f"SOTA앙상블 {len(_RE.SOTA_ENSEMBLE_PATHS)}-seed"
                  if _RE.USE_SOTA_ENSEMBLE else Path(model_path).name)
        print(f"[deploy_ros2] 추론엔진 준비 완료. 과일={fruit}, 모델={_mdesc}")

        # 첫 실행: 안전 → 파지 → 스퀴즈 → 추론
        demo_id = 0
        print(f"\n--- 데모 {demo_id} ---")
        D._write_marker(marker_path, "S", demo_id)
        grip_position = _grip(bridge, paxini)
        _squeeze_and_infer(bridge, paxini, engine, grip_position, result_pub)
        D._write_marker(marker_path, "E", demo_id)

        # 추론 후 메뉴 반복: 1=다시 스퀴즈+추론, 2=안전복귀 후 재파지→스퀴즈, 3=안전복귀 후 종료
        while True:
            action = _ask_next_action()
            if action == "3":   # 안전위치 복귀 후 종료
                print("=================== 안전 위치 복귀 후 종료 ===================")
                D.move_hand_to(bridge, D.HAND_SAFE_POSITION, D.HAND_MOVE_DURATION)
                break

            demo_id += 1
            print(f"\n--- 데모 {demo_id} ---")
            D._write_marker(marker_path, "S", demo_id)
            if action == "2":   # 안전위치 복귀 후 '다시 파지'
                grip_position = _grip(bridge, paxini)
            # action 1·2 공통: 스퀴즈 + 추론 (1은 현재 파지 그대로 재스퀴즈)
            _squeeze_and_infer(bridge, paxini, engine, grip_position, result_pub)
            D._write_marker(marker_path, "E", demo_id)
        print("\n시퀀스 완료.")
    finally:
        D._set_squeeze_flag(False)
        bridge.detach()
        executor.shutdown()
        # spin_thread 가 콜백 처리 중일 수 있으므로, 컨텍스트를 먼저 무효화해
        # spin() 이 ExternalShutdownException 으로 즉시 빠져나오게 한 뒤 join 한다.
        # (반대 순서로 하면 활발히 메시지가 오는 상황에서 join 이 걸려 destroy_node 와
        #  경합하다 멈추는 경우가 있었음)
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    # 라이브 DDS 네트워크(활성 노드 다수)에서는 rclpy.shutdown() 이후 남은 Node/
    # Publisher C 객체의 GC/인터프리터 종료 정리가 소켓 송신 대기로 오래 걸리거나
    # 멈출 수 있다. main() 안의 애플리케이션 레벨 정리(스레드 join 등)는 이미 끝났으니,
    # 이후의 네이티브 teardown 대기는 건너뛰고 바로 종료한다.
    _exit_code = 0
    try:
        main()
    except SystemExit as _e:
        if _e.code and not isinstance(_e.code, int):
            print(_e.code, file=sys.stderr)
        _exit_code = _e.code if isinstance(_e.code, int) else (1 if _e.code else 0)
    except KeyboardInterrupt:
        _exit_code = 130
    sys.stdout.flush(); sys.stderr.flush()   # P2#4: os._exit 전 flush (파이프/리다이렉트 시 마지막 출력 유실 방지)
    os._exit(_exit_code)
