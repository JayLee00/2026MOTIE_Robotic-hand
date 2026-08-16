"""시퀀스 제어권 클라이언트 — Start(S)/End(E)/하트비트/체이닝 래퍼.

제어 PC의 sequence_arbiter 프로토콜을 감싼다:
  Start(S) = /sequence/request_control 승인 → RUNNING (감사로그 "sq<id> S")
  End(E)   = /sequence/release_control       → DONE   (감사로그 "sq<id> E")
  하트비트  = 소유 중 /sequence/heartbeat(Int32=client_id) 계속 발행.
             heartbeat_timeout_sec(기본 3초) 끊기면 arbiter가 자동 회수 → IDLE

체이닝 규칙: 다음 시퀀스는 latched /sequence_state에서 {seq_id=N-1, state=DONE}
확인 후 시작한다. {N-1, IDLE}은 하트비트 타임아웃 회수(실패)이므로 진행 금지.

배정표 (팀 합의): seq_id = client_id = 1=Pick, 2=Inhand, 3=Stiffness, 4=Place.
상수는 dual_arm_msgs/msg/SequenceState의 SEQ_*.

NOTE: 현재는 각 시퀀스가 스스로 체이닝하지만, 추후 오케스트레이터 노드가
순서를 지시하는 방식으로 대체될 수 있음 (체인 재실행·다회 실행 조율 포함).
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Int32

from dual_arm_msgs.msg import SequenceState
from dual_arm_msgs.srv import ReleaseControl, RequestControl

# latched(/sequence_state) 수신에는 TRANSIENT_LOCAL 필수
LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class SequenceError(RuntimeError):
    """시퀀스 프로토콜 오류 공통 베이스."""


class ControlDenied(SequenceError):
    """request_control 거부 — 다른 owner가 제어권 보유 중."""


class PreviousAborted(SequenceError):
    """직전 시퀀스가 하트비트 타임아웃으로 회수됨(IDLE) — 진행 금지."""


class ArbiterUnavailable(SequenceError):
    """sequence_arbiter 서비스 미가동/무응답."""


class SequenceClient:
    """시퀀스 1회 실행의 Start/End/하트비트를 감싸는 클라이언트.

    사용 예 (시퀀스 2, Inhand):
        rclpy.init()
        client = SequenceClient(SequenceState.SEQ_INHAND)
        client.wait_for_previous_done(SequenceState.SEQ_PICK)
        with client:          # 진입=Start(S), 정상 탈출=End(E), 예외 탈출=abort()
            ...실제 동작...
        client.shutdown()
    """

    def __init__(self, seq_id, client_id=0, heartbeat_hz=2.0):
        # client_id=0(미지정) → 배정표대로 seq_id와 동일 번호 사용
        self.seq_id = int(seq_id)
        self.client_id = int(client_id) if client_id != 0 else self.seq_id
        self._heartbeat_period = 1.0 / heartbeat_hz
        self._node = Node(f"sequence_client_{self.client_id}")
        self._last_state = None
        self._node.create_subscription(
            SequenceState, "/sequence_state", self._on_state, LATCHED_QOS)
        self._pub_heartbeat = self._node.create_publisher(
            Int32, "/sequence/heartbeat", 10)
        self._cli_request = self._node.create_client(
            RequestControl, "/sequence/request_control")
        self._cli_release = self._node.create_client(
            ReleaseControl, "/sequence/release_control")
        self._hb_stop = threading.Event()
        self._hb_thread = None

    # ── 체이닝 ──
    def wait_for_previous_done(self, prev_seq_id, timeout=None):
        """직전 시퀀스 완료 대기. DONE → 리턴, IDLE(회수) → PreviousAborted."""
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        while True:
            if not rclpy.ok():
                raise SequenceError("rclpy 종료됨")
            rclpy.spin_once(self._node, timeout_sec=0.2)
            st = self._last_state
            if st is not None and st.seq_id == prev_seq_id:
                if st.state == SequenceState.DONE:
                    return
                if st.state == SequenceState.IDLE:
                    raise PreviousAborted(
                        f"sq{prev_seq_id} 하트비트 타임아웃으로 회수됨(IDLE)")
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"sq{prev_seq_id} DONE 대기 {timeout}s 초과")

    # ── Start / End / abort ──
    def start(self, timeout=5.0):
        """Start(S): request_control 승인 + 하트비트 발행 시작."""
        res = self._call(
            self._cli_request,
            RequestControl.Request(client_id=self.client_id, seq_id=self.seq_id),
            timeout)
        if not res.granted:
            raise ControlDenied(f"거부 (owner={res.current_owner}): {res.message}")
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    def end(self, timeout=5.0):
        """End(E): release_control → DONE (다음 시퀀스가 이어받음) + 하트비트 정지."""
        try:
            res = self._call(
                self._cli_release,
                ReleaseControl.Request(client_id=self.client_id, seq_id=self.seq_id),
                timeout)
            if not res.released:
                raise SequenceError(f"반납 실패: {res.message}")
        finally:
            self._stop_heartbeat()

    def abort(self):
        """비정상 종료: release 없이 하트비트만 정지.

        heartbeat_timeout_sec(기본 3초) 후 arbiter가 자동 회수(IDLE).
        DONE을 내지 않으므로 다음 시퀀스가 잘못 진행하지 않는다.
        (프로세스 크래시 시에도 데몬 스레드가 죽어 동일 동작)
        """
        self._stop_heartbeat()

    def shutdown(self):
        self._stop_heartbeat()
        self._node.destroy_node()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.end()
        else:
            self.abort()
        return False

    # ── 내부 ──
    def _on_state(self, msg):
        self._last_state = msg

    def _call(self, client, request, timeout):
        if not client.wait_for_service(timeout_sec=timeout):
            raise ArbiterUnavailable(
                f"{client.srv_name} 서비스 없음 — arbiter 실행 중인지 확인")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout)
        if future.result() is None:
            raise ArbiterUnavailable(f"{client.srv_name} 응답 없음")
        return future.result()

    def _heartbeat_loop(self):
        # 발행은 spin 불필요 → 단순 데몬 스레드 (RELIABLE pub ↔ arbiter의
        # best_effort sub는 QoS 호환)
        msg = Int32(data=self.client_id)
        self._pub_heartbeat.publish(msg)
        while not self._hb_stop.wait(self._heartbeat_period):
            self._pub_heartbeat.publish(msg)

    def _stop_heartbeat(self):
        self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=1.0)
            self._hb_thread = None
