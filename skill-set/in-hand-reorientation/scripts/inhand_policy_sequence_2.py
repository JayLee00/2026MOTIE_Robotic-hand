#!/usr/bin/env python3
"""시퀀스 2(Inhand) — VTDP 학습 정책 배선판. Start(제어권 획득) → 정책 실행 → End(DONE).

기존 inhand_sequence_2.py 의 HDF5 open-loop 재생 자리를 kist_deploy_pkg 의
visuo-tactile diffusion policy(run_kist_vtdp.py)로 교체한 것이다. 시퀀스 규약,
pose_commander 팔 이동, 마무리 goto+squeeze(살짝 폈다 다시 잡기)는 그대로 유지한다.

동작 순서:
  0) 정책 프로세스를 **미리** spawn (프리워밍) — 모델 로드 + CUDA 예열이 파지(seq 1)와
     겹쳐 돈다. 정책은 engage 전엔 /hand/*/q_target 을 한 건도 발행하지 않으므로 안전.
  1) Pick(1) DONE 대기 (wait_for_previous_done)
  2) Start(S): request_control → RUNNING + 하트비트
  3) pose_commander 로 목표 pose 이동 (기존과 동일)
  4) engage 발행(/teleop/hand_engage/right, RELIABLE+TRANSIENT_LOCAL) → 정책이
     100Hz 로 q_target 발행 시작 (cmd_mode=1/cmd_servo=true 는 정책이 첫 틱에 1회 발행)
  5) --policy-duration 초 동안 실행. /kist_vtdp/debug[15](halt) 감시 — halt/조기종료 = 실패
     ※ 정책은 스스로 종료하지 않는다(성공판정 없음). 시간 기반 종료는 임시안이고,
       FoundationPose 6DoF 기반 "라벨 축 정렬 시 종료"가 후속 과제다 (docs/DEV_REPORT.md).
  6) disengage → SIGINT → 프로세스 정리. 손은 마지막 타겟을 그대로 홀드한다(이완 아님).
  7) hand_goto_target(서서히 프리셋 이동) → hand_manual_squeeze(오므리기) — 재파지 마무리
  8) 정상 탈출 = End(E) → state=DONE → Stiffness(3)가 현재 손 자세를 이어받는다

실행 (conda 금지, /usr/bin/python3):
    source tools/env/setup_env.sh
    /usr/bin/python3 scripts/inhand_policy_sequence_2.py --policy-duration 15
사전 조건: 제어 PC sequence_arbiter + /hand/right/joint_states + /paxini/right/ft
          + /front_cam/.../compressed. 글러브 텔레옵(q_target 발행자 중복)은 꺼져 있을 것.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, Int32

from dual_arm_msgs.msg import SequenceState
from sequence_client import SequenceClient, SequenceError

HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent                                   # in-hand-reorientation/
PKG = SKILL_ROOT / 'kist_deploy_pkg'
VTDP_REPO = PKG / '1_policy' / 'kist-vtdp-wrapper'
RUN_KIST_VTDP = PKG / '1_policy' / 'diffusion_policy' / 'run_kist_vtdp.py'

# ── 정책 실행 계약 (kist_deploy_pkg/1_policy/DEPLOY.md) ─────────────────────
ENGAGE_TOPIC = '/teleop/hand_engage/right'
DEBUG_TOPIC = '/kist_vtdp/debug'
DEBUG_IDX_HALT = 15                  # Float64MultiArray[16] 의 halt 래치
READY_PATTERN = '예열 완료'          # run_kist_vtdp.py 가 추론 3회 예열 후 stdout 에 찍음
WARMUP_TIMEOUT_S = 240.0             # 모델 로드 + CUDA 콜드스타트 상한 (5090 첫 실행 여유)
POLICY_STOP_GRACE_S = 15.0           # SIGINT 후 정상 종료 대기

# engage 는 RELIABLE + TRANSIENT_LOCAL 이어야 정책이 받는다 (QoS 불일치 = 무수신)
LATCH = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                   durability=DurabilityPolicy.TRANSIENT_LOCAL,
                   history=HistoryPolicy.KEEP_LAST, depth=1)

# ── 팔 이동: pick 후 제시 자세 (inhand 이동 = stiffness·place 캡처가 물려받음) ──
# 2026-08-16: Cartesian pose_commander → 관절각 goto_q.py(MoveIt 충돌회피 plan→execute)
# 로 교체. 플랜 성공 = planning scene(정적 박스+자가충돌) 검사 통과.
# 구 좌표 (참고용 보존 — pose_commander stdin 형식, right_fr3_link0 기준 link8 pose):
#   POSE_TARGET = '0.2333 0.1590 -0.0668 -0.3373 0.2612 0.3084 0.8502'
ARM_Q_TARGET = ['-0.2866', '1.4185', '0.2677', '-1.9216', '0.7769', '1.2157', '2.0401']
GOTO_Q = str(SKILL_ROOT.parent.parent / 'tools' / 'goto_q.py')
POST_MOVE_DELAY = 1.0

# ── 마무리 재파지 (2026-08-16 사용자 지정: OPEN 2s 보간 → REGRIP 쥐기) ────────
# 정책 종료(홀드) 상태에서 mode 1 그대로: ① 살짝 펴는 자세로 2초 선형 보간
# ② 이어서 쥐는 자세로 보간 → 쥔 채 stiffness(3)에 인계.
# (thumb×4, index×4, middle×4, ring×4 — encoder counts)
HAND_OPEN_AFTER_POLICY = [4096, -4096, 0, 0,
                          0, 1500, 1500, 1500,
                          0, 1500, 1500, 1500,
                          0, 1500, 1500, 1500]
HAND_REGRIP_AFTER_OPEN = [4096, -4096, 2000, 2000,
                          0, 2000, 2000, 3000,
                          0, 2000, 2000, 3000,
                          0, 2000, 2000, 3000]
HAND_RAMP_S = 1.0            # 보간 시간 [s] (열기/쥐기 각각)
HAND_RAMP_HZ = 100.0         # 보간 발행 주기

# (구) goto+squeeze 마무리 — 위 OPEN/REGRIP 보간으로 대체, 참고 보존
# HAND_GOTO = str(SKILL_ROOT / 'in-hand' / 'hand_goto_target.py')
# HAND_SQUEEZE = str(SKILL_ROOT / 'in-hand' / 'hand_manual_squeeze.py')
# MANUAL_TARGET = ['1.57', '-1.0', '0.9897', '0.8907', '-0.2477', '0.4453', '1.1134',
#                  '0.5566', '0.0002', '0.4687', '0.9682', '0.5709', '0.2972', '0.4453',
#                  '0.8907', '0.4453']
# GOTO_RAMP_SECS = '3.0'; SQUEEZE_JOINTS = [1, 2, 5, 6, 9, 10, 14]
BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)


class PolicyError(RuntimeError):
    """정책 실행 실패 — with client: 안에서 던지면 abort(하트비트 정지 → IDLE 회수)."""


# ───────────────────────────────────────────────────────────────────────────
# 정책 프로세스
# ───────────────────────────────────────────────────────────────────────────
class PolicyProc:
    """run_kist_vtdp.py 를 원본 수정 없이 subprocess 로 부린다.

    - engage 전 무발행이므로 프리워밍을 위해 일찍 띄워도 안전하다 (DEPLOY.md).
    - stdout 리더 스레드가 READY_PATTERN 으로 예열 완료를 감지한다.
    - 종료: SIGINT → 정상 종료 대기 → SIGKILL. 손은 마지막 타겟 홀드(설계 의도).
    """

    def __init__(self, side: str, device: str, extra_args: list[str]):
        env = dict(os.environ, KIST_VTDP_REPO=str(VTDP_REPO))
        cmd = ['/usr/bin/python3', '-u', str(RUN_KIST_VTDP),
               '--side', side, '--device', device] + extra_args
        print(f'[inhand-policy] spawn: {" ".join(cmd)}', flush=True)
        self.ready = threading.Event()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(RUN_KIST_VTDP.parent), env=env, start_new_session=True)
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self):
        for line in self.proc.stdout:
            sys.stdout.write(f'    [policy] {line}')
            sys.stdout.flush()
            if READY_PATTERN in line:
                self.ready.set()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait_ready(self, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready.is_set():
                return
            if not self.alive():
                raise PolicyError(
                    f'정책 프로세스가 예열 전에 종료했다 (rc={self.proc.returncode}) — '
                    '위 [policy] 로그 확인 (torch/토픽/카메라 해상도)')
            time.sleep(0.5)
        raise PolicyError(f'정책 예열 타임아웃 ({timeout:.0f}s)')

    def stop(self):
        if self.alive():
            try:
                os.killpg(self.proc.pid, signal.SIGINT)   # Ctrl-C 상당 — 요약 출력 후 종료
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=POLICY_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                print('[inhand-policy] SIGINT 무시 → SIGKILL', file=sys.stderr)
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=5)
        print(f'[inhand-policy] 정책 종료 (rc={self.proc.returncode}) — 손은 홀드 유지',
              flush=True)


# ───────────────────────────────────────────────────────────────────────────
# engage 발행 + /kist_vtdp/debug 감시 (SequenceClient 노드와 독립)
# ───────────────────────────────────────────────────────────────────────────
class PolicyBridge:
    def __init__(self):
        self.node = rclpy.create_node('inhand_policy_wrapper')
        self._lock = threading.Lock()
        self._last_debug = None
        self.hand_q = None                     # 측정 손 관절 [16] counts
        self._hand_cmd = None                  # 우리가 마지막으로 명령한 타겟
        self.pub_engage = self.node.create_publisher(Bool, ENGAGE_TOPIC, LATCH)
        self.pub_hand = self.node.create_publisher(
            Float32MultiArray, '/hand/right/q_target', BEST_EFFORT)
        self.pub_mode = self.node.create_publisher(Int32, '/hand/right/cmd_mode', 10)
        self.node.create_subscription(
            JointState, '/hand/right/joint_states', self._on_hand, BEST_EFFORT)
        self.node.create_subscription(Float64MultiArray, DEBUG_TOPIC, self._on_debug, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spinning = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._spinning and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_debug(self, msg):
        with self._lock:
            self._last_debug = list(msg.data)

    def _on_hand(self, msg):
        if len(msg.position) >= 16:
            self.hand_q = [float(msg.position[j]) for j in range(16)]

    def hand_ramp(self, target, duration_s: float, label: str):
        """mode 1 유지한 채 손 타겟을 선형 보간으로 이동 (rclpy publish 는 스레드 안전 —
        executor 스레드가 spin 중이어도 메인 스레드에서 발행 가능)."""
        start = self._hand_cmd if self._hand_cmd is not None else \
            (list(self.hand_q) if self.hand_q is not None else None)
        if start is None:
            raise PolicyError('hand 측정(/hand/right/joint_states) 미수신 — 보간 시작점 없음')
        steps = max(1, int(duration_s * HAND_RAMP_HZ))
        print(f'[inhand-policy] hand 보간 [{label}] : {duration_s:.1f}s', flush=True)
        for i in range(1, steps + 1):
            t = i / steps
            q = [a + (b - a) * t for a, b in zip(start, target)]
            self.pub_hand.publish(Float32MultiArray(data=[float(v) for v in q]))
            time.sleep(1.0 / HAND_RAMP_HZ)
        self.pub_hand.publish(Float32MultiArray(data=[float(v) for v in target]))
        self._hand_cmd = list(target)
        print(f'[inhand-policy] hand 보간 [{label}] 완료', flush=True)

    def halted(self) -> bool:
        with self._lock:
            d = self._last_debug
        return bool(d and len(d) > DEBUG_IDX_HALT and d[DEBUG_IDX_HALT] != 0.0)

    def engage(self, on: bool):
        self.pub_engage.publish(Bool(data=on))
        print(f'[inhand-policy] engage={on}', flush=True)

    def q_target_publishers(self) -> int:
        return self.node.count_publishers('/hand/right/q_target')

    def shutdown(self):
        self._spinning = False
        self._thread.join(timeout=2.0)
        self.node.destroy_node()


# ───────────────────────────────────────────────────────────────────────────
# 기존 체인 재사용부 (inhand_sequence_2.py 와 동일 로직, 경로만 동적)
# ───────────────────────────────────────────────────────────────────────────
def _run_and_wait(cmd, label, stdin_text=None, cwd=None):
    print(f'[inhand-policy] launching {label}', flush=True)
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE if stdin_text is not None else None,
            text=True, cwd=cwd)
    except FileNotFoundError:
        print(f'[inhand-policy] 실행 실패(경로/워크스페이스 source 확인): {label}',
              file=sys.stderr)
        return None
    try:
        if stdin_text is not None:
            proc.stdin.write(stdin_text)
            proc.stdin.flush()
            proc.stdin.close()
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    print(f'[inhand-policy] {label} 종료 (returncode={proc.returncode})', flush=True)
    return proc.returncode


# (구) run_hand_goto / run_hand_squeeze — 2026-08-16 OPEN/REGRIP 보간으로 대체돼 제거

# ───────────────────────────────────────────────────────────────────────────
def run_policy_chain(node, bridge: PolicyBridge, policy: PolicyProc,
                     side: str, duration_s: float):
    print('in-hand manipulation start (VTDP policy)', flush=True)

    # 1) 팔 이동 — 관절각 목표로 MoveIt 충돌회피 이동 (플랜 실패 = 체인 중단)
    rc = _run_and_wait(['/usr/bin/python3', GOTO_Q, *ARM_Q_TARGET, '--yes'], 'goto_q.py')
    if rc != 0:
        raise PolicyError(f'팔 이동 실패 (goto_q rc={rc}) — 플랜/실행 로그 확인')
    node.get_logger().info(f'이동 완료 → {POST_MOVE_DELAY:.0f}초 대기')
    time.sleep(POST_MOVE_DELAY)

    # 2) 예열 확인 (파지 동안 대부분 끝나 있음)
    policy.wait_ready(WARMUP_TIMEOUT_S)
    if bridge.halted():
        raise PolicyError('정책이 engage 전에 halt 됐다 (카메라 해상도/intrinsics 확인)')

    # q_target 발행자 현황 — 정책 외 발행자(글러브 텔레옵 등)가 살아 있으면 손이 튄다.
    # place 서버 등이 발행자 객체만 등록해 둔 경우도 세어지므로 강제 중단은 하지 않는다.
    n_pub = bridge.q_target_publishers()
    if n_pub > 2:
        node.get_logger().warn(
            f'/hand/right/q_target 발행자 {n_pub}개 — 글러브 텔레옵이 켜져 있지 않은지 확인!')

    # 3) engage → 정책 구동
    bridge.engage(True)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < duration_s:
            if bridge.halted():
                raise PolicyError('정책 halt 래치 감지 (NaN 타겟/카메라 이상) — 체인 중단')
            if not policy.alive():
                raise PolicyError(f'정책 프로세스 조기 종료 (rc={policy.proc.returncode})')
            time.sleep(0.2)
    finally:
        # 4) disengage — 정책이 새 타겟 생성을 멈추고 손은 마지막 타겟 유지
        bridge.engage(False)
        time.sleep(0.5)
        policy.stop()

    node.get_logger().info(
        f'정책 {duration_s:.0f}s 완료 → 마무리: OPEN {HAND_RAMP_S:.0f}s 보간 → REGRIP 쥐기')
    # 5) 마무리 (2026-08-16 사용자 지정): mode 1 그대로 살짝 폈다가 다시 쥔다.
    #    시작점 = 측정 hand state (정책이 남긴 홀드 자세).
    bridge.pub_mode.publish(Int32(data=1))     # 단계 관례: Position 재확인 (1→1)
    time.sleep(0.1)
    bridge.hand_ramp(HAND_OPEN_AFTER_POLICY, HAND_RAMP_S, '살짝 펴기 (OPEN)')
    time.sleep(0.3)
    bridge.hand_ramp(HAND_REGRIP_AFTER_OPEN, HAND_RAMP_S, '다시 쥐기 (REGRIP)')
    # (구) run_hand_goto(side); run_hand_squeeze(side) — OPEN/REGRIP 보간으로 대체


def main():
    parser = argparse.ArgumentParser(
        description='시퀀스 2(Inhand, VTDP 정책): Pick DONE 대기 → Start → 정책 → End(DONE)')
    parser.add_argument('--hand-side', default='right')
    parser.add_argument('--wait-timeout', type=float, default=None,
                        help='Pick(1) DONE 대기 제한 [s] (기본: 무한 대기)')
    parser.add_argument('--policy-duration', type=float, default=15.0,
                        help='정책 실행 시간 [s] (정책은 스스로 종료하지 않는다)')
    parser.add_argument('--device', default='cuda:1',
                        help='정책 추론 GPU (FoundationPose/Molmo 와 분리: 기본 cuda:1)')
    parser.add_argument('--print-only', action='store_true',
                        help='정책/팔/손 실행 생략, Start/End 전이만 (규약 테스트용)')
    args, extra = parser.parse_known_args()

    if not RUN_KIST_VTDP.is_file():
        sys.exit(f'run_kist_vtdp.py 가 없다: {RUN_KIST_VTDP} — kist_deploy_pkg 배치 확인')

    rclpy.init()
    client = SequenceClient(SequenceState.SEQ_INHAND)   # client_id = seq_id = 2
    bridge = None
    policy = None
    try:
        if not args.print_only:
            bridge = PolicyBridge()
            # 프리워밍: Pick 이 도는 동안 모델 로드/예열을 겹쳐 끝낸다 (engage 전 무발행)
            policy = PolicyProc(args.hand_side, args.device, extra)

        print('[inhand-policy] Pick(1) DONE 대기 중...', flush=True)
        client.wait_for_previous_done(SequenceState.SEQ_PICK, timeout=args.wait_timeout)

        if policy is not None and not policy.alive():
            raise PolicyError(
                f'정책이 대기 중 죽었다 (rc={policy.proc.returncode}) — [policy] 로그 확인')

        with client:   # Start(S): request_control → RUNNING + 하트비트
            if args.print_only:
                print('[inhand-policy] print-only: 전이만 수행', flush=True)
            else:
                run_policy_chain(client._node, bridge, policy,
                                 args.hand_side, args.policy_duration)
        print('[inhand-policy] 완료: End(E) → DONE — Stiffness(3)가 손 자세를 이어받음',
              flush=True)
    except KeyboardInterrupt:
        print('[inhand-policy] Ctrl+C → abort (하트비트 정지 → 3초 후 IDLE 회수)',
              file=sys.stderr)
        sys.exit(130)
    except (SequenceError, PolicyError) as e:
        print(f'[inhand-policy] 실패: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        if policy is not None:
            policy.stop()
        if bridge is not None:
            bridge.shutdown()
        client.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
