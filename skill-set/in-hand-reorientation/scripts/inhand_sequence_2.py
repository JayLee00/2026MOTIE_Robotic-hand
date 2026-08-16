#!/usr/bin/env python3
"""시퀀스 2(Inhand) 정식 클라이언트 — Start(제어권 획득) → 작업 → End(DONE).

docs_dev/SEQUENCE_GUIDE.md 규칙 그대로:
  배정표(고정): 1=Pick, 2=Inhand, 3=Stiffness, 4=Place (client_id도 동일 번호)
  state: 0=IDLE(대기/실패회수) · 1=RUNNING · 2=DONE(정상종료)

동작 순서:
  1) Pick(1) DONE 대기 (wait_for_previous_done) — 직전 시퀀스가 정상 종료해야 진행
  2) Start(S): request_control 승인 → arbiter가 seq_id=2 RUNNING, owner=2로 전이
              + 하트비트 자동 발행 → 여기서 "in-hand manipulation start" 출력
  3) 실제 작업(블로킹, 완료까지 대기):
       pose_commander.py 로 목표 pose 이동 → 2초 → hand_joint_target_publisher(HDF5 재생, 완료 대기)
       → hand_goto_target (manual target 위치로 서서히 이동) → hand_manual_squeeze ('+' 2회) → 종료
  4) 정상 탈출 = End(E): release_control → state=DONE
     → 다음 시퀀스 Stiffness(3)가 이어받고, 이어서 Place(4)까지 진행 가능
     예외/Ctrl+C = abort(): release 없이 하트비트 정지 → 3초 후 IDLE 회수(실패)

** 실행 (conda python 아님, /usr/bin/python3): **
    source /opt/ros/humble/setup.bash
    source ~/isaac_ws/dex_soldering/dex_ros/isaac-ros/kistar_ws/install/setup.bash
    # ↑ dual_arm_msgs, sequence_client, franka_kistar_bringup 모두 이 워크스페이스에 빌드됨
    /usr/bin/python3 scripts/inhand_sequence.py
사전 조건: 제어 PC에서 sequence_arbiter 실행 중.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import rclpy

from dual_arm_msgs.msg import SequenceState
from sequence_client import SequenceClient, SequenceError

REPO_ROOT = '/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/in-hand-reorientation'

# Start 직후 실행할 pose_commander 명령
POSE_COMMANDER_CMD = [
    'ros2', 'run', 'franka_kistar_bringup', 'pose_commander.py', '--ros-args',
    '-p', 'gui:=true',
    '-p', 'planning_group:=right_arm',
    '-p', 'end_effector_link:=right_fr3_link8',
    '-p', 'reference_frame:=right_fr3_link0',
    '-p', 'planning_time:=5.0',
    '-p', 'traj_action:=/right_arm_controller/follow_joint_trajectory',
]
# pose_commander.py stdin: 목표 pose(x y z qx qy qz qw) → 실행 확인(y) → 종료(quit)
# ('quit'이 있어야 _input_loop가 종료되어 proc.wait()가 "이동 완료" 신호가 된다)
POSE_TARGET = '0.2333 0.1590 -0.0668 -0.3373 0.2612 0.3084 0.8502'
POSE_COMMANDER_STDIN = f'{POSE_TARGET}\ny\nquit\n'

# pose 이동 완료 후 대기 시간 [s]
POST_MOVE_DELAY = 1.0

# 이동 완료 + POST_MOVE_DELAY 후 실행할 hand joint target publisher (완료까지 대기).
# HDF5 궤적 파일(--file, 기본 in-hand/data/test_int.hdf5)을 재생해 손 관절 목표를 발행.
HAND_PUBLISHER = os.path.join(REPO_ROOT, 'in-hand', 'hand_joint_target_publisher.py')

# publisher 완료 후 실행할 hand manual squeeze (오무리기 2회 후 종료).
HAND_SQUEEZE = os.path.join(REPO_ROOT, 'in-hand', 'hand_manual_squeeze.py')
# 스퀴즈에 참여할 real-order 조인트. 기본 마스크 (1,2,5,6,9,10,13,14)에서
# thumb_joint_1(=real-order 13, URDF/MoveIt hand joint 순서 기준)을 제외해 고정한다.
#   0-3 index_joint_0..3 / 4-7 middle / 8-11 ring / 12-15 thumb_joint_0..3
SQUEEZE_JOINTS = [1, 2, 5, 6, 9, 10, 14]
SQUEEZE_KEY = '+'            # 오무리는 방향 ('+'=tighten/오므리기, '-'=loosen)
SQUEEZE_PRESSES = 1          # '+' 입력 횟수 (오무리기 2회)
SQUEEZE_READY_TIMEOUT = 12.0  # squeeze 노드가 명령 받을 준비될 때까지 최대 대기 [s]
SQUEEZE_STARTUP_WAIT = 1.0   # 준비 확인 후 servo/mode 안정 대기 [s]
SQUEEZE_INTERVAL = 1.5       # 각 '+' 후 슬루 완료+홀드 대기 [s]

# squeeze 직전에 손가락을 서서히 옮길 manual target joint 위치 (radians, 16관절).
# publisher 와 동일한 cube_rotate 인코딩으로 encoder count 변환 후 소프트 램프로 이동.
HAND_GOTO = os.path.join(REPO_ROOT, 'in-hand', 'hand_goto_target.py')
#
# ── Topdown_Grasp configs/hand.yaml 의 hand_grasp 포즈와 물리적으로 동일 ──────
# 두 스크립트 모두 같은 /hand/right/q_target(Float32[16] count) receiver 로 발행하므로
# 배열 위치 k = 같은 물리 모터다. 아래 값은 hand_goto_target.py 의 cube_rotate 인코딩
# 을 거친 뒤, Topdown(run_topdown_grasp.py: enc=rad*8192/pi, DRO→HW 재정렬)이 내는
# encoder count 와 위치별로 100% 일치하도록 역산한 값이다 (검증 완료).
#   순서(HW/발행 순): thumb×4, index×4, middle×4, ring×4  (기존 주석의 index-first 는 오기)
#   기준 hand.yaml 각도(deg): thumb j1=90 j2=-90 / finger abd=±15, j2=30, j3=60, j4=30
# ※ hand.yaml 이 바뀌면 scratchpad/convert_pose.py 로 재생성할 것.
# 이전 값(참고):
# MANUAL_TARGET = ['1.575','-1.571','0.162','0.692', '-0.298','1.','0.832','0.168',
#                  '-0.183','1.','0.521','0.295', '0.526','1.','0.662','0.009']
MANUAL_TARGET = [
    '1.57', '-1.0', '0.9897', '0.8907',   # thumb  (j1=90, j2=-90, +2 bend=60)
    '-0.2477', '0.4453', '1.1134', '0.5566',   # index  (abd=-15, j2=30, j3=60, j4=30)
    '0.0002',  '0.4687', '0.9682', '0.5709',   # middle (abd=0,   j2=30, j3=60, j4=30)
    '0.2972',  '0.4453', '0.8907', '0.4453',   # ring   (abd=+15, j2=30, j3=60, j4=30)
]
GOTO_RAMP_SECS = '3.0'       # manual target 까지 서서히 이동할 시간 [s]


def _run_and_wait(cmd, label, stdin_text=None, cwd=None):
    """자식 프로세스를 실행하고 종료까지 대기. 반환코드 반환(실행 실패 시 None)."""
    print(f'[inhand] launching {label}', flush=True)
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE if stdin_text is not None else None,
            text=True, cwd=cwd)
    except FileNotFoundError:
        print(f'[inhand] 실행 실패(경로/워크스페이스 source 확인): {label}', file=sys.stderr)
        return None
    try:
        if stdin_text is not None:
            proc.stdin.write(stdin_text)
            proc.stdin.flush()
            proc.stdin.close()
        proc.wait()
    except KeyboardInterrupt:
        # Ctrl+C: 자식 정리 후 상위로 전파 (→ SequenceClient가 abort 처리)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    print(f'[inhand] {label} 종료 (returncode={proc.returncode})', flush=True)
    return proc.returncode


def run_pose_commander(node):
    """pose_commander 실행 → 로봇 이동 완료까지 대기 (프로세스 종료 = 이동 완료)."""
    node.get_logger().info('launching pose_commander.py')
    _run_and_wait(POSE_COMMANDER_CMD, 'pose_commander.py',
                  stdin_text=POSE_COMMANDER_STDIN)


def run_hand_publisher(node, side):
    """hand_joint_target_publisher 실행 → HDF5 재생 완료까지 대기 (블로킹).

    squeeze 와 같은 /hand/{side}/q_target 토픽을 쓰므로, 겹치지 않도록 이 publisher
    가 끝난(프로세스 종료 = 재생 완료) 뒤에 squeeze 를 수행한다.
    """
    node.get_logger().info(f'launching hand_joint_target_publisher.py --side {side}')
    _run_and_wait(['/usr/bin/python3', HAND_PUBLISHER, '--side', side],
                  'hand_joint_target_publisher.py', cwd=REPO_ROOT)


def run_hand_goto(node, side):
    """손가락을 MANUAL_TARGET(radians) 위치로 서서히 이동 (완료까지 대기).

    hand_goto_target.py 가 measured pose → target 을 소프트 램프(GOTO_RAMP_SECS)로
    이동시킨 뒤 종료한다. servo 는 켜둔 채 끝나므로 이어서 squeeze 가 그 자세를
    baseline 으로 잡고 오무릴 수 있다.
    """
    node.get_logger().info(
        f'launching hand_goto_target.py --side {side} (ramp {GOTO_RAMP_SECS}s)')
    _run_and_wait(
        ['/usr/bin/python3', HAND_GOTO, '--side', side,
         '--ramp-secs', GOTO_RAMP_SECS, '--target', *MANUAL_TARGET],
        'hand_goto_target.py', cwd=REPO_ROOT)


def run_hand_squeeze(hand_side, presses=SQUEEZE_PRESSES, interval=SQUEEZE_INTERVAL):
    """hand_manual_squeeze를 실행하고 SQUEEZE_KEY('+')를 presses회 눌러 오무린 뒤 종료.

    각 '+'는 offset_goal만 바꾸고 노드가 --speed로 점진 슬루하므로, 프레스 사이에
    interval만큼 대기해 실제 손가락 동작이 끝나도록 한다.

    중요: 노드 기동(서브스크라이버 대기·servo·베이스라인 캡처)이 끝나 input_loop가
    명령을 받을 준비가 됐음을 stdout('commands:' 프롬프트)으로 확인한 뒤 프레스를
    보낸다. 준비 전에 몰아 보내면 파이프에 쌓였다가 한꺼번에 소비되어(슬루 전에 quit)
    손가락이 안 움직인다.
    """
    label = f'hand_manual_squeeze.py --side {hand_side}'
    print(f'[inhand] launching {label}', flush=True)
    try:
        proc = subprocess.Popen(
            ['/usr/bin/python3', HAND_SQUEEZE, '--side', hand_side,
             '--joints', *[str(j) for j in SQUEEZE_JOINTS]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=REPO_ROOT)
    except FileNotFoundError:
        print(f'[inhand] 실행 실패(경로 확인): {label}', file=sys.stderr)
        return None

    # 자식 출력을 계속 흘려보내며(파이프 막힘 방지) 준비 신호를 감지하는 리더 스레드
    ready = threading.Event()

    def _reader():
        for line in proc.stdout:
            sys.stdout.write(f'    [squeeze] {line}')
            sys.stdout.flush()
            if 'commands:' in line or 'squeeze step=' in line:
                ready.set()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        ready.wait(timeout=SQUEEZE_READY_TIMEOUT)
        if proc.poll() is not None:
            print('[inhand] squeeze 노드 조기 종료 (로봇 RT/receiver·joint_states 확인)',
                  file=sys.stderr)
            return proc.returncode
        if not ready.is_set():
            print('[inhand] squeeze 준비 신호 지연 — 그래도 진행', file=sys.stderr)
        time.sleep(SQUEEZE_STARTUP_WAIT)  # servo/mode 안정 대기

        for i in range(presses):
            print(f'[inhand] squeeze {i + 1}/{presses} ({SQUEEZE_KEY})', flush=True)
            proc.stdin.write(f'{SQUEEZE_KEY}\n')
            proc.stdin.flush()
            time.sleep(interval)          # 슬루 완료 + 홀드
        proc.stdin.write('quit\n')        # 정상 종료 (target 유지, servo 그대로)
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait()
    except (KeyboardInterrupt, BrokenPipeError):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    print(f'[inhand] {label} 종료 (returncode={proc.returncode})', flush=True)
    return proc.returncode


def run_inhand_chain(node, run_command, side):
    """트리거 시 실행되는 전체 체인 (블로킹):
    pose → hand publisher(HDF5 완료 대기) → squeeze 2회 → 종료."""
    print('in-hand manipulation start', flush=True)
    if not run_command:
        return
    # 1~2) pose_commander 실행 → 이동 완료까지 대기
    run_pose_commander(node)
    # 3) 이동 완료 후 대기
    node.get_logger().info(f'이동 완료 → {POST_MOVE_DELAY:.0f}초 후 hand publisher 실행')
    time.sleep(POST_MOVE_DELAY)
    # 4) hand publisher 실행 → HDF5 재생 완료까지 대기
    run_hand_publisher(node, side)
    # 5) squeeze 직전에 손가락을 manual target 위치로 서서히 이동 (완료까지 대기)
    node.get_logger().info('hand publisher 완료 → manual target 위치로 서서히 이동')
    run_hand_goto(node, side)
    # 6) manual target 도달 후 squeeze 2회 → (반환 시 End(E)/DONE)
    node.get_logger().info('manual target 도달 → squeeze 2회 수행 후 종료')
    run_hand_squeeze(side)


def main():
    parser = argparse.ArgumentParser(
        description='시퀀스 2(Inhand): Pick DONE 대기 → Start → pose→2초→hand → End(DONE)')
    parser.add_argument('--print-only', action='store_true',
                        help='pose/hand 실행 생략, Start/End(DONE) 전이만 수행(테스트용)')
    parser.add_argument('--hand-side', default='right',
                        help='hand_joint_target_publisher --side 값 (기본: right)')
    parser.add_argument('--wait-timeout', type=float, default=None,
                        help='Pick(1) DONE 대기 제한 [s] (기본: 무한 대기)')
    args = parser.parse_args()

    rclpy.init()
    client = SequenceClient(SequenceState.SEQ_INHAND)  # client_id = seq_id = 2
    try:
        print('[inhand] Pick(1) DONE 대기 중...', flush=True)
        client.wait_for_previous_done(SequenceState.SEQ_PICK, timeout=args.wait_timeout)

        with client:  # Start(S): request_control → RUNNING + 하트비트
            run_inhand_chain(client._node,
                             run_command=not args.print_only, side=args.hand_side)
        # 정상 탈출 = End(E) → state=DONE
        print('[inhand] 완료: End(E) → state=DONE — Stiffness(3)가 이어받음', flush=True)
    except KeyboardInterrupt:
        print('[inhand] Ctrl+C → abort (release 없이 하트비트 정지 → 3초 후 IDLE 회수)',
              file=sys.stderr)
        sys.exit(130)
    except SequenceError as e:
        print(f'[inhand] 실패: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        client.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
