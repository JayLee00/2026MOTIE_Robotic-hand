#!/usr/bin/env python3
"""
Atomic step functions for Grasp_fruit executors.

각 함수는 GraspExecutor 인스턴스를 받아 단일 동작을 수행한다.
조합 예시:

    result = step_approach(node, approach, confirm=True)    # → (joints, traj)
    j1, approach_traj = result
    result = step_descend(node, target, seed=j1, confirm=True)  # → (joints, traj)
    j2, descend_traj = result
    step_close_hand(node, enc, confirm=True)
    step_lift(node, approach, seed=j2, descend_traj=descend_traj, confirm=True)
    step_go_home(node, approach_traj=approach_traj)   # approach 역재생 → HOME
"""

from __future__ import annotations
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration


# ---------------------------------------------------------------------------
# 궤적 유틸리티
# ---------------------------------------------------------------------------

def reverse_trajectory(jt: JointTrajectory) -> JointTrajectory:
    """JointTrajectory를 시간 역전하여 반환 (LIFT = reversed DESCEND).

    approach→target 하강 궤적을 저장해두면, 동일 경로를 역재생하여
    IK 재계산 없이 100% 성공률로 target→approach 상승이 가능하다.
    """
    rev = JointTrajectory()
    rev.joint_names = list(jt.joint_names)

    if not jt.points:
        return rev

    total = (jt.points[-1].time_from_start.sec +
             jt.points[-1].time_from_start.nanosec * 1e-9)

    for pt in reversed(jt.points):
        t_orig = (pt.time_from_start.sec +
                  pt.time_from_start.nanosec * 1e-9)
        t_new  = total - t_orig
        sec    = int(t_new)
        nsec   = int(round((t_new - sec) * 1e9))

        new_pt = JointTrajectoryPoint()
        new_pt.positions       = list(pt.positions)
        if list(pt.velocities):
            new_pt.velocities  = [-v for v in pt.velocities]
        new_pt.time_from_start = RosDuration(sec=sec, nanosec=nsec)
        rev.points.append(new_pt)

    return rev


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _exec(node, jt: JointTrajectory, post_delay: float = 0.0) -> float:
    """trajectory 실행 → 실제 완료까지 대기 후 반환.

    우선 경로 (pose_commander.py 방식):
      전체 JointTrajectory 를 FollowJointTrajectory 액션(/right_arm_controller/…)으로
      1회 전송하면 컨트롤러가 waypoint 사이를 스플라인 보간해 한 번에 부드럽게 실행한다.
      액션 result 가 완료 신호이므로 타이밍 추정/폴링이 필요 없다.

    fallback (액션 서버 없을 때):
      기존 q_target 스트리밍 후 joint 수렴 polling. bridge 가 없을 때만 사용.
    """
    dur = (jt.points[-1].time_from_start.sec +
           jt.points[-1].time_from_start.nanosec * 1e-9)

    if getattr(node, '_traj_client', None) is not None:
        node._exec_traj_action(jt)   # 액션 전송 + 로봇 실제 완료 대기 (블로킹)
        if post_delay > 0:
            time.sleep(post_delay)
        return dur

    # ── fallback: q_target 스트리밍 + 수렴 polling ──
    node._exec_arm(jt)   # execute_mode 에 맞는 발행 (direct_franka_topic → q_target 스트리밍)
    target = list(jt.points[-1].positions)

    # 팔이 출발하기 전에 polling 하면 현재 위치 ≈ 목표로 오판할 수 있으므로
    # duration의 20% 또는 최소 0.5s 대기 후 polling 시작
    min_wait = max(0.5, dur * 0.2)
    time.sleep(min_wait)

    if node._current_joints is not None:
        # 실제 도달까지 polling → 타이머 짧으면 더 기다리고, 길면 일찍 반환
        node._wait_for_motion(target, timeout=dur + 10.0, tol=0.05)
    else:
        # /franka/joint_position 미수신 → timer fallback
        node._wait_for_traj(max(0.0, dur - min_wait))

    if post_delay > 0:
        time.sleep(post_delay)
    return dur


def _plan(node, goal, label: str, seed=None,
          confirm: bool = True) -> JointTrajectory | None:
    """_plan_step 래퍼. seed 가 있으면 jvals 로 전달."""
    from utils.arm import HOME_JOINT_NAMES
    jnames = HOME_JOINT_NAMES if seed is not None else None
    return node._plan_step(goal, label,
                           jnames=jnames, jvals=seed,
                           confirm=confirm)


# ---------------------------------------------------------------------------
# Arm 이동 steps
# ---------------------------------------------------------------------------

def step_approach(node, approach_pose, seed=None,
                  confirm: bool = True) -> tuple[list, JointTrajectory] | None:
    """현재 → approach 위치 이동.
    Returns: (joint_values, trajectory) 또는 None.
    trajectory는 step_go_home(approach_traj=...) 으로 역재생 가능.
    """
    jt = _plan(node, approach_pose, 'APPROACH', seed=seed, confirm=confirm)
    if jt is None:
        node.get_logger().error('[step_approach] 실패')
        return None
    _exec(node, jt)
    return list(jt.points[-1].positions), jt


def step_approach_descend(node, approach_pose, target_pose, seed=None,
                          confirm: bool = True):
    """APPROACH→TARGET 을 한 궤적으로 실행 (접근점 무정지 통과).

    두 구간을 기존과 동일하게 각각 Cartesian 계획(충돌검사 포함)한 뒤,
    waypoint 를 이어붙여 한 번에 재타이밍(_add_timestamps) → 접근점에서
    속도 0 으로 멈추지 않고 감속-통과 후 이어서 하강한다. 경로 자체는
    기존 2-step 과 동일 (직선 접근 + 수직 하강).

    Returns:
      (joints, approach_traj, descend_traj)  성공
      None   계획 실패 → 호출측에서 기존 2-step 폴백
      False  사용자 취소
    """
    from utils.arm import HOME_JOINT_NAMES
    from moveit_msgs.msg import RobotTrajectory

    def _plan_seg(pose, label, jn, jv):
        for ms in (0.01, 0.02, 0.05):
            cart = node._plan_cartesian(pose, label, jnames=jn, jvals=jv,
                                        max_step=ms)
            if cart is not None and cart.fraction >= 0.999:
                return cart
        return None

    jnames = HOME_JOINT_NAMES if seed is not None else None
    cartA = _plan_seg(approach_pose, 'APPROACH', jnames, seed)
    if cartA is None:
        return None
    jtA = cartA.solution.joint_trajectory
    jA  = list(jtA.points[-1].positions)

    cartB = _plan_seg(target_pose, 'TARGET', HOME_JOINT_NAMES, jA)
    if cartB is None:
        return None
    jtB = cartB.solution.joint_trajectory

    # 경로 이어붙여 한 번에 재타이밍 → 접근점 무정지 (모서리 자동 감속)
    combined = JointTrajectory()
    combined.joint_names = list(jtA.joint_names)
    combined.points = list(jtA.points) + list(jtB.points[1:])
    jt = node._add_timestamps(combined)

    rt = RobotTrajectory()
    rt.joint_trajectory = jt
    node._display_trajectory(cartA.start_state, rt, 'APPROACH+TARGET')
    if confirm and not node._confirm(
            '  [APPROACH+TARGET] 실행하시겠습니까? (y/n): '):
        print('  [APPROACH+TARGET] 취소됨.')
        return False

    _exec(node, jt)

    # LIFT / HOME 역재생용: 구간별 개별 타이밍 버전 (경로 동일)
    approach_traj = node._add_timestamps(jtA)
    descend_traj  = node._add_timestamps(jtB)
    return list(jt.points[-1].positions), approach_traj, descend_traj


def step_descend(node, target_pose, seed=None,
                 confirm: bool = True) -> tuple[list, JointTrajectory] | None:
    """approach → target (수직 하강).
    Returns: (joint_values, trajectory) 또는 None.
    trajectory는 step_lift(descend_traj=...) 으로 역재생 가능.
    """
    jt = _plan(node, target_pose, 'TARGET', seed=seed, confirm=confirm)
    if jt is None:
        node.get_logger().error('[step_descend] 실패')
        return None
    _exec(node, jt)
    return list(jt.points[-1].positions), jt


def step_lift(node, approach_pose, seed=None,
              descend_traj: JointTrajectory | None = None,
              confirm: bool = True) -> list | None:
    """target → approach 복귀 (lift).
    descend_traj가 있으면 역재생, 없으면 재계획.
    """
    if descend_traj is not None:
        node.get_logger().info('[LIFT] 하강 궤적 역재생 (IK 재계산 없음)')
        jt = reverse_trajectory(descend_traj)
        if confirm:
            print('\n  [LIFT] 하강 경로 역재생으로 상승')
            if not node._confirm('  [LIFT] 실행하시겠습니까? (y/n): '):
                print('  [LIFT] 취소됨.')
                return None
        _exec(node, jt)
        return list(jt.points[-1].positions)

    jt = _plan(node, approach_pose, 'LIFT', seed=seed, confirm=confirm)
    if jt is None:
        node.get_logger().error('[step_lift] 실패')
        return None
    _exec(node, jt)
    return list(jt.points[-1].positions)


def step_move_to_pose(node, pose, label: str,
                      seed=None, confirm: bool = False,
                      post_delay: float = 0.5) -> list | None:
    """임의 pose로 이동 (자동 동작 세트용)."""
    jt = _plan(node, pose, label, seed=seed, confirm=confirm)
    if jt is None:
        node.get_logger().error(f'[{label}] 이동 실패')
        return None
    _exec(node, jt, post_delay=post_delay if not confirm else 0.0)
    return list(jt.points[-1].positions)


# ---------------------------------------------------------------------------
# Home 이동
# ---------------------------------------------------------------------------

def step_go_home(node, confirm: bool = False,
                 post_delay: float = 0.5,
                 approach_traj: JointTrajectory | None = None) -> list:
    """HOME 관절값으로 이동.
    approach_traj가 있으면 역재생 (100% 성공), 없으면 OMPL/direct fallback.
    """
    from utils.arm import HOME_JOINT_NAMES, HOME_JOINT_VALUES

    if approach_traj is not None:
        node.get_logger().info('[HOME] approach 역재생으로 복귀 (IK 재계산 없음)')
        jt = reverse_trajectory(approach_traj)
        if confirm:
            print('\n  [HOME] approach 경로 역재생으로 복귀')
            if not node._confirm('  [HOME] 실행하시겠습니까? (y/n): '):
                print('  [HOME] 취소됨.')
                return list(HOME_JOINT_VALUES)
        _exec(node, jt, post_delay=post_delay if not confirm else 0.0)
        node.get_logger().info('[HOME] 완료')
        return list(HOME_JOINT_VALUES)

    node.get_logger().info('[step_go_home] 홈 이동 중...')
    res = node._plan_joints(HOME_JOINT_NAMES, HOME_JOINT_VALUES, 'HOME')
    if res is not None:
        jt = res.planned_trajectory.joint_trajectory
        if confirm:
            node._display_trajectory(res.trajectory_start, res.planned_trajectory, 'HOME')
            if not node._confirm('  [HOME] 초기자세로 이동하시겠습니까? (y/n): '):
                print('  [HOME] 취소됨.')
                return list(HOME_JOINT_VALUES)
    else:
        node.get_logger().warning('[HOME] MoveGroup 실패 → direct trajectory')
        jt = node._make_joint_traj(list(HOME_JOINT_VALUES))
        if confirm and not node._confirm('  [HOME-fallback] 초기자세로 이동하시겠습니까? (y/n): '):
            print('  [HOME] 취소됨.')
            return list(HOME_JOINT_VALUES)
    _exec(node, jt, post_delay=post_delay if not confirm else 0.0)
    node.get_logger().info('[step_go_home] 완료')
    return list(HOME_JOINT_VALUES)


# ---------------------------------------------------------------------------
# Hand steps
# ---------------------------------------------------------------------------

def step_init_hand(node) -> None:
    """핸드를 HAND_INIT_ENC (대기/충돌회피 자세) 로 이동 (확인 없음)."""
    from utils.hand import HAND_INIT_ENC, HAND_STEPS, HAND_PERIOD
    # from std_msgs.msg import Int16MultiArray            # 옛 (Int16)
    from std_msgs.msg import Float32MultiArray            # 새 제어 PC (/hand/right/q_target)

    start  = node._last_hand_enc if node._last_hand_enc else list(HAND_INIT_ENC)
    target = list(HAND_INIT_ENC)

    if max(abs(s - t) for s, t in zip(start, target)) < 50:
        return  # 이미 충분히 가까움

    node.get_logger().info('[HAND_INIT] 대기 자세로 이동...')
    for i in range(1, HAND_STEPS + 1):
        alpha  = i / HAND_STEPS
        interp = [max(-32768, min(32767, int(round(s + alpha * (g - s)))))
                  for s, g in zip(start, target)]
        # msg = Int16MultiArray(); msg.data = interp        # 옛 (Int16)
        msg      = Float32MultiArray()
        msg.data = [float(v) for v in interp]
        node._hand_pub.publish(msg)
        if i < HAND_STEPS:
            time.sleep(HAND_PERIOD)
    node._last_hand_enc = list(HAND_INIT_ENC)
    node.get_logger().info('[HAND_INIT] 완료')


def step_close_hand(node, enc: list, confirm: bool = True) -> bool:
    """손 파지. Returns True(실행) / False(취소)."""
    if confirm and not node._confirm('  [HAND] 손가락 파지하시겠습니까? (y/n): '):
        print('  [HAND] 취소됨.')
        return False
    node._exec_hand(enc)
    return True


def step_release_hand(node, confirm: bool = False) -> None:
    """손 열기 (기본 확인 없음)."""
    if confirm and not node._confirm('  [RELEASE] 물체를 놓겠습니까? (y/n): '):
        print('  [RELEASE] 건너뜀.')
        return
    node._do_release_hand()


def step_place_from_home(node, place_z_descent: float) -> bool:
    """HOME → 하강 (Cartesian 계획) → release → 상승 (역재생 HOME 복귀).

    흐름:
      (HOME) → PLACE_DESCENT [Cartesian 계획, 1회만]
             → RELEASE
             → PLACE_ASCENT  [역재생, planning 없음]
             → (HOME)

    step_go_home()가 이미 HOME에 도착한 상태에서 호출해야 한다.
    완료 후 별도 step_go_home() 불필요.

    Returns: True 성공, False 실패
    """
    from utils.arm import HOME_JOINT_VALUES
    from utils.grasp import base_to_world, world_to_base

    # step_go_home 직후 속도 감속 여유 (trajectory duration 기반 wait는
    # _exec 내부에서 이미 처리됨 — 여기서는 settling만 보장)
    time.sleep(0.5)

    node.get_logger().info('[step_place_from_home] FK로 HOME EE 위치 계산 중...')
    home_ee = node._compute_fk(list(HOME_JOINT_VALUES))
    if home_ee is None:
        node.get_logger().error('[step_place_from_home] FK 실패')
        return False

    xyz_b  = home_ee[:3]
    quat_b = home_ee[3:]  # [qx, qy, qz, qw]

    T_wb = node._summary.get('T_world_base')
    if T_wb is not None:
        # base 프레임 FK 결과를 world 프레임으로 변환 → world Z로 하강 → base 역변환
        xyz_w, quat_w = base_to_world(T_wb, xyz_b, quat_b)
        place_xyz_w   = [xyz_w[0], xyz_w[1], xyz_w[2] - place_z_descent]
        place_xyz_b, place_quat_b = world_to_base(T_wb, place_xyz_w, quat_w)
        node.get_logger().info(
            f'[step_place_from_home] HOME (world) Z={xyz_w[2]:.3f}m  '
            f'descent={place_z_descent:.3f}m  place Z={place_xyz_w[2]:.3f}m')
    else:
        # T_world_base 없으면 base Z로 하강 (fallback)
        node.get_logger().warning('[step_place_from_home] T_world_base 없음 → base Z로 하강')
        place_xyz_b  = [xyz_b[0], xyz_b[1], xyz_b[2] - place_z_descent]
        place_quat_b = quat_b

    place_target = node._make_pose(*place_xyz_b, *place_quat_b)

    # HOME_JOINT_VALUES를 seed로 고정 — _current_joints 타이밍 오차로
    # Cartesian 시작점이 틀어지는 것을 방지
    descent_jt = _plan(node, place_target, 'PLACE_DESCENT',
                       seed=list(HOME_JOINT_VALUES), confirm=False)
    if descent_jt is None:
        node.get_logger().error('[step_place_from_home] PLACE_DESCENT 실패')
        return False
    _exec(node, descent_jt)

    # 릴리즈
    step_release_hand(node, confirm=False)

    # 상승: 하강 역재생 → HOME 복귀 (planning 없음)
    node.get_logger().info('[PLACE_ASCENT] 하강 역재생으로 HOME 복귀 (IK 재계산 없음)')
    _exec(node, reverse_trajectory(descent_jt))

    return True
