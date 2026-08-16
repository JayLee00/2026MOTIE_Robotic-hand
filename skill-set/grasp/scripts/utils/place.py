#!/usr/bin/env python3
"""
PlaceExecutor — Pick → Home → Place → Release → Home.

STEP 1-4 : Grasp (y/n 확인)
STEP 5-9 : Home → Place → Release → Home (자동)

place 위치 = HOME EE XY 그대로, Z 만 place_z_descent 만큼 하강.
"""

import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from utils.grasp import GraspExecutor, world_to_base
from utils.step import (
    step_approach, step_descend, step_lift,
    step_init_hand, step_close_hand, step_release_hand, step_go_home,
    step_place_from_home,
)


class PlaceExecutor(GraspExecutor):
    """
    STEP 1-4 : Grasp (y/n 확인)
    STEP 5-9 : Home → Place → Release → Home (자동)
    """

    def __init__(self, summary, execute_mode, speed_factor,
                 approach_offset, place_z_descent, summary_json_path='',
                 disable_collision: bool = False):
        self._place_z_descent = place_z_descent
        super().__init__(summary, execute_mode, speed_factor,
                         approach_offset, summary_json_path=summary_json_path,
                         disable_collision=disable_collision)

    def _execute(self):
        grasp  = self._summary['grasps'][0]
        bp     = grasp['base_pose']
        xyz_w  = bp['xyz']
        quat_w = bp['quat_xyzw']
        enc    = grasp['joint_angles_enc']
        T_wb   = self._summary.get('T_world_base')

        if T_wb is not None:
            xyz_b, quat_b = world_to_base(T_wb, xyz_w, quat_w)
            xyz_b_app, quat_b_app = world_to_base(
                T_wb, [xyz_w[0], xyz_w[1], xyz_w[2] + self._approach_offset], quat_w)
        else:
            self.get_logger().warning('T_world_base 없음 — base frame 그대로 사용')
            xyz_b, quat_b = xyz_w, quat_w
            xyz_b_app, quat_b_app = (
                [xyz_w[0], xyz_w[1], xyz_w[2] + self._approach_offset], quat_w)

        print('\n' + '=' * 60)
        print('  Pick-Place-via-Home Executor  (STEP 1~9)')
        print(f'  Grasp (base): {[round(v,3) for v in xyz_b]}')
        print(f'  Place Z     : {self._place_z_descent:.3f} m  (HOME EE Z 기준 하강)')
        print('=' * 60)

        time.sleep(1.0)
        step_init_hand(self)
        step_go_home(self, confirm=True)
        target   = self._make_pose(*xyz_b,     *quat_b)
        approach = self._make_pose(*xyz_b_app, *quat_b_app)

        # ── STEP 1-4: Grasp (y/n) ────────────────────────────────────────
        result = step_approach(self, approach, confirm=True)
        if result is None: return
        j1, approach_traj = result
        self._approach_traj = approach_traj

        result = step_descend(self, target, seed=j1, confirm=True)
        if result is None: return
        j2, descend_traj = result

        print(f'\n  STEP 3/9  HAND 파지  enc={enc}')
        if not step_close_hand(self, enc, confirm=True): return

        j4 = step_lift(self, approach, seed=j2,
                       descend_traj=descend_traj, confirm=True)
        if j4 is None: return

        # ── STEP 5-9: 자동 세트 ───────────────────────────────────────────
        print('\n' + '─' * 60)
        print('  [AUTO] Home → Place → Release → Home')
        print('─' * 60)

        step_go_home(self, confirm=False, approach_traj=approach_traj)

        ok = step_place_from_home(self, place_z_descent=self._place_z_descent)
        if not ok:
            self.get_logger().error('Place 실패')
            return
        step_init_hand(self)

        self._success = True
        print('\n  [OK] Pick-Place-via-Home 완료')

    def _finalize(self):
        print('\n' + '─' * 60)
        print('  [FINALIZE]')
        print('─' * 60)
        if self._success:
            print('  정상 완료.')
        else:
            print('  ⚠  실행이 중단되었습니다.')
            step_release_hand(self, confirm=True)
            step_go_home(self, confirm=True)
