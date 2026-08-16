#!/usr/bin/env python3
"""
시나리오1용 GraspExecutor — Pick 후 물체를 놓지 않고 그 자리에서 유지 (Inhand 대기).

원본 utils/grasp.py 의 GraspExecutor 를 상속해 `_finalize` 만 override 한다:
  · 원본 _finalize: step_release_hand(손 열기) + step_go_home → 물체 놓고 홈 복귀
  · 시나리오1    : release / go_home 둘 다 스킵 → lift 위치에서 물체 쥔 채 정지

원본(grasp.py)은 건드리지 않는다. Pick(1) 종료 후 Inhand(2)가 물체를 이어받는다.
"""
from utils.grasp import GraspExecutor


class GraspScenario1Executor(GraspExecutor):
    def _finalize(self):
        print('\n' + '─' * 60)
        print('  [FINALIZE] 시나리오1 — 물체 쥔 채 그 자리 유지')
        print('  (release / go_home 스킵 → Inhand(2) 이어받기 대기)')
        print('─' * 60)
        # 손 안 열고(release X), home 도 안 감(go_home X).
        # robot_executor 의 _hold_hand_position(2s)이 쥔 자세를 유지해 줌.
