#!/usr/bin/env python3
"""_grip_with_retry 검증 — 로봇 없이. 시도 횟수·손 펴기 타이밍·경계값을 확인한다.

실행:  cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
       python3 tools/check_grip_retry.py        # 종료코드 0 = 전부 통과

_grip / _peak_forces / D.move_hand_to 를 스텁으로 갈아끼워 호출 순서를 기록한다.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/inference-physics-property/stiffness_deploy_ros2")
sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/inference-physics-property/stiffness_deploy_ros2/launch")
import collect_ros2 as C                                            # noqa: E402

THR = 7.0
log = []


def run(name, peaks, *, min_fingers=C.GRIP_MIN_FINGERS, max_retry=C.GRIP_MAX_RETRY,
        judge_force=None):
    """peaks = 시도별 finger 힘 리스트(마지막 값이 그 뒤로도 반복).

    judge_force=None → 판정 힘 = 목표 힘(THR). 아래 경계 케이스들이 THR 기준으로
    만들어져 있으므로 기본값을 None 으로 둬서 '재시도·개수 판정' 로직만 시험한다.
    판정 힘 분리 자체는 맨 아래 별도 케이스에서 확인한다.
    """
    log.clear()
    seq = list(peaks)

    def fake_grip(bridge, paxini):
        log.append("grip")
        return [0.0] * 16

    def fake_peak(paxini, settle_sec=0.0, **kw):
        # 실제 _peak_forces 는 (peak, still_rising) 튜플을 준다(조기탈출·상승 힌트용).
        v = seq[min(log.count("grip") - 1, len(seq) - 1)]
        return np.asarray(v, np.float32), False

    def fake_open(bridge, pos, dur):
        log.append("hand_open")

    C._grip, C._peak_forces, C.D.move_hand_to = fake_grip, fake_peak, fake_open
    pos, ok, peak, n = C._grip_with_retry(None, None, threshold=THR,
                                          judge_force=judge_force,
                                          min_fingers=min_fingers, max_retry=max_retry)
    print(f"\n### {name}")
    print(f"    호출 순서   : {log}")
    print(f"    grip {log.count('grip')}회 · hand_open {log.count('hand_open')}회")
    print(f"    반환        : ok={ok} n_reached={n} peak={np.round(peak, 1).tolist()}")
    return log.count("grip"), log.count("hand_open"), ok, n


print(f"[설정] GRIP_MIN_FINGERS={C.GRIP_MIN_FINGERS} GRIP_MAX_RETRY={C.GRIP_MAX_RETRY} "
      f"임계={THR}N → 최대 시도 {C.GRIP_MAX_RETRY + 1}회")

M = C.GRIP_MIN_FINGERS          # 기대값을 상수에서 끌어온다(상수를 바꿔도 테스트가 따라옴)
OK3 = [11.4, 7.5, 16.5, 2.2]    # 3개 도달 (실측 세션 160837 = grip_reached_fingers 3)
OK2 = [11.4, 0.0, 16.5, 2.2]    # 2개만 도달 (§F7 실측 분포 — 접촉 없는 손가락이 늘 있다)
NG0 = [1.0, 0.5, 2.0, 0.0]      # 0개 도달

checks = []
g, o, ok, n = run("1) 1회차 성공 (실측 세션 160837 = 3개 도달)", [OK3])
checks.append(("1회차 성공 → grip 1 / hand_open 0 / ok", (g, o, ok, n) == (1, 0, True, 3)))

g, o, ok, n = run("2) 2회차에 성공 (재시도가 실제로 일어나나)", [NG0, OK3])
checks.append(("2회차 성공 → grip 2 / hand_open 1 / ok", (g, o, ok) == (2, 1, True)))

g, o, ok, n = run("3) 3회차(마지막)에 성공", [NG0, NG0, OK3])
checks.append(("3회차 성공 → grip 3 / hand_open 2 / ok", (g, o, ok) == (3, 2, True)))

g, o, ok, n = run("4) 전부 실패 (재시도 소진)", [NG0])
checks.append(("전부 실패 → grip 3 / hand_open 2 / not ok", (g, o, ok) == (3, 2, False)))
checks.append(("마지막 시도 뒤엔 손 펴지 않음(중단 경로가 처리)", log[-1] == "grip"))

g, o, ok, n = run(f"5) 경계: 정확히 {M}개가 임계와 같은 값", [[THR] * M + [0.0] * (4 - M)])
checks.append((f"임계 '이상' {M}개 → 성공(n={M})", (ok, n) == (True, M)))

g, o, ok, n = run(f"6) 경계: {M - 1}개만 도달", [[THR] * (M - 1) + [THR - 0.01] + [0.0] * (4 - M)])
checks.append((f"{M - 1}개만 → 실패", (ok, n) == (False, M - 1)))

g, o, ok, n = run(f"7) 2개만 도달 (현 min_fingers={M} 기준)", [OK2])
checks.append((f"접촉 2개는 min_fingers={M} 에서 " + ("성공" if M <= 2 else "실패"),
               ok is (M <= 2)))

g, o, ok, n = run("8) min_fingers=4 (전 손가락 요구 — 참고)", [OK3], min_fingers=4)
checks.append(("접촉 없는 손가락 있으면 4개 요구는 실패", ok is False))

# ── 9) 판정 힘 = 뽑힌 파지 임계(기본) vs 고정값(옵션) ──────────────────────────
#   현재 정책(사용자 지시, 2026-07-28): GRIP_JUDGE_FORCE_N=None → **범위에서 뽑힌 목표 임계로
#   판정**한다. 새 설계가 grasp 를 기준으로 스퀴즈 임계(grasp+delta)와 해제 검증(grasp×0.8)까지
#   만들기 때문에 판정 기준을 하나로 통일한 것이다.
#   대가: 목표가 범위 위쪽으로 뽑히면 실패할 수 있다(실측 2번째 손가락 6.4~7.7N).
#   그건 파지 재시도 → run grip_fail → **자세 조합 재수집** 으로 흡수되므로 데이터 구멍은 없다.
#   judge_force 를 명시하면 분리도 여전히 가능하다(아래 9-b) — 되돌릴 때 쓸 수 있게 유지.
REAL = [[7.7, 8.3, 6.6, 0.0], [6.4, 8.0, 6.2, 0.0], [6.6, 8.5, 5.8, 0.0]]
by_target, fixed5 = {}, {}
for tgt in (6.0, 7.0, 7.92, 8.0):
    THR = tgt
    by_target[tgt] = run(f"9-a) 목표={tgt}N · 판정=목표 그대로(기본)",
                         REAL, judge_force=None)[2]
    fixed5[tgt] = run(f"9-b) 목표={tgt}N · 판정=5.0N 고정(옵션)",
                      REAL, judge_force=5.0)[2]
checks.append(("기본값이 '목표 임계로 판정'(GRIP_JUDGE_FORCE_N=None)",
               C.GRIP_JUDGE_FORCE_N is None))
checks.append((f"기본: 목표가 높으면 실패할 수 있다(의도된 대가) — {by_target}",
               by_target[6.0] and not by_target[8.0]))
checks.append((f"옵션: judge_force 명시 시 목표와 무관하게 판정 — {fixed5}",
               all(fixed5.values())))

print("\n" + "=" * 72)
bad = [d for d, k in checks if not k]
for d, k in checks:
    print(f"  {'✔' if k else '✘'} {d}")
print(f"\n{'전부 통과' if not bad else f'실패 {len(bad)}건'}")
sys.exit(1 if bad else 0)
