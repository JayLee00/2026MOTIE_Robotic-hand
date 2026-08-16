#!/usr/bin/env python3
"""파지·스퀴즈 임계 결합 + 스퀴즈 후 엄지 복귀 검증 — 로봇 없이.

실행:  cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
       python3 tools/check_force_thresholds.py        # 종료코드 0 = 전부 통과

확인 항목:
  1. grasp = uniform(6,8) / squeeze = grasp + uniform(3,5) → 9~13N, delta 가 실제로 그 범위
  2. 엄지 복귀: **엄지 관절만** 파지 위치로 명령하고 나머지는 현재값 유지(파지 안 흐트러짐)
  3. 힘이 남아 있으면(약해져도) 폐기하지 않는다
  4. 힘이 사실상 0 이면 놓침 → grip_lost (조합은 소진되지 않고 재수집)
"""
import sys

import numpy as np

sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/inference-physics-property/stiffness_deploy_ros2")
sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/inference-physics-property/stiffness_deploy_ros2/launch")
import collect_ros2 as C                                              # noqa: E402

checks = []

print(f"[설정] GRIP_FORCE_RANGE={C.GRIP_FORCE_RANGE} "
      f"SQUEEZE_DELTA_RANGE={C.SQUEEZE_DELTA_RANGE} "
      f"SQUEEZE_FORCE_RANGE={C.SQUEEZE_FORCE_RANGE}")
print(f"       THUMB_RETURN_AFTER_SQUEEZE={C.THUMB_RETURN_AFTER_SQUEEZE} "
      f"THUMB_RETURN_DURATION={C.THUMB_RETURN_DURATION}s "
      f"RELEASE_SETTLE_SEC={C.RELEASE_SETTLE_SEC} "
      f"GRIP_LOST_FORCE_N={C.GRIP_LOST_FORCE_N}N")

print("\n=== 1) 임계 결합 (grasp + delta) ===")
gs, ss, ds = [], [], []
for _ in range(2000):
    g = C._draw(C.GRIP_FORCE_RANGE, C.D.GRIP_FORCE_THRESHOLD)
    s, d = C._squeeze_threshold(g)
    gs.append(g); ss.append(s); ds.append(d)
    assert abs(s - (g + d)) < 1e-6, (s, g, d)
gs, ss, ds = np.array(gs), np.array(ss), np.array(ds)
print(f"   grasp   {gs.min():.2f} ~ {gs.max():.2f} N   (기대 6.0~8.0)")
print(f"   delta   {ds.min():.2f} ~ {ds.max():.2f} N   (기대 3.0~5.0)")
print(f"   squeeze {ss.min():.2f} ~ {ss.max():.2f} N   (기대 9.0~13.0)")
checks.append(("grasp 이 6~8N 범위", 6.0 <= gs.min() and gs.max() <= 8.0))
checks.append(("delta 가 3~5N 범위", 3.0 <= ds.min() and ds.max() <= 5.0))
checks.append(("squeeze = grasp+delta 이고 9~13N", 9.0 <= ss.min() and ss.max() <= 13.0))
checks.append(("squeeze 가 항상 grasp 보다 크다", bool((ss > gs).all())))

# ── 엄지 복귀 검증용 스텁 ──────────────────────────────────────────────────
#   힘 기준 재조임은 **실측에서 물체를 놓쳐 폐기**했다(재조임 2회차에 [0,0,0,0]N).
#   지금은 엄지 관절만 파지 위치로 되돌리고, 힘은 관측·기록만 한다.
def run_thumb_return(force, grip_pos=None, cur=None):
    """반환 (held, n_hold, 명령된 target, move 호출 여부)."""
    GRIP = grip_pos or [100 + i for i in range(16)]      # 파지 위치(엄지 0~3 포함)
    CUR = cur or [900 + i for i in range(16)]            # 현재 위치(전 관절 다름)
    sent = {}

    class Msg:
        j_pos = [CUR]

    class Bridge:
        def read(self): return Msg()

    C.D.move_hand_to = lambda b, pos, dur: sent.update(target=list(pos), dur=dur)
    C._peak_forces = lambda p, s=0.0, **kw: (
        np.asarray([force] * 2 + [0.0, 0.0], np.float32), False)
    held, peak, n_hold = C._thumb_return(Bridge(), None, GRIP)
    return held, n_hold, sent.get("target"), sent.get("dur")


print("\n=== 2) 엄지 관절만 파지 위치로, 나머지는 현재값 유지 ===")
held, n, tgt, dur = run_thumb_return(8.0)
thumb = [j for f in C.D.SQUEEZE_FORCE_FINGERS
         for j in range(f * C.D.JOINTS_PER_FINGER, (f + 1) * C.D.JOINTS_PER_FINGER)]
others = [j for j in range(16) if j not in thumb]
print(f"   엄지 관절 {thumb} → 명령값 {[tgt[j] for j in thumb]} (파지 위치 = 100~103)")
print(f"   그 외 관절 첫 4개 {others[:4]} → 명령값 {[tgt[j] for j in others[:4]]} (현재값 = 904~)")
print(f"   이동 시간 = {dur}s  ·  held={held} 접촉 {n}개")
checks.append(("엄지 관절만 파지 위치로 명령",
               all(tgt[j] == 100 + j for j in thumb)))
checks.append(("나머지 관절은 현재값 그대로(파지 흐트러뜨리지 않음)",
               all(tgt[j] == 900 + j for j in others)))
checks.append(("이동 시간이 THUMB_RETURN_DURATION", dur == C.THUMB_RETURN_DURATION))
checks.append(("힘 기준 재조임 함수를 더 쓰지 않는다",
               not hasattr(C, "_restore_grip_force")))

print("\n=== 3) 힘이 남아 있으면 held=True (약해져도 폐기하지 않음) ===")
for f in (8.0, 3.0, 1.0):
    held, n, _t, _d = run_thumb_return(f)
    print(f"   접촉력 {f:4.1f}N → held={held} (기준 {C.GRIP_LOST_FORCE_N:g}N)")
    checks.append((f"{f:g}N 이면 유지로 본다", held is True))

print("\n=== 4) 힘이 사실상 0 → 놓침 감지 ===")
held, n, _t, _d = run_thumb_return(0.0)
print(f"   접촉력 0.0N → held={held} 접촉 {n}개")
checks.append(("전 손가락 < GRIP_LOST_FORCE_N 이면 놓침 판정", held is False))
checks.append(("grip_lost 가 재수집 대상", "grip_lost" in C.RETRY_OUTCOMES))

# ── 5) 놓침을 '자동 확정' 하지 않는다 (사용자 지시 2026-07-29) ────────────────────────
#   문제였던 동작: 엄지 복귀 B 에서 힘≈0 → 코드가 grip_lost 로 확정 → 판정 프롬프트를
#   건너뛰고 다음 run 이 바로 시작. 시퀀스가 끝까지 돌아 스퀴즈 A·B 가 **둘 다 기록된**
#   run 을 사람 확인 없이 폐기한 셈이다.
#   지금: B 는 base["suggest_outcome"] 로 '제안' 만 하고, 확정은 _ask_outcome 이 받는다.
#   A 는 그대로 자동 — 스퀴즈 B 가 아예 없어 물어봐도 재수집 말고 선택지가 없다.
print("\n=== 5) 엄지 복귀 B 놓침 → 자동 확정 금지, 사람 판정 필수 ===")
import inspect                                                       # noqa: E402
src = inspect.getsource(C._run_sequence)
AUTO = 'return names, "grip_lost", base'
n_auto = src.count(AUTO)
b_suggest = 'base["suggest_outcome"] = "grip_lost"' in src
print(f"   A(스퀴즈 B 생략) 자동 grip_lost 반환 = {n_auto > 0} · 자동 반환 {n_auto}곳"
      " (A 경로 1곳만이어야 함)")
print(f"   B(A·B 다 기록됨) 제안만          = {b_suggest}")
checks.append(("A 경로는 자동 grip_lost (물어볼 것이 없다)", n_auto > 0))
checks.append(("A 경로 자동 반환이 1곳뿐(B 는 자동 아님)", n_auto == 1))
checks.append(("B 경로는 제안만 남긴다", b_suggest))
checks.append(("grip_lost 를 사람이 직접 고를 수도 있다",
               "grip_lost" in C.OUTCOMES.values()))

typed_cases = [("", "grip_lost"), ("1", "success"), ("4", "discard"), ("5", "grip_lost")]
_orig_prompt = C._prompt
for typed, want in typed_cases:
    C._prompt = lambda _text, _t=typed: _t
    import io                                                        # noqa: E402
    buf, so = io.StringIO(), sys.stdout
    sys.stdout = buf
    got, _stop = C._ask_outcome(5, remaining=4, suggest="grip_lost", why="테스트")
    sys.stdout = so
    shown = "코드 의심: grip_lost" in buf.getvalue()
    print(f"   입력 {typed!r:5s} → {got:10s} (제안 표시 {shown})")
    checks.append((f"입력 {typed!r} → {want}", got == want and shown))
C._prompt = lambda _text: ""
checks.append(("제안 없으면 Enter 기본값은 그대로 success",
               C._ask_outcome(1)[0] == "success"))
C._prompt = _orig_prompt

print("\n" + "=" * 72)
bad = [d for d, k in checks if not k]
for d, k in checks:
    print(f"  {'✔' if k else '✘'} {d}")
print(f"\n{'전부 통과' if not bad else f'실패 {len(bad)}건'}")
sys.exit(1 if bad else 0)
