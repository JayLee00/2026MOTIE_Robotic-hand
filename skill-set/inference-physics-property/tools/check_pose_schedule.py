#!/usr/bin/env python3
"""자세 스케줄 + 개체 이름 + 폴더명 검증 — 로봇 없이.

실행:  cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
       python3 tools/check_pose_schedule.py

확인 항목:
  1. palm_up/palm_down 이 각각 '기본 + LR택1 + FB택1' = 3개인가
  2. 조합 = 3×3×5 = 45 이고 **중복 없이 전부** 들어 있나
  3. 실패 판정(grip_fail/discard)이면 같은 조합이 재투입되어 결국 45개 전부 성공하나
  4. 개체 이름 입력 형태 3종이 올바르게 해석되나
  5. 폴더명 '<개체>_<파지자세>_<ts>' 가 만들어지고, 되읽을 때 개체 이름이 복원되나
"""
import collections
import itertools
import sys

sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/inference-physics-property/stiffness_deploy_ros2")
sys.path.insert(0, "/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/inference-physics-property/stiffness_deploy_ros2/launch")
import collect_ros2 as C                                              # noqa: E402
import bag_to_session as BS                                           # noqa: E402

print("=== 1) 제시 자세 세트 (기본 + LR택1 + FB택1) ===")
for _ in range(3):
    up = C._pick_present_set(C.PALM_UP_BASE, C.PALM_UP_PAIRS, "palm-up")
    dn = C._pick_present_set(C.PALM_DOWN_BASE, C.PALM_DOWN_PAIRS, "palm-down")
    assert len(up) == 3 and up[0] == "palm_up", up
    assert len(dn) == 3 and dn[0] == "palm_down", dn
    assert ("palm_up_tilt_left" in up) ^ ("palm_up_tilt_right" in up), up
    assert ("palm_up_tilt_fwd" in up) ^ ("palm_up_tilt_back" in up), up
print("   ✔ 3회 반복 모두 3개 · 기본 포함 · LR 배타 · FB 배타")

print("\n=== 2) 조합 커버리지 ===")
up, dn, grips, combos = C._build_schedule(C.GRIP_POSE_CANDIDATES)
expect = set(itertools.product(up, dn, grips))
assert len(combos) == 45, len(combos)
assert set(combos) == expect, "조합 집합 불일치"
assert len(set(combos)) == len(combos), "중복 조합 있음"
print(f"   ✔ {len(combos)}개 = {len(up)}×{len(dn)}×{len(grips)} · 중복 0 · 빠짐 0")
cnt = collections.Counter(g for _u, _d, g in combos)
print(f"   ✔ 파지 자세별 등장 횟수 = {dict(cnt)}  (각 9회 = 3×3)")
assert set(cnt.values()) == {9}, cnt

print("\n=== 3) 실패 재투입 → 45개 전부 성공까지 (큐 시뮬레이션) ===")
# main() 의 큐 로직과 동일한 규칙을 재현: 실패면 append, 성공이면 done += 1
FAIL_UNTIL = 2          # 각 조합을 2번 실패시킨 뒤 성공하게 만든다
tries = collections.Counter()
queue, done, runs = list(combos), 0, 0
while queue:
    combo = queue.pop(0)
    runs += 1
    tries[combo] += 1
    outcome = "success" if tries[combo] > FAIL_UNTIL else "grip_fail"
    if outcome in C.RETRY_OUTCOMES:
        queue.append(combo)
    else:
        done += 1
print(f"   조합 {len(combos)}개 · 각 {FAIL_UNTIL}회 실패 후 성공 → run {runs}회, 완료 {done}개")
assert done == len(combos), (done, len(combos))
assert runs == len(combos) * (FAIL_UNTIL + 1), runs
assert all(v == FAIL_UNTIL + 1 for v in tries.values()), "조합별 시도 횟수 불균등"
print(f"   ✔ 모든 조합이 정확히 {FAIL_UNTIL + 1}회 시도되고 45개 전부 완료")

print("\n=== 4) 개체 이름 해석 ===")
for given, expected in (("1", "ecoflex_1"), ("3", "ecoflex_3"),
                        ("ecoflex_1", "ecoflex_1"), ("ecoflex 2", "ecoflex_2"),
                        ("", "ecoflex")):
    got = C._ask_specimen("ecoflex", given if given else None) if given else "ecoflex"
    if given:
        print(f"   --specimen {given!r:12s} → {got!r}")
        assert got == expected, (got, expected)
print("   ✔ 숫자·전체이름·공백포함 모두 정상 (빈 값은 대화형 프롬프트 경로)")

print("\n=== 5) 폴더명에 파지 자세 넣기 + 되읽기 ===")
#   폴더명은 '<개체>_<파지자세>_<YYYYmmdd>_<HHMMSS>'. 파지 자세가 여러 개인 세션은
#   '-' 로 잇고(≤3개), 그보다 많으면 개수로 줄인다('5pose'). 실제 자세는 run 단위로
#   session.h5 /runs_names 에 남으므로 폴더명은 한눈 라벨이면 된다.
TS = "20260729_000753"
CASES = (
    # (개체, 쓴 pose txt 목록, 기대 태그, 폴더명 되읽었을 때의 개체 이름)
    ("ecoflex_1", ["ecoflex.txt"], "ecoflex", "ecoflex_1"),
    ("ecoflex_12", ["ecoflex.txt"], "ecoflex", "ecoflex_12"),
    ("tomato_3", ["tomato.txt", "plum.txt"], "tomato-plum", "tomato_3"),
    ("kiwi_1", list(C.GRIP_POSE_CANDIDATES), "5pose", "kiwi_1"),
    # stem 에 '_' 가 있어도 태그는 1토큰이어야 한다(안 그러면 되읽기가 밀린다)
    ("ecoflex_2", ["my_soft_v2.txt"], "my-soft-v2", "ecoflex_2_my-soft-v2"),
)
for spec, grips, want_tag, want_spec in CASES:
    tag = C._pose_tag(grips)
    folder = f"{spec}_{tag}_{TS}"
    fruit, got = BS._guess_names(folder)
    print(f"   {spec:11s} + {len(grips)}자세 → {folder:42s} → 개체 {got!r} 물체 {fruit!r}")
    assert tag == want_tag, (tag, want_tag)
    assert "_" not in tag, f"태그가 1토큰이 아니다: {tag}"
    assert got == want_spec, (got, want_spec)
print("   ✔ 태그 생성 + 개체 이름 복원 정상")
print("   ℹ 마지막 줄: pose 디렉터리에 없는 txt(my_soft_v2)는 태그로 못 알아본다 —"
      " 폴더명 추정은 outcomes.json 이 없을 때의 폴백이고, 실제 세션은 그 파일의"
      " specimen 이 우선이라 영향 없다.")

#   구 폴더명(자세 토큰 없음)도 그대로 읽혀야 한다 — 이미 모은 세션을 못 읽으면 안 된다.
OLD = (("ecoflex_1_20260728_231155", "ecoflex_1"),
       ("collect_ecoflex_20260728_224840", "ecoflex"),
       ("ecoflex_20260728_224840", "ecoflex"))     # 개체명 == 자세명 → 떼면 안 된다
for folder, want in OLD:
    _f, got = BS._guess_names(folder)
    print(f"   (구) {folder:34s} → 개체 {got!r}")
    assert got == want, (folder, got, want)
print("   ✔ 자세 토큰 없는 옛 폴더명도 그대로 해석 (하위호환)")

print("\n전부 통과")
