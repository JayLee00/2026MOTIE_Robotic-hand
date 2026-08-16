#!/usr/bin/env python3
"""P2(kin 소스 고정) 검증 — 로봇/SHM 없이. docs/todolist.md 2번 과제의 완료 판정 재현.

실행:  cd ~/motie_ws/stiffness_deploy_ros2 && source env.sh
       python3 tools/check_jkin_source_pin.py

read_live_sample 에 스텁(SHM raw=111 / mN 파일=999)을 넣어 가드뿐 아니라
실제 ft 출처까지 확인한다. 6케이스 전부 표의 판정과 같아야 한다."""
import os, sys, time, subprocess
from pathlib import Path

PKG = Path("/home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/inference-physics-property/stiffness_deploy_ros2")
F = Path("/tmp/deep_ws_raw_06_hand_j_kin_mN.txt")

CHILD = r'''
import sys, numpy as np
sys.path.insert(0, "%s"); sys.path.insert(0, "%s/launch")
import real_deploy_inference_final as RE

# --- 가드 ---
try:
    RE.assert_jkin_source_pinned()
    print("GUARD=PASS")
except SystemExit as e:
    print("GUARD=REFUSED"); print("   msg:", str(e).splitlines()[0])
    sys.exit(0)

# --- 실제 데이터 경로: SHM raw 는 111, mN 파일은 999 로 구분 ---
from core.shm_common import Hand_DOF, Kinesthetic_Sensor_Num as KN, Kinesthetic_Sensor_DOF as KD
class Msg:
    j_pos = [[0.0] * Hand_DOF]
    j_kin = [[[111.0] * KD for _ in range(KN)]]
class Shm:
    def read(self): return Msg()
class Pax:
    def read(self): return np.zeros((4, 32, 3), np.float32), 0.0, 1, 0
s = RE.read_live_sample(Shm(), Pax())
src = {111.0: "SHM_raw", 999.0: "mN_file"}.get(float(s["ft"][0]), f"?({s['ft'][0]})")
print(f"FT_SOURCE={src}")

# --- (b) provenance 대조: 체크포인트 라벨 ↔ 배포 소스 ---
for label, meta in (("라벨없음(기존 ckpt)", {}),
                    ("jkin_source=SHM_raw", {"jkin_source": "SHM_raw"}),
                    ("jkin_source=mN", {"jkin_source": "mN_side_channel"}),
                    ("mN_present=0", {"raw_hand_j_kin_mN_present": 0}),
                    ("mN_present=1", {"raw_hand_j_kin_mN_present": 1})):
    try:
        RE.assert_jkin_source_matches_ckpt(meta)
        r = "PASS"
    except SystemExit:
        r = "REFUSED"
    print(f"PROV\t{label}\t{r}")
''' % (PKG, PKG)


def run(label, *, switch, file_state):
    F.unlink(missing_ok=True)
    if file_state == "fresh":
        F.write_text(" ".join(["999"] * 12))
    elif file_state == "stale":
        F.write_text(" ".join(["999"] * 12))
        os.utime(F, (time.time() - 5, time.time() - 5))
    elif file_state == "malformed":
        F.write_text("1 2 3")
    env = dict(os.environ, USE_MN_SIDE_CHANNEL=switch)
    print(f"\n### {label}")
    print(f"    (USE_MN_SIDE_CHANNEL={switch}, mN 파일={file_state})")
    r = subprocess.run([sys.executable, "-c", CHILD], env=env,
                       capture_output=True, text=True)
    for ln in (r.stdout + r.stderr).strip().splitlines():
        print("   ", ln)


run("1) OFF + 파일 없음  (수집 당시와 동일 → 통과·SHM raw)", switch="0", file_state="none")
run("2) OFF + 파일 존재  (조용한 스케일 변경 차단 → 경고·SHM raw 고정)", switch="0", file_state="fresh")
run("3) ON  + 파일 없음  (학습과 다른 스케일 폴백 금지 → 실행 거부)", switch="1", file_state="none")
run("4) ON  + 파일 만료  (5초 전 → 만료 취급 → 실행 거부)", switch="1", file_state="stale")
run("5) ON  + 파일 형식오류 (12개 아님 → 거부)", switch="1", file_state="malformed")
run("6) ON  + 파일 정상  (수집이 mN 였던 경우 → 통과·mN 사용)", switch="1", file_state="fresh")

F.unlink(missing_ok=True)
print(f"\n[정리] {F} 삭제 = {not F.exists()}")
