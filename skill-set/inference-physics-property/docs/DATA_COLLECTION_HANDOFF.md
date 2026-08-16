# 데이터 수집 ROS2 전환 — 실행 인수인계 (HANDOFF)

> 이 문서는 **다른 컴퓨터/새 세션에서 이어서 작업**하기 위한 자기완결 인수인계다.
> 전체 설계 근거는 **같은 폴더의 `docs/ros2_data_collection_plan.md`(정본)** 에 있고, 이 문서는 **실행에 필요한 것만** 추린 요약이다. 둘 다 이 레포 안에 있어 **Gen3 없이 자기완결** — 이 레포만 pull 하면 된다.
> 최종 갱신: 2026-07-27 · 상태: 수집 파이프라인 + 9단계 시퀀스 + 부가기능(팔 이동·구간 flag·힘 랜덤화·데모 판정) **코드 구현 완료**. 로봇 실측만 남음.

---

## 0. 한 줄 목표

딥러닝 학습 데이터를 **"배포 시 모델이 실제로 보게 될 신호를, 배포와 동일한 파이프라인·rate로"** 수집하도록, **수집 코드를 배포 코드(`deploy.py` + ROS2 브리지)와 하나로** 만든다. train/deploy 불일치(힘 저평가·시퀀스 OOD, `docs/TROUBLESHOOTING.md` F1·B3·F절)를 근본에서 없앤다.

---

## 1. 확정된 결정 (why는 `docs/ros2_data_collection_plan.md` 참조)

- **q1 통합**: 런타임 코어는 **이 워크스페이스 하나**로. 오프라인 자산만 Gen3에.
- **q2a 모션**: SHM 직접 방식 폐기, **ROS2 브리지 재사용** → parity 구조적 보장.
- **q2b 로깅**: **rosbag2(mcap) raw + 버전 고정 결정론 변환기 → HDF5**. 재현성은 포맷이 아니라 버전관리된 전처리 변환에서 결정.
- **정렬/rate**: causal ZOH(보간 금지), 격자 rate ≤ 배포 loop rate ≤ 토픽 rate, 업샘플 금지(다운샘플만).
- **촉각 소스**: 모델이 resultant(4×3)+127점 tactile 둘 다 쓸 가능성 → 수집·배포 공통 **`/paxini/right/raw`**.

---

## 2. ★ 결정적 게이트 — 가장 먼저 확인 (P0의 `/raw`)

```bash
cd ~/stiffness_deploy_ros2 && source env.sh   # ROS_DOMAIN_ID=9, 시스템 python
ros2 topic list | grep -E 'paxini|hand|franka'      # /paxini/right/raw 존재?
ros2 topic hz  /paxini/right/raw                     # 있으면 rate
python3 stiffness_deploy_ros2/launch/deploy_ros2_exp_ftcheck.py   # ft vs Σ127 (H1/H2)
python3 stiffness_deploy_ros2/launch/deploy_ros2_exp.py          # 배포 loop rate·샘플수
```
- **`/raw` 보이면** → 127점 수집 가능. **안 보이면** → 제어 PC `shm_state_publisher`에 발행 요청(그전까지 `/ft` 4×3만).
- **런타임 전제**: 제어 PC 스택(C++ + `shm_state_publisher` + arm/hand receiver + `paxini_writer`)이 같은 `ROS_DOMAIN_ID=9`. 팔 이동 쓰면 **플래닝 PC `move_group`** 도 필요(§6).

---

## 3. 실행 계획 (단계·게이트)

| 단계 | 무엇을 | 상태 |
|---|---|---|
| **P0 진단** | 토픽 rate / `/raw` 유무 / ft vs Σ127 / loop rate / 모델입력계약 | 도구 존재, **로봇 실측 남음** |
| **P1 계약** | 촉각소스=`/raw`, target rate+정책, loop 설계 | 문서 |
| **P2 코드 통합** | 브리지/모션 1벌, `/raw` 브리지 | ✅ |
| **P3 collect_ros2** | 배포 동일 경로 자동 수집 루프 + 9단계 시퀀스 | ✅ 구현완료(§6) |
| **P4 rosbag+변환기** | raw 기록 + 구간별 결정론 변환기 | ✅ 구현완료(§6) |
| **P5 parity 검증** | 라이브 HDF5 ↔ bag HDF5 대조 | ✅ 구현완료(§6, attr매칭 소폭 남음) |
| **P6 학습/분석** | 실데이터 수집 후 학습 | ⚠️ Gen3(오프라인) |

---

## 4. 재사용 자산 (있음)

- ROS2 브리지: `launch/deploy_ros2.py` — `Ros2ShmBridge`·`Ros2PaxiniBridge` (+ `_grip` 헬퍼).
- `/raw` 구독 브리지: `launch/deploy_ros2_exp_rawft.py` (`Ros2RawPaxiniBridge`).
- 모션 프리미티브: `launch/deploy.py` (`move_hand_to`, `..._until_force`, `move_hand_to_squeeze`).
- 진단: `deploy_ros2_exp.py`(rate/샘플)·`_ftcheck.py`(ft vs Σ127)·`_forcecurl.py`.
- 추론 엔진(입력 계약): `launch/real_deploy_inference_final.py` (`read_live_sample`·USE_TACTILE·USE_JKIN·FACTOR).
- core: `core/shm_common.py`·`core/paxini_shm.py`.
- 수집·변환·검증·팔이동 코드는 전부 **구현 완료 → §6**.

---

## 5. 열린 결정 / 외부 협의

1. **`/paxini/right/raw` 발행** — 제어 PC `shm_state_publisher` 담당자와 협의.
2. **배포 loop rate 고정 / 센서콜백 전환** — target rate 결정(`TROUBLESHOOTING.md` F절).
3. **팔 이동 게이트** — 플래닝 PC `move_group`(dex_ros) 기동 + `ARM_POSES` 실제 값 입력.

---

## 6. 구현 상태 (2026-07-27) — 전체 파이프라인 + 시퀀스/기능

**수집 → 변환 → 검증** 파이프라인 + 9단계 시퀀스 + 부가기능(팔 이동·구간 flag·힘 랜덤화·데모 판정) **코드 구현 완료. 로봇 실측만 남음.**

### 신규 파일 (`launch/`)
- `recording_engine.py` — `RecordingEngine`(배포 `add_sample`과 동일 규칙·단일 read, **모델 불필요**) + `HDF5DemoWriter`(구간별 그룹, 이름 규칙 `{segment}__run{NNN}`).
- `collect_ros2.py` — ★ 수집 엔트리포인트. 9단계 시퀀스 자동 N run + 동시 rosbag + 마커/구간/판정 토픽.
- `moveit_arm_mover.py` — **팔 이동 모듈**(dex_ros MoveIt `/move_action` **plan-only → q_target 재생**, 충돌회피). **Cartesian pose + joint 각도** 목표 지원. collect/deploy 미의존(arm_sink만 받음).
- `capture_pose.py` — **teach & capture**: 팔을 원하는 자세로 잡고 이름 입력 → 관절각 캡처 → `ARM_POSES` 형식으로 출력(복사·붙여넣기).
- `test_moveit_mover.py` — **MoveIt 팔 이동 테스트**: 기본 plan-only(안전), `--execute` 실제 이동, `--print-current` 현재 자세, `--joints`/`--pose` 목표.
- `bag_to_hdf5.py` — rosbag → **구간별** HDF5 변환기(Option 2a). 스퀴즈 A/B + **palm-down** 추출.
- `verify_parity.py` — P5: 라이브 HDF5 ↔ bag HDF5 데모·프레임 대조.
- `setup.py` 엔트리포인트: `collect_ros2`·`bag_to_hdf5`·`verify_parity`·`moveit_arm_mover`(+기존 deploy). `package.xml`: `moveit_msgs`·`shape_msgs`·`geometry_msgs`·`rosbag2_py`·`rosidl_runtime_py`. `requirements.txt`: `h5py`.

### 수집 시퀀스 (`collect_ros2`, run 당)
| 단계 | 담당 | 구간 flag(`/collect/segment`) |
|---|---|---|
| 안전위치 시작 | 팔 = MoveItArmMover | safe_start |
| 파지위치 이동 | 팔 = MoveItArmMover | move_grip |
| 물체 자동 파지 | 손 = `_grip` | grip |
| palm-up 이동 | 팔 = MoveItArmMover | move_palm_up |
| **스퀴즈A ★** | 손 = squeeze + HDF5 기록 | **squeeze_A** |
| **palm-down 이동 ★** | 팔 = MoveItArmMover | **move_palm_down** |
| **스퀴즈B ★** | 손 = squeeze + HDF5 기록 | **squeeze_B** |
| 물체 내려놓기 | 팔 = 파지위치 복귀 + 손 열기 | release |
| 안전위치 끝 | 팔 = MoveItArmMover | safe_end |

- **모든 팔 이동 = MoveItArmMover**(충돌회피). 손 파지/스퀴즈는 배포와 동일(`deploy(D)` 프리미티브).
- **구간 flag = `/collect/segment`(String) 라벨** — ★ 구간 각각 고유 라벨. rosbag에 담겨 **전처리 포함여부** 결정. 스퀴즈 ★는 HDF5 그룹 `segment` attr로도.

### 상단 고정 설정 (`collect_ros2.py` 편집)
- `ARM_POSES` — safe/grip/palm_up/palm_down. 형식 `{"joints":[j0..j6]}`(권장, `capture_pose.py` 출력) 또는 `{"position":(x,y,z),"orientation":(qx,qy,qz,qw)}`(플랜지 `right_fr3_link8`·world). **None이면 그 팔 이동 생략**(안전). **grip = 파지 위치 = 물체 내려놓기(해제) 위치**.
- `GRIP_FORCE_RANGE`·`SQUEEZE_FORCE_RANGE` — **힘 임계 랜덤 범위 [N]**(None=고정). 파지=run당 1회, 스퀴즈=스퀴즈마다 랜덤. 실제값은 그룹 attr에 기록(재현성).
- `OUTCOMES` — 데모 판정 라벨.

### 데모(run) 성공 판정
run 종료 후 터미널 프롬프트: `[1]success [2]grip_fail [3]not_judged [4]discard [Enter=success]`. `--no-judge`로 끔(전부 unjudged). 저장 **3곳**:
- HDF5 그룹 attr `outcome`, `/collect/demo_outcome` 토픽(bag), `outcomes.json`(세션 폴더 sidecar).
→ 학습 시 실패 제외/분리: `bag_to_hdf5 --skip-outcomes grip_fail,discard` 또는 `outcome` attr 필터.

### 설치 / 실행
```bash
pip install --user h5py                                       # HDF5 기록
sudo apt install ros-humble-moveit-msgs ros-humble-shape-msgs # 팔 이동(msg만, 라이브러리 아님)
source env.sh
# (팔 이동 시) 플래닝 PC: ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py joint_state_mode:=direct ...
python3 stiffness_deploy_ros2/launch/collect_ros2.py --fruit tomato --num-demos 20
#  --num-demos = 시퀀스 run 수(run당 스퀴즈 A/B 2그룹)
#  --paxini raw(127점) | --no-bag | --bag-storage sqlite3 | --no-judge
#  (선택) move_group 연결 스모크: python3 .../launch/moveit_arm_mover.py X Y Z   # plan-only
```

### 산출물 `collect_logs/collect_<fruit>_<ts>/`
- `collect_*.h5` — 그룹 `squeeze_A__run000, squeeze_B__run000, ...`(run+구간이 이름에). 각 그룹:
  `joint(n,16)·ft(n,12)·resultant(n,4,3)·tactile(n,4,127,3)·valid·squeeze_on·t_mono_ns`
  + attrs(`run·segment·outcome·grip/squeeze_force_threshold_n`). 세션 attrs에 provenance(git_sha·FACTOR·범위·paxini_source 등).
- `bag/` — 계약 토픽 raw + `/collect/{demo_marker, segment, squeeze_on, demo_outcome}`.
- `outcomes.json` — run별 판정 sidecar.

### bag → HDF5 (구간별, palm-down 포함)
```bash
python3 stiffness_deploy_ros2/launch/bag_to_hdf5.py collect_logs/collect_<fruit>_<ts>/bag
#  → from_bag.h5. 그룹 {segment}__run{NNN} + outcome attr.
#  실패 제외: --skip-outcomes grip_fail,discard  |  일부만: --segments move_palm_down
#  다른 rate 재추출: --rate 100  |  촉각: --paxini raw|ft|auto
```
- **스퀴즈**: 손 tick(`/hand/right/q_target`) + `joint/ft/resultant/tactile` (라이브와 동일 프레임, squeeze_on으로 창 좁힘).
- **palm-down**: 팔 tick(`/franka/right/q_target`) + `arm_joint(7)/arm_q_target/hand_joint/resultant/tactile` — 라이브 HDF5엔 없는(손 add_sample 없음) **팔·촉각 구간을 bag에서 복원**.
- 필드 처리(joint=int(round)·ft→point0·Σ127)·valid 필터는 라이브 브리지 미러링. `rosbag2_py` 필요.

### parity 검증(P5)
```bash
python3 stiffness_deploy_ros2/launch/verify_parity.py <collect_*.h5> <from_bag.h5>
#  데모·프레임 단위: |Δn|·best-lag·채널별 max_abs_err·PASS/FAIL (종료코드 0/1)
```
(그룹명이 `{segment}__run` 통일됨 → `(run,segment)` attr 매칭으로 소폭 업데이트하면 스퀴즈 parity 직접 비교 가능. palm-down은 라이브 대응물 없어 대상 아님.)

### 데이터 읽기 (학습 로더)
```python
import h5py
with h5py.File(p) as f:
    for name in f:
        g = f[name]
        if g.attrs.get("outcome") in ("grip_fail", "discard"):     # 실패 제외
            continue
        if g.attrs["segment"] == "move_palm_down":                 # 팔/촉각
            arm, tac = g["arm_joint"][:], g["tactile"][:]
        else:                                                       # squeeze_A/B (모델 입력)
            joint, res = g["joint"][:], g["resultant"][:]
```

### 검증 수준
- ✅ 모든 신규 파일 `py_compile` + 핵심 로직 numpy 단위테스트 통과: 구간 창/run 할당/squeeze_on 좁힘/구간 추출(valid 필터·palm-down)/outcome 파싱·skip/trajectory resample/parity 정렬·오차/힘 랜덤.
- ⚠️ **런타임 미검증**(ROS + move_group + 로봇 필요).

### 게이트 / 남은 일
- **팔 이동**: 플래닝 PC `move_group`(joint_state_mode:=direct) + `ARM_POSES` 실제 값. 미설정이면 팔 이동 생략(손 시퀀스는 동작).
- **촉각 127점**: P0 `/paxini/right/raw` 발행(§2).
- **남은 일**: 로봇 실측 **end-to-end**(수집→변환→검증) 1회 + `verify_parity`를 `(run,segment)` attr 매칭으로 소폭 업데이트.

### 전체 파이프라인
`collect_ros2` → `bag_to_hdf5` → `verify_parity` (상세 커맨드는 `quick_start.txt` §D).

---

## 7. 다른 컴퓨터에서 이어가는 법

```bash
# 이 레포만 pull 하면 됨 (Gen3 불필요 — 설계문서·핸드오프 모두 이 안에 있음)
git -C ~/stiffness_deploy_ros2 pull
# 그다음: §2 게이트 → §6 실행
```

새 Claude 세션이면 이 문서(§0~§6)만 읽어도 전체 맥락·구현 상태·다음 액션을 복원할 수 있다. 더 깊은 근거(ZOH vs 보간, rate 논증, 재현성 분석)는 **같은 폴더의 `docs/ros2_data_collection_plan.md`** 참조.
