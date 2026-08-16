# stiffness_deploy_ros2

`deploy.py`(SHM 직접 제어)의 **모션 시퀀스 · 힘 임계 판정 · 강성 추론 로직을 그대로 재사용**하고,
SHM 직접 접근(I/O 계층)만 **Dual_Arm_Hand_Ctrl 워크스페이스의 ROS2 토픽**으로 치환해 실행하는
브리지 패키지. (원본 Gen3 저장소에서 `deploy_ros2.py` 실행에 필요한 파일만 추린 것)

> `deploy.py` 로직과 `Dual_Arm_Hand_Ctrl` 는 **수정하지 않는다.** I/O 계층만 토픽으로 교체.

---

## 1. 포함 파일 (deploy_ros2 의존 트리)

```
stiffness_deploy_ros2/
├── package.xml            # ROS2 의존성(rosdep 설치용)
├── setup.py / setup.cfg   # colcon(ament_python) 빌드 및 ros2 run 진입점
├── requirements.txt       # pip 의존성(torch 등)
├── resource/stiffness_deploy_ros2
├── docs/
│   ├── TROUBLESHOOTING.md         # ★ 배포 문제해결 & 개발 노트 (QoS/디바이스/도메인 불일치 등)
│   ├── UPDATE_RATE_CHECK.md       # ★ update rate 계측 절차 + 결과 기록 템플릿
│   ├── UPDATE_RATE_CONCLUSION.md  # ★ update rate 결론 보고(발표용 요약)
│   └── todolist.md                # ★ 남은 과제(이어서 작업용) — 배경/위치/완료판정
├── tools/
│   ├── measure_update_rate.sh     # rate 계측 원-커맨드 러너(결과 자동 저장)
│   ├── sensor_update_rate.py      # ★ 센서 실측 갱신율(값 변화 기준) — hz 는 통신속도일 뿐
│   ├── analyze_change_rate.py     # 값 변화 간격 분포 분석(최소~최대·평균, 채널별)
│   └── rate_summary.py            # 계측 로그 → 마크다운 요약(재분석용)
└── stiffness_deploy_ros2/
    ├── launch/
    │   ├── deploy_ros2.py             # ★ 진입점 (ROS2 브리지) — 손만 안전→파지→스퀴즈, 추론 후 메뉴 선택
    │   ├── deploy_task3_ros2.py       # ★ 진입점 (ROS2 브리지) — 시퀀스 체인용(이미 파지한 상태에서 스퀴즈만)
    │   ├── deploy_ros2_exp.py         # 실험/계측 전용(엔진 wrapper) — 스퀴즈당 샘플수·rate·힘 출력
    │   ├── deploy_task3.py            # task3 시퀀스 로직(비-ROS) — 파지확인(1s)→스퀴즈
    │   ├── deploy.py                  # 모션 시퀀스/힘 판정 (Franka 고정·물체원위치 제거·과일별 임계값)
    │   ├── real_deploy_inference_final.py # ★ 추론 엔진(방식2/3 통합+과일조건, 변위 O/X)
    │   ├── real_deploy_inference_old.py   # 구 추론 엔진(과일별)
    │   ├── model.py                   # 모델 정의(StiffnessRegressor 등 — 학습본과 일치 필요)
    │   ├── hand_pose_io.py            # 포즈 txt 로더(rad↔count)
    │   ├── fruit_thresholds.yaml      # 과일별 파지/스퀴즈 힘 임계값
    │   └── *.txt                      # 과일 파지 포즈 + 안전 포즈
    │       (initial_pose / kiwi / tomato / plum / lemon)
    ├── core/
    │   ├── shm_common.py              # 상수(Arm_DOF, Hand_DOF …) + ShmAccess
    │   └── paxini_shm.py              # PaXini SHM 리더(상수/타입)
    ├── models/                        # 강성 추론 체크포인트(.pth) — 레포 번들
    │   ├── 260630_1006_transformer_fruit_A_...s64.pth    # tomato 과일별 모델
    │   └── 260707_*_lstm_m{2,3}_d{X,O}_...pth            # 통합(방식2/3) 모델
    └── labels/{general,trial2}/       # 정규화 통계(class/name/stiffness.yaml) — 레포 번들
```

> 배포 중 발생한 문제와 해결(추론 안 됨·동작 끊김·손가락 드리프트·도메인 불일치 등)은
> **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** 에 정리되어 있다.

> 모델(.pth)·라벨(yaml)이 **레포에 포함**되어 있어 별도 절대경로 설정 없이 바로 동작한다.
> 경로는 `real_deploy_inference_old.py` 가 `Path(__file__)` 기준(`models/`·`labels/`)으로 찾는다.

`deploy_ros2.py` 는 `deploy.py` 를 그대로 import 재사용하며, `deploy.py` 내부의
`from launch.real_deploy_inference_old …` 충돌(ROS `launch` 패키지)을 스스로 우회한다.
따라서 위 `launch/` · `core/` 상대 배치를 **바꾸지 말 것.**

---

## 2. 설치

통합 워크스페이스의 `src/` 아래에 이 저장소를 두고:

```bash
# 1) ROS2 의존성 (rclpy, sensor_msgs, std_msgs, numpy, pyyaml)
cd <통합_ws>
rosdep install --from-paths src/stiffness_deploy_ros2 --ignore-src -r -y

# 2) pip 의존성 (torch 등 — package.xml rosdep 키가 없는 것)
pip install -r src/stiffness_deploy_ros2/requirements.txt

# 3) (선택) colcon 빌드 — ros2 run 으로 실행하려는 경우
colcon build --packages-select stiffness_deploy_ros2
source install/setup.bash
```

---

## 3. 모델 / 라벨 (레포 번들 — 경로 설정 불필요)

추론 엔진이 쓰는 **체크포인트(.pth)와 정규화 통계(labels/*.yaml)가 이 레포에 포함**되어 있어,
과거처럼 `real_deploy_inference_old.py` 의 절대경로를 대상 장비에 맞게 고칠 필요가 없다.
경로는 `Path(__file__)` 기준으로 `models/`·`labels/` 를 가리키도록 이미 연결돼 있다.

| 자원 | 위치 |
|------|------|
| tomato 강성 모델 | `stiffness_deploy_ros2/models/260630_1006_transformer_fruit_A_...s64.pth` |
| 정규화 통계 | `stiffness_deploy_ros2/labels/{class,name,stiffness}.yaml` |

데모는 **tomato 기준**으로 준비됨. `FRUIT_CONFIG` 에서 모델이 `None` 인 과일(plum/kiwi/lemon)은
실행 시 안내 후 종료된다 — 다른 과일을 쓰려면 해당 `.pth` 를 `models/` 에 넣고 `FRUIT_CONFIG` 에 등록.

### 과일별 힘 임계값 (`launch/fruit_thresholds.yaml`)

파지/스퀴즈 힘 임계값을 **과일별로** 지정한다. 과일 선택 시 `set_thresholds_for_fruit(fruit)` 가
이 파일을 읽어 `GRIP_FORCE_THRESHOLD`(파지)·`SQUEEZE_FORCE_THRESHOLD`(스퀴즈)를 갱신한다.

```yaml
default:            # fruits 에 없거나 키가 빠지면 이 값 사용
  grip: 7.0
  squeeze: 10.0
fruits:
  tomato: { grip: 7.0, squeeze: 10.0 }
  kiwi:   { grip: 5.0, squeeze: 8.0 }   # 예: 과일마다 다르게
```

- `grip` = 파지 접촉력 [N] — **직접 파지하는 `deploy` / `deploy_ros2` 만 사용**.
- `squeeze` = 스퀴즈 정지력 [N] (thumb) — **4개 파일 공통**. (`deploy_task3*` 는 이미 파지 상태라 squeeze 만 사용)
- 파일이 없거나 파싱 실패 시 기본값(grip 7.0 / squeeze 10.0)으로 안전하게 진행.
- 직접 실행(python3)은 값 변경 즉시 반영, colcon 설치본은 재빌드·설치 후 반영.

---

## 4. 실행

전제: **Dual_Arm_Hand_Ctrl 스택이 실행 중**이어야 한다 — C++ 컨트롤러 +
`shm_state_publisher_node` + `arm_q_target_receiver_node` + `hand_target_receiver_node`
(+ PaXini writer). 자세한 기동은 원본 Gen3 README 및 `Dual_Arm_Hand_Ctrl/docs/Quick_commands.md` 참고.

진입점은 두 가지(+ 계측용 1개)다.

| 진입점 | 용도 | 시퀀스 |
|--------|------|--------|
| `deploy_ros2` | **단독/테스트용** | 손만 안전→파지→스퀴즈→추론. **추론 후 메뉴**로 [1]재스퀴즈 [2]안전복귀 후 재파지→스퀴즈 [3]안전복귀 후 종료 선택 |
| `deploy_task3_ros2` | **시퀀스 체인용** | **이미 물체를 파지한 상태**에서 시작 → 파지 확인(1s) → 스퀴즈 → 추론, 1회. 시퀀스 제어권(Stiffness=3) 프로토콜 사용 |
| `deploy_ros2_exp` | **실험/계측** | deploy_ros2 와 동일 동작 + 스퀴즈당 샘플수·유효 rate·도달 힘 출력 |

> **모든 진입점에서 Franka(팔)는 이동하지 않고 손(Hand)만 움직인다.** 팔은 실행 전 자세에 고정, 물체 원위치 단계 없음.
> `deploy_task3_ros2` 는 실행 전에 **이미 물체를 파지**하고 있어야 하며, 완료는 시퀀스 End(제어권 반납)로 다음(Place)에 알린다.
> ROS2 브리지(토픽 I/O)는 세 진입점이 `deploy_ros2` 의 것을 공유한다.

```bash
source /opt/ros/humble/setup.bash
source <Dual_Arm_Hand_Ctrl>/ros2/install/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# (A) 단독/테스트용: 안전→파지→스퀴즈→추론 후 메뉴 선택
python3 src/stiffness_deploy_ros2/stiffness_deploy_ros2/launch/deploy_ros2.py
ros2 run stiffness_deploy_ros2 deploy_ros2          # colcon 빌드 후

# (B) 실제 task용: 이미 파지한 상태에서 스퀴즈만
python3 src/stiffness_deploy_ros2/stiffness_deploy_ros2/launch/deploy_task3_ros2.py
ros2 run stiffness_deploy_ros2 deploy_task3_ros2    # colcon 빌드 후
```

실행 후 과일 번호를 입력한다(`deploy.py` 와 동일 UX). 정상 시:

```
# deploy_ros2 (테스트용, 팔 고정·손만)
[deploy_ros2] 추론엔진 준비 완료. 과일=..., 모델=...
--- 데모 0/10 ---
... (손: 안전→파지→스퀴즈, 스퀴즈 직후 절대강성/등급 출력) ...

# deploy_task3_ros2 (실제 task용)
[deploy_task3_ros2] 추론엔진 준비 완료. 과일=..., 모델=...
※ 이미 물체를 파지한 상태에서 시작합니다.
... (파지 확인 1s → 스퀴즈 → 절대강성/등급 출력) ...
```

---

## 5. 공용 워크스페이스에 올린 뒤 — 무엇부터? (bring-up 순서)

아래 순서대로 확인하면 **안전한 것 → 통합**으로 단계적으로 올릴 수 있다. 각 단계가 통과해야 다음으로 넘어간다.

### 0단계 — 배치 & 빌드

```bash
cd <통합_ws>
rosdep install --from-paths src/stiffness_deploy_ros2 --ignore-src -r -y
pip install -r src/stiffness_deploy_ros2/requirements.txt   # torch 등

# 시퀀스용 패키지(제어 PC 저장소에서 복사): dual_arm_msgs + sequence_client
#   → deploy_task3_ros2(시퀀스 이어받기) 실행에 필수. deploy_ros2(테스트)만 쓸 거면 생략 가능.
colcon build --packages-select dual_arm_msgs sequence_client stiffness_deploy_ros2
source install/setup.bash
```

### 1단계 — 환경/디스커버리 확인

```bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
ros2 node list      # 제어 PC 스택(shm_state_publisher 등)이 보이는지
```

### 2단계 — 토픽 계약 확인 (이 패키지가 **구독**하는 것들이 실제 오는지)

```bash
ros2 topic echo --once /franka/right/joint_states   # 팔 상태
ros2 topic echo --once /hand/right/joint_states     # 손 상태(count)
ros2 topic echo --once /hand/right/kin              # 추론 입력(12)
ros2 topic hz          /paxini/right/ft             # 우측 촉각 ~85Hz (없으면 힘=0 진행)
```

하나라도 안 오면 → `attach()` 타임아웃으로 실행이 바로 종료된다(7절 문제해결 참고). 먼저 이걸 해결.

### 3단계 — 설정 sanity 체크

```bash
# (a) 과일별 임계값 config 확인
cat src/stiffness_deploy_ros2/stiffness_deploy_ros2/launch/fruit_thresholds.yaml

# (b) 시퀀스 상수명 확인 — deploy_task3_ros2 가 SequenceState.SEQ_STIFFNESS 를 씀
ros2 interface show dual_arm_msgs/msg/SequenceState | grep SEQ_
#   → SEQ_STIFFNESS / SEQ_INHAND 가 있는지. 이름이 다르면 deploy_task3_ros2.py 의 상수명을 맞춘다.
```

### 4단계 — 단독 스모크 테스트 (`deploy_ros2`, 시퀀스 프로토콜 없음) ← **여기부터 실제 실행**

가장 먼저 이걸 돌린다. **arbiter/제어권 불필요, 팔 고정, 손만** 움직여 파이프라인 전체(토픽·모델·힘·추론)를 한 번에 검증한다.

```bash
python3 src/stiffness_deploy_ros2/stiffness_deploy_ros2/launch/deploy_ros2.py   # 과일 번호 입력
```

확인 포인트:
- `[threshold] <과일>: 파지=..N, 스퀴즈=..N` 로그가 **그 과일 값**으로 찍히는지
- **팔이 전혀 안 움직이는지** (손만 안전→파지→스퀴즈)
- 스퀴즈 직후 `[추론 결과] 절대강성/등급` 출력
- ※ 11데모 반복이므로 확인만 하면 `Ctrl+C`. 물체 없이 돌리면 힘=0 → 손이 target 까지만 닫히는 dry-run.

### 5단계 — 시퀀스 통합 테스트 (`deploy_task3_ros2`, 제어권 프로토콜)

4단계가 정상이면 진행. **전제**: 제어 PC 가 `require_control:=true` 로 launch, `sequence_arbiter` 실행, **물체를 이미 파지**(Inhand가 넘겨준 상태 또는 수동 파지).

```bash
# 상태 확인(직전 Inhand=2 가 DONE 이어야 3번이 시작)
ros2 topic echo /sequence_state --qos-durability transient_local --qos-reliability reliable

python3 src/stiffness_deploy_ros2/stiffness_deploy_ros2/launch/deploy_task3_ros2.py
```

확인 포인트:
- `[sequence] 직전 Inhand(#2) DONE 대기...` → `제어권 획득 → Stiffness(#3) 시작`
- 스퀴즈+추론 완료 후 자동 **End** → `/sequence_state` 가 `{seq_id=3, DONE}` 으로 바뀌는지
- 3번만 단독으로 돌릴 땐 직전(2) DONE 이 없어 계속 대기한다 → 제어 PC 담당자에게 arbiter 재시작/2번 DONE 상태를 요청.

---

## 6. 토픽 인터페이스

> 토픽명은 실제 시스템(`ros2 topic list`, 2026-07-06) 기준 `/<side>/` 규약(side=right). L 확장은 `right`→`left`.

**명령 (deploy_ros2 → Ctrl, publish)**

| 토픽 | 타입 | 내용 |
|------|------|------|
| `/franka/right/q_target` | Float64MultiArray[7]  | 팔 관절 목표 |
| `/hand/right/q_target`   | Float32MultiArray[16] | 손 관절 목표 |
| `/hand/right/cmd_servo`  | Bool  | 서보 on/off |
| `/hand/right/cmd_mode`   | Int32 | 핸드 모드(0=volt,1=pos,2=cur) |

**상태 (Ctrl → deploy_ros2, subscribe)**

| 토픽 | 타입 | 내용 |
|------|------|------|
| `/franka/right/joint_states` | JointState        | R팔 q = position[0:7] |
| `/hand/right/joint_states`   | JointState        | R손 q = position[0:16] |
| `/hand/right/kin`            | Float32MultiArray | R손 kinesthetic (4x3), 추론 입력 |
| `/paxini/right/ft`          | Float32MultiArray | R손 손가락별 합력 (4x3), 힘 판정 |

QoS: 상태 best-effort, 단발 명령(servo/mode) reliable, **고속 스트림(q_target)은 best-effort**
(수신 노드와 매칭 + write 블로킹 방지 — [TROUBLESHOOTING §C1](docs/TROUBLESHOOTING.md) 참고). 단일 팔/손(R) 기준.

---

## 7. 문제 해결

> 배포 중 겪은 상세 문제·원인·해결과 개발 노트는 **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** 참고.
> 아래는 실행 시 자주 만나는 것만.

| 증상 | 원인 / 해결 |
|------|------------|
| `상태 토픽 미수신` 후 종료 | Ctrl `shm_state_publisher_node`(+C++ 컨트롤러) 미실행 / `ROS_DOMAIN_ID` 불일치 |
| `/paxini/right/ft 미수신` 경고 | PaXini writer 미실행 → 힘=0 으로 진행(안전 fallback) |
| 팔/손이 안 움직임 | Ctrl `arm_q_target_receiver_node`/`hand_target_receiver_node` 미실행, 또는 명령 QoS 불일치 |
| `ModuleNotFoundError: torch/yaml` | `pip install -r requirements.txt` 미실행 |
| `'plum' 설정 없음`/모델 안내 후 종료 | `FRUIT_CONFIG` 의 해당 과일 `.pth` 경로 미설정 |
