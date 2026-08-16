# ARCHITECTURE

## 1. 두 대의 PC

```
┌──────────────────────────────┐           ┌──────────────────────────────────────┐
│ Control PC   192.168.0.100   │           │ 이 PC (prime-ws) 192.168.0.101       │
│                              │           │                                      │
│  Dual_Arm RT 런타임 (SHM)    │           │  pipeline/run_pipeline.py  ← 순서 제어│
│  trajectory_receiver 브리지  │           │  skill-set/grasp            (seq 1)  │
│  ★ sequence_arbiter          │◀─ DDS ──▶│  skill-set/in-hand-…        (seq 2)  │
│  PaXini 촉각 writer          │  domain 9 │  skill-set/inference-…      (seq 3)  │
│  ★ front RealSense 발행      │           │  skill-set/place            (seq 4)  │
│  Franka 듀얼 암 + KISTAR 핸드│           │  MoveIt 트윈(move_group) + RViz      │
└──────────────────────────────┘           │  place 모델 서비스 5종 (HTTP, 로컬)  │
                                           └──────────────────────────────────────┘
```

- **실기를 실제로 움직이는 것은 Control PC 뿐이다.** 이 PC 의 skill 들은 목표(관절/포즈/손
  타겟)를 토픽으로 보낸다. MoveIt 트윈은 `ros2_control:=false use_fake_hardware:=true` 로
  떠서 계획·충돌검사·시각화만 한다.
- **카메라 발행자는 Control PC** 다(통합 시 확정된 사항). 이 PC 는 구독만 한다.
- 두 PC 는 `ROS_DOMAIN_ID=9`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` 로 같은 버스에 붙는다.

## 2. 시퀀스 규약 — 순서를 강제하는 것은 arbiter 다

**고정 배정 번호: 1=Pick, 2=Inhand, 3=Stiffness, 4=Place** (client_id 도 동일, 0 금지)

각 skill 프로그램은 스스로 다음을 한다:

```python
client = SequenceClient(3)          # 내 번호
client.wait_for_previous_done(2)    # 직전 번호 DONE 대기 (1번은 이 줄 없음)
with client:                        # 진입 = request_control + 1Hz+ 하트비트
    ...                             # 로봇 동작
client.shutdown()                   # 정상 탈출 = release_control → DONE
```

`/sequence_state` (`dual_arm_msgs/SequenceState`, latched) — 동일 내용이
`/sequence/shm_state` (`Int32MultiArray [seq_id, state, owner]`) 로도 나온다.

| state | 의미 |
|---|---|
| 0 IDLE | 대기 중 **또는 실패 회수** (하트비트 3초 끊김 → arbiter 가 강제 회수) |
| 1 RUNNING | 그 번호가 제어권을 쥐고 동작 중 |
| 2 DONE | **정상 종료** — 오직 이것만 "성공"을 뜻한다 |

즉 러너(`run_pipeline.py`)는 **감독자이지 지휘자가 아니다.** 하는 일은
(a) 단계 진입 시 프로그램 spawn, (b) `/sequence_state` 관측으로 성공/실패 판정,
(c) 실패 시 프로세스 그룹 종료(→ 하트비트 끊김 → arbiter 회수 → 뒷단계 중단).

### latched DONE 함정

이전 체인이 남긴 DONE 은 arbiter 재시작 전까지 latched 로 남는다. 그래서 러너는
**RUNNING 을 한 번 본 뒤의 DONE 만** 성공으로 인정한다. 하지만 skill 쪽
`wait_for_previous_done` 은 그런 구분을 하지 않으므로, **체인 재실행 전에는 arbiter 재시작**이
정석이다(러너가 preflight 에서 경고한다).

## 3. 단계별 내부

### seq 1 — 물체 파지 (`skill-set/grasp`)
진입점 `scripts/run_scenario1_host.py` (**시스템 python3.10**, rclpy 때문)

```
SequenceClient(1) Start
  → 카메라 캡처 (ROS 구독 → NPZ: rgb/depth/K)
  → [subprocess: conda grasp_fruit / py3.12]  SAM3 텍스트 검출 → mask PNG
  → [subprocess: conda grasp_fruit]           top-down 파지점 계산 → summary JSON
  → [subprocess: 시스템 py3.10]               robot_executor_scenario1.py 로 pick (쥔 채 유지)
  → End(DONE)
```
파이썬이 갈라지는 이유: ROS Humble 의 rclpy 는 py3.10 이고 SAM3 스택은 py3.12 다.
SAM3 는 HuggingFace `transformers.Sam3Model` 로 로드한다(가중치는 로컬 HF 캐시).

### seq 2 — 손 안 조작 (`skill-set/in-hand-reorientation`)
진입점 `scripts/inhand_sequence_2.py` (시스템 python3.10)

```
wait_for_previous_done(1) → Start
  → pose_commander.py 로 목표 pose 이동 (MoveIt 트윈 경유)
  → hand_joint_target_publisher.py 로 HDF5 손 관절 궤적 재생 (기본 in-hand/data/test_int.hdf5)
  → hand_manual_squeeze.py 로 오므리기
  → End(DONE)
```

### seq 3 — 물성(강성) 추론 (`skill-set/inference-physics-property`)
진입점 `stiffness_deploy_ros2/launch/deploy_task3_ros2.py` (시스템 python3.10 + `pip --user torch`)

```
wait_for_previous_done(2) → Start
  → 파지 상태 확인(현재 자세 hold) → 스퀴즈 모션 → PaXini/관절 시퀀스로 강성 추론
  → End(DONE)
```
과일별 모델(`models/*.pth`)과 포즈 프리셋(`launch/*.txt`)을 쓴다. 실행 시 과일 번호를
stdin 으로 받는다 — 러너가 `echo <n> |` 로 주입한다.
**팔은 움직이지 않고 손 스퀴즈만** 한다(이미 파지한 상태 전제).

### seq 4 — 물체 내려놓기 (`skill-set/place`)
진입점 `vision_pipeline/skill_server.py` — **상시 서버**다(다른 셋과 다름).

```
[Phase A] seq 1 이 RUNNING 이 되는 순간 parent(목적지) prewarm
          — 트레이 클라우드 · 구멍 · 웨이포인트 높이. 팔 동작 없음. 앞 단계들과 겹쳐 돈다.
[Phase B] wait_for_previous_done(3) → 제어권 획득
          — 팔은 이미 ~child_pose 이므로 **이동하지 않고** 그 자리에서 child 캡처
          → 인지 → 배치 → release → parent_pose 복귀 → End(DONE)
그리고 다음 과일을 위해 다시 대기(루프)
```

모델은 별도 HTTP 서비스 5종으로 상주한다(무거운 모델을 매번 로드하지 않기 위해):

| 서비스 | 포트 | conda env | GPU |
|---|---|---|---|
| `molmo_service` (포인팅) | 8810 | `molmo` | GPU0 단독 (~17GB) |
| `sam_service` | 8811 | `sam3` | GPU1 (~10GB) |
| `anyplace_service` | 8801 | `anyplace_cu128` | GPU1 (~2GB) |
| `igr_service` (Act-VH 형상완성 + 핸드 FK 클라우드) | 8816 | `anyplace_cu128` | GPU1 (~1GB) |
| `grid_service` (이미지 그리드 뷰어) | 8815 | `anyplace_cu128` | CPU |

진행 상황은 `/place/status` (`std_msgs/String`) 로 발행된다 —
`tools/diagnostics/place_logger.py` 로 구독할 수 있다.

## 4. move_group 은 정확히 1개

place skill 서버의 preflight 는 `/move_action` 서버가 **정확히 1개**일 것을 요구한다.
2개가 되면 과거 "모든 arm move 실패" 버그가 재현된다. 구 분산환경에서는 "산업부 PC 에서만
트윈을 띄운다"는 규칙이었고, 통합 후에는 **이 PC 에서 1개** 로 유지한다. 러너도 2개 이상을
발견하면 실행을 중단한다.

## 5. 수동 실행 (단계별 검증)

러너 없이 각 단계를 손으로 돌릴 수 있다. 먼저 `source tools/env/setup_env.sh`.

```bash
# seq 1 파지
cd skill-set/grasp
/usr/bin/python3 scripts/run_scenario1_host.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot --yes --query orange

# seq 2 손 안 조작
cd skill-set/in-hand-reorientation && /usr/bin/python3 scripts/inhand_sequence_2.py

# seq 3 물성 추론  (과일 번호: [1]자두 [2]키위 [3]토마토 [4]레몬)
cd skill-set/inference-physics-property
source env.sh && echo 4 | python3 stiffness_deploy_ros2/launch/deploy_task3_ros2.py

# seq 4 내려놓기 — 모델 서비스 먼저
bash skill-set/place/vision_pipeline/run_services.sh          # 터미널 A (상시)
cd skill-set/place && /usr/bin/python3 -u -m vision_pipeline.skill_server scenario=fruit hand_pc=true

# 진행 로그
/usr/bin/python3 tools/diagnostics/place_logger.py
```

## 6. 러너의 범위 — 무엇을 하고 무엇을 하지 않는가

`pipeline/run_pipeline.py` 는 **의도적으로 얇다.**

| 한다 | 하지 않는다 |
|---|---|
| 상시 서버(모델 5종/트윈/place) 확인·기동 | 자연어 해석 — 대상은 CLI 인자로만 받는다 |
| 단계 진입 시 skill 프로그램 spawn | 장면 지각 — 파지 skill 의 SAM3 가 직접 검출한다 |
| `/sequence_state` 관측으로 성공/실패 판정 | 작업 종류 분기 — 항상 4단계 고정 |
| 실패 시 프로세스 그룹 종료 → 체인 중단 | 순서 강제 — arbiter 가 한다 (§2) |

의존은 시스템 python3 + PyYAML 뿐이다. 별도 venv, 모델 서버, 설정 DSL 이 없다.

이 얇음이 설계 의도다. 상위 판단(무엇을 어디에 어떤 순서로)은 추후 VLM high-level planner 의
몫이고, planner 는 이 스크립트를 인자만 바꿔 호출하면 된다:

```bash
./run_fruit_demo.sh --fruit <물체명> --stiffness-fruit <과일>
```

종료 코드는 성공 0 / 실패 1 / 사용자 중단 130 이라 상위 계층이 결과를 그대로 받아볼 수 있고,
`--dry-run`(사전 점검만) 과 `--skip`(단계 일부) 로 통합 중 단계별 검증이 가능하다.

## 7. 손 제어 모드 (Position / Voltage) — 시스템 안전 불변식

KISTAR 핸드는 두 모드로 동작한다. `/hand/right/cmd_mode` (`std_msgs/Int32`):

| 값 | 모드 | `/hand/right/q_target` (Float32[16]) 의 의미 |
|---|---|---|
| 1 | **Position** | 엔코더 **counts** (수백~수천) |
| 0 | **Voltage** | raw **PWM duty** (±2100) |

**같은 토픽, 같은 타입, 전혀 다른 단위다.** 그래서 다음이 성립해야 한다:

> **불변식 — Voltage 모드는 내려놓기(seq 4)의 release 구간에서만 존재하고, 그 구간을 벗어나는
> 모든 경로에서 Position 으로 복귀해야 한다.**

깨지면 무슨 일이 일어나는가: Voltage 모드가 남은 채 다음 단계(파지·손 안 조작·물성 추론)가
Position counts 를 보내면 **수백~수천이 raw duty 로 해석되어 핸드가 폭주한다.**

### 지키는 방법 (3중)

| 계층 | 위치 | 내용 |
|---|---|---|
| ① 정상 경로 | `place/vision_pipeline/orchestrator.py` `_release_and_retract` | release/retract 전체가 `try/finally`. [R5] `hand_safe_shutdown()` (duty 0 → **servo OFF 먼저** → mode=Position → servo OFF 유지)가 정상·예외·중단 **모든 경로**에서 실행된다 |
| ② 프로세스 이상 종료 | `place/vision_pipeline/backends/ros_backend.py` `_arm_hand_safety_net` | `atexit` + SIGINT/SIGTERM 핸들러. **우리가 Voltage 로 바꾼 경우에만** 복원한다(건드린 적 없는 파지는 그대로 둬서 물체를 떨어뜨리지 않는다) |
| ③ 단계 진입 | `pipeline/config.yaml` | 파지·손 안 조작·물성 추론 각 단계 명령 앞에 `cmd_mode = 1`(Position) 발행 |

SIGKILL·전원 차단은 어떤 계층으로도 막을 수 없다. 그래서 러너는 단계 종료 시
**SIGTERM → 5초 유예 → SIGKILL** 순으로 보내 ②가 동작할 시간을 준다.

Position→Voltage 전환 자체도 hot-switch 금지 규칙을 따른다:
servo OFF → mode 설정 → **duty 0 시딩 ×2** (`q_target` 이 BEST_EFFORT 라 한 번 유실되면 이전
Position counts 가 duty 로 읽힌다) → servo ON. 모든 duty 는 `_safe_duty16()` 으로 ±500 하드
클램프 + 비유한값 0 처리.

### 그립 인수인계 (Stiffness 3 → Place 4)

물체는 파지(1)부터 계속 쥐어져 있고, 물성 추론(3)이 그 파지를 유지하는 Position target 을
`/hand/right/q_target` 으로 스트리밍한다. seq 4 로 제어권이 넘어오면 **직전 소유자는 스트리밍을
멈추고**, `require_control:=true` 에서는 소유자가 아닌 노드의 target 은 무시된다.

그래서 place 서버는 제어권 획득 직후 **직전 단계가 보내던 target 을 그대로 이어받아** release
직전까지 20Hz 로 유지한다:

- `ros_backend` 가 카메라 전용 executor 스레드에서 `/hand/right/q_target` 을 구독해 **외부**
  target 을 캐시한다(메인 노드는 모델 HTTP 호출·MoveIt 대기 중 spin 되지 않아 놓친다).
- `hand_hold_start()` 가 그 값을 스냅샷해 전용 스레드에서 재발행한다
  (rclpy publish 는 executor 없이 동작하므로 긴 블로킹 구간에도 유지된다).
- release 경로(`hand_release_sequence`)와 `hand_safe_shutdown()` 이 **Voltage 전환 전에**
  hold 를 멈춘다 — counts 를 Voltage 모드로 흘리면 그 자체가 폭주다.
- 인수할 target 이 없거나 30초 이상 오래되었으면 **아무것도 붙잡지 않는다.** 16-DoF 목표를
  추측해 발행하면 그 자체가 파지를 다시 명령하는 것이라, 수신 노드가 래치한 값에 맡기는 편이
  안전하다. 이 경우 경고를 로그와 `/place/status` 에 남긴다.

회귀 테스트: `python3 skill-set/place/vision_pipeline/test_hand_safety.py` (ROS·로봇 불필요)
