# RUNBOOK — 데모 실행 절차

> 최초 설치는 [INSTALL.md](INSTALL.md). 이 문서는 **설치가 끝난 상태에서 매번 하는 일**이다.
>
> ⚠ 아래 절차는 실제 로봇을 움직인다. **E-stop 옆 인원 상주 필수.**

---

## 0. 한 장 요약

```bash
# [Control PC 담당자]  RT 런타임 → control_pc.launch(require_control:=true) → PaXini writer → 카메라
# [이 PC]
cd ~/prime/ChanukHwang/RobotAgentSystem
tools/diagnostics/preflight.sh          # 로봇 안 움직임 — 여기서 전부 초록이어야 함
./run_fruit_demo.sh --fruit orange      # 파지 → 손 안 조작 → 물성 추론 → 내려놓기
```

---

## 1. Control PC (192.168.0.100) — 그쪽 담당자

순서대로. 모두 `ROS_DOMAIN_ID=9`.

### 1-1. Dual_Arm RT 런타임 (SHM 0x7951 생성)
```bash
cd /home/prime/Dual_Arm_Hand_Ctrl
sudo FRANKA_ARM_R_IP=172.16.0.1 FRANKA_ARM_L_IP=172.17.0.1 \
  ./build/test/Dual_Arm_Hand_Imp_Ctrl_V1_0 enp1s0f0 enp1s0f1
```

### 1-2. control_pc 브리지 + sequence_arbiter + **front 카메라**
```bash
cd /home/prime/Dual_Arm_Hand_Ctrl/ros2
source /opt/ros/humble/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=9
ros2 launch trajectory_receiver control_pc.launch.py require_control:=true
```
- `require_control:=true` — 제어권 없이 보낸 타겟은 무시된다(정상). 시퀀스 모드의 전제.
- `arbiter:=true`(기본) — `sequence_arbiter` 가 뜬다. 이게 없으면 아무 단계도 시작하지 않는다.
- **`camera:=true`(기본) — front RealSense 도 이 launch 가 함께 띄운다.** 2026-08 통합 시
  Control PC 의 bring-up 에 카메라 파라미터가 영구 반영되어, 별도 카메라 launch 가 필요 없다.
  (카메라만 따로 띄우려면 나머지를 `state_pub:=false trajectory:=false ee:=false arm_q:=false
  hand:=false arbiter:=false` 로 끄면 된다.)

> **체인을 다시 돌리기 전에는 arbiter 재시작을 요청할 것.** 직전 체인의 DONE 이 latched 로
> 남아 있으면 다음 체인의 `wait_for_previous_done` 이 기다리지 않고 곧장 통과한다.

### 1-3. PaXini writer
⚠ **물체를 잡기 전, 손끝 무접촉 상태**에서 실행한다(시작 순간을 0점 tare).
```bash
cd /home/prime/Dual_Arm_Hand_Ctrl
python3 ./tools/paxini_writer.py --hand r
```

---

## 2. 이 PC — 점검

```bash
cd ~/prime/ChanukHwang/RobotAgentSystem
tools/camera/check_camera.sh            # 카메라 3스트림 (Control PC 발행)
tools/diagnostics/preflight.sh          # 전체 점검 (로봇 미동작)
```

`preflight.sh` 가 보는 것:

결과는 **두 등급**으로 나온다:

| 등급 | 의미 |
|---|---|
| `PASS` | 정상 |
| `AUTO` | 지금은 없지만 **실행 시 러너가 자동 기동한다** — 조치 불필요 |
| `BLOCK` | **사람이 조치해야 한다** — 이것만 해결하면 된다 |

| 항목 | 등급 | 실패 시 |
|---|---|---|
| `ROS_DOMAIN_ID=9` | BLOCK | `tools/env/setup_env.sh` 를 source 했는지 |
| `/sequence_state` 발행자 | BLOCK | Control PC 의 `sequence_arbiter` 미기동 (§1-2) |
| 카메라 3스트림 발행자 | BLOCK | Control PC 카메라 미기동 / 방화벽 / 도메인 불일치 |
| `move_group` | AUTO / BLOCK | 0개면 러너가 기동(AUTO). **2개 이상이면 BLOCK** — 중복 `/move_action` = 전 이동 실패 버그 |
| 모델 서비스 5종 `/health` | AUTO | 러너가 기동 (Molmo 로딩 ~1분+) |

`BLOCK` 이 없으면 종료 코드 0 이고, `AUTO` 만 남았다면 그대로
`./run_fruit_demo.sh` 를 돌리면 된다.

preflight 는 latched DONE 이 남아 있으면 경고한다 → arbiter 재시작 요청.

> **"목록에 없다"를 "죽었다"로 읽지 말 것.** `ros2 node list` · `ros2 topic list` ·
> `ros2 service list` 는 **셋 다 ros2 데몬 캐시**를 거친다. 데몬이 방금 떴거나 오래된 상태면
> 살아 있는 것을 빈 목록으로 보고한다. 실제로 "토픽 목록은 없음, 그런데 같은 토픽이 30Hz 로
> 수신 중"인 자기모순 출력이 나온 적이 있다.
>
> 신뢰할 수 있는 확인 순서:
> ```bash
> ros2 topic hz /front_cam/front/color/image_raw   # ① 실제 데이터 — 가장 확실
> ros2 topic list --no-daemon                      # ② 데몬 우회
> ros2 daemon stop                                 # ③ 캐시 비우고 다시
> ```
> `tools/camera/check_camera.sh` 와 `tools/diagnostics/preflight.sh` 는 모두 ①/rclpy 직접
> 조회로 판정하므로 이 함정에 걸리지 않는다.

---

## 3. 실행

```bash
./run_fruit_demo.sh --fruit orange
```

| 인자 | 의미 |
|---|---|
| `-f, --fruit <name>` | 파지할 물체명 = SAM3 텍스트 쿼리 (기본 `orange`) |
| `--stiffness-fruit {plum,kiwi,tomato,lemon}` | 강성 모델 과일. 미지정 시 `--fruit` 에서 매핑 |
| `--skip grasp,inhand,...` | 단계 건너뛰기 (단계별 검증용) |
| `--twin off` / `--services off` | 자동 기동 끄기 (이미 떠 있는 것을 쓸 때) |
| `--place-logger` | `/place/status` 이벤트 로거 동반 기동 |
| `--no-stiffness-gui` | 강성 결과 GUI 를 띄우지 않음 (기본은 띄움) |
| `--timeout-scale 2.0` | 모든 단계 타임아웃 배율 |
| `--dry-run` | preflight 만 |

### 화면에 뜨는 것 (분산환경 때와 동일)

| 화면 | 언제 | 무엇이 띄우나 |
|---|---|---|
| **RViz** | 트윈 기동 시 (러너가 자동 기동하거나 이미 떠 있으면 재사용) | `dual_fr3_kistar_moveit.launch.py use_rviz:=true` |
| **grid 브라우저 탭** (`:8815`) | 모델 서비스 기동 시 자동으로 브라우저를 연다 | `grid_service.py` — place 단계의 캡처/마스크 이미지 그리드 |
| **place 디버그 마커** | place 단계 진행 중 | `/place_debug/markers` → **RViz 안에** 표시 (별도 창 아님). 후보 포즈·포인트클라우드 |
| **강성 결과 창** | 물성 추론(seq 3) 시작 시 | `stiffness_gui.py` 별도 프로세스. 배포가 끝나도 결과를 계속 보여주려 자동 종료하지 않는다(창을 닫으면 정리) |

세 GUI 모두 러너에서 **DISPLAY 를 상속**받는다. DISPLAY 없는 셸(예: `ssh` 무-X)에서 실행하면
아무 창도 뜨지 않는다 — preflight 가 이 경우 경고를 찍는다:

```
WARN  DISPLAY 가 없다 — RViz · 강성 결과 GUI · grid 브라우저가 뜨지 않는다 (헤드리스로 진행됨)
```

헤드리스로 돌리려면 `--no-stiffness-gui` + `--twin off`(RViz 없이 미리 띄운 트윈 사용)를 쓴다.

### 무슨 일이 일어나는가

```
[러너]  모델 서비스 5종 확인/기동  →  MoveIt 트윈 확인/기동
        →  place skill 서버 기동 (READY 대기)
        →  seq 1 파지 spawn        → arbiter 가 [1, DONE] 낼 때까지 대기
        →  seq 2 손 안 조작 spawn  → [2, DONE] 대기
        →  seq 3 물성 추론 spawn   → [3, DONE] 대기
        →  seq 4 는 place 서버가 스스로 이어받음 → [4, DONE] 관측 → 종료
```

- 순서를 강제하는 것은 러너가 아니라 **Control PC 의 arbiter** 다. 각 skill 은
  `SequenceClient(n)` 으로 직전 번호의 DONE 을 스스로 기다린다.
- place 서버는 seq 1 이 RUNNING 이 되는 순간 **parent(목적지) 인지를 prewarm** 한다 —
  파지/손안조작/물성추론이 도는 동안 겹쳐서 끝내므로 마지막 단계가 빨라진다.
- 어느 단계든 실패하면 러너가 프로세스 그룹을 죽인다 → 하트비트 끊김 → arbiter 가 3초 내
  IDLE 로 회수 → **뒷단계는 시작하지 않는다.**

### 로그

```
logs/run_MMDD_HHMMSS/
├── pipeline.log        러너 타임라인
├── skill_grasp.log     각 단계 stdout/stderr
├── skill_inhand.log
├── skill_stiffness.log
├── place_server.log
├── model_services.log
└── twin.log
```

---

## 4. 처음 브링업할 때 (권장 순서)

1. `tools/diagnostics/preflight.sh` — 전부 통과할 때까지
2. `./run_fruit_demo.sh --fruit orange --skip inhand,stiffness,place` — 파지만
3. `--skip stiffness,place` → `--skip place` → 전체
4. 각 단계는 개별 실행으로도 검증 가능 → [ARCHITECTURE.md](ARCHITECTURE.md) §수동 실행

---

## 5. 매번 확인할 것

- [ ] **arbiter 재시작** 했는가 (직전 체인의 latched DONE 제거)
- [ ] **PaXini writer** 를 손끝 무접촉 상태에서 켰는가 (0점 tare)
- [ ] **카메라 캘리브레이션** — 카메라/로봇 배치가 바뀌었다면 파지·내려놓기 두 곳 모두 갱신
      ([MIGRATION.md](MIGRATION.md) §리스크)
- [ ] **강성 과일 매핑** — 강성 모델에 orange 가 없어 lemon 으로 대용 중.
      데모 과일이 바뀌면 `--stiffness-fruit` 또는 `pipeline/config.yaml` 수정
- [ ] **move_group 1개** — RViz 를 따로 띄웠다면 트윈이 2개가 아닌지
- [ ] E-stop 접근 가능한 위치, 첫 동작은 저속
