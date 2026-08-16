# 시나리오1 로봇 실행 가이드 — 처음부터 전체 순서

> Pick(1) → 물체 쥔 채 유지 → 제어권 반납(DONE) → Inhand(2) 로 넘김.
> 제어 PC(메인 PC, 로봇 연결)에서 실행. 카메라·SAM3·PCA + 시퀀스 제어권까지.
> 참고: Dual_Arm_Hand_Ctrl/docs_dev/{ROS2_TOPIC_GUIDE, SEQUENCE_GUIDE}.md

---

## 0. 네트워크 (모든 터미널 공통 — 맨 위에)

```bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
```
- 제어 PC 한 대에서 다 돌리면 localhost라 subnet 문제 없음.
- 로봇 IP: **172.16.0.1** (FCI) / 제어 PC 내부: **192.168.0.100**

---

## 1. 일회성 셋업 (처음 한 번만)

```bash
# ① Topdown_Grasp + 비전 환경
cd ~/GW    # (원하는 위치)
git clone https://github.com/jjuuaaee/Topdown_Grasp_sja.git
cd Topdown_Grasp_sja
CONDA_BASE=~/miniconda3 bash setup_pipeline_all.sh     # conda grasp_fruit (torch cu128, SAM3)
huggingface-cli login                                  # SAM3 gated 접근 (facebook/sam3)

# ② configs/paths.yaml 을 제어 PC 경로로 수정 (CONDA_BASE / KISTAR_WS / MOUNT_MAP)

# ③ MoveIt 컨테이너(ros2_humble) 빌드 — kistar_ws + trac_ik(태그 2.0.1)

# ④ 시퀀스 라이브러리 빌드 (제어 PC 저장소, 필수!)
cd ~/Dual_Arm_Hand_Ctrl/ros2
colcon build --packages-select dual_arm_msgs sequence_client
source install/setup.bash
```

---

## 2. 런타임 시작 (매번, 터미널 순서대로)

> ⚠️ 순서 중요: **① RT 컨트롤러(SHM 생성) → ② ROS2 브리지 → ③ MoveIt → ④ 카메라 → ⑤ 파시니**

### 터미널 1 — RT 컨트롤러 (SHM 생성자, 제일 먼저) [Dual_Arm_Hand_Ctrl]
```bash
cd ~/Dual_Arm_Hand_Ctrl/build/test && sudo su
./Dual_Arm_Hand_Imp_Ctrl_V1_0        # 별칭: sudo su && shm
```
→ SHM `0x7951` 생성 + 1kHz RT 루프 (팔+손 실시간 제어)

### 터미널 2 — ROS2 브리지 + 시퀀스 arbiter [Dual_Arm_Hand_Ctrl]
```bash
source /opt/ros/humble/setup.bash
source ~/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash
ros2 launch trajectory_receiver control_pc.launch.py require_control:=true
```
> ⚠️ **`require_control:=true` 필수** (시나리오1은 제어권 강제). 별칭 `nd`는 기본 false라 그냥 쓰면 안 됨.
> 이 launch가 띄우는 것: shm_state_publisher(/franka·/hand·/paxini 상태) + q_target/hand 수신기 + **sequence_arbiter**

### 터미널 3 — MoveIt (IK/계획) [dex_ros]
```bash
ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py \
    joint_state_mode:=direct \
    robot_ip:=192.168.0.100 \
    use_rviz:=true
```
> 우리 로봇 실행이 `/move_action`·`/compute_ik` 을 씀 → move_group 필요.
> ⚠️ 확인: 우리 executor는 group `fr3_arm` 기준. 이 dual-arm MoveIt의 그룹/프레임명과 맞는지 점검 필요.

### 터미널 4 — 카메라 노드 [dex_ros]
```bash
ros2 launch franka_kistar_bringup realsense_front.launch.py \
    enable_color:=true enable_depth:=true pointcloud:=true
```
> ⚠️ **우리 파이프라인은 `/front_cam/front/aligned_depth_to_color/image_raw` 필요.**
> 이 launch에 depth-color 정렬이 포함되는지 확인, 없으면 `align_depth.enable:=true` 추가.

### 터미널 5 — 파시니 촉각 (Inhand/촉각용) [Dual_Arm_Hand_Ctrl]
```bash
paxini    # writer (손 무부하 상태로 — --calibrate 0점화) → /paxini/right/ft 발행
```

### 로봇 준비 (브라우저)
- **Franka Desk (172.16.0.1)** → 로봇 **파란불 + 조인트 잠금 해제 + FCI 활성**
- 에러/노란불이면 → **Recovery** 먼저

---

## 3. 시나리오1 실행 (터미널 6)

```bash
cd ~/GW/Topdown_Grasp_sja
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
source /opt/ros/humble/setup.bash                          # 또는 jazzy
source ~/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash        # sequence_client (필수!)

~/miniconda3/envs/grasp_fruit/bin/python scripts/run_scenario1.py \
    --camera_source ros \
    --execute_robot \
    --calibration configs/calibration/extrinsic_20260612_170053.json
```
프롬프트:
```
비디오 녹화? > no
Query> orange
→ ① Pick 제어권 획득 (seq=1 RUNNING)
   ② 카메라 캡처 → SAM3 검출 → PCA → 로봇 pick (물체 쥔 채 유지)
   ③ Pick 성공 → 제어권 반납 (seq=1 DONE) → Inhand(2) 차례
```
- `--place` 안 붙임 (시나리오1은 Pick 전용, 코드에서 강제 OFF)
- exit/quit = Pick 취소(abort, DONE 아님)

---

## 4. 검증 (터미널 7)

```bash
ros2 topic echo /sequence_state --qos-durability transient_local --qos-reliability reliable
```
→ **`seq_id=1, state=2(DONE), owner=0`** 나오면 성공 ("1 2 DONE 0" = Inhand 차례)

상태 흐름:
| 시점 | seq_id | state | owner |
|---|---|---|---|
| Pick 시작 | 1 | 1 RUNNING | 1 |
| Pick 종료 → Inhand | 1 | **2 DONE** | 0 |

---

## 실행 순서 요약 (치트시트)

```
[1회] clone + conda + 컨테이너 + sequence_client 빌드 + paths.yaml
  ↓ (매번, 순서대로)
T1: shm (RT, SHM 생성)
T2: control_pc.launch.py require_control:=true (ROS2 + arbiter)
T3: MoveIt (dual_fr3_kistar_moveit, robot_ip:=192.168.0.100)
T4: 카메라 (realsense_front + align_depth)
T5: paxini (촉각)
  ↓ 로봇 Desk unlock
T6: run_scenario1.py --camera_source ros --execute_robot
      Query> orange
  ↓
T7: /sequence_state 확인 (1 2 DONE 0)
```

---

## 5. 우리 코드가 쓰는 토픽 (제어 PC 인터페이스)

> grasp.py/step.py 를 새 제어 PC 토픽으로 변경함 (옛 토픽은 주석처리).

**발행 (우리 → 제어 PC):**
| 토픽 | 타입 | 내용 |
|---|---|---|
| `/franka/right/q_target` | Float64MultiArray[7] | 팔 관절 목표 [rad] |
| `/hand/right/q_target` | Float32MultiArray[16] | 손 목표 [count] |
| `/hand/right/cmd_servo` | Bool | 손 서보 on |

**구독 (제어 PC → 우리):**
| 토픽 | 타입 | 내용 |
|---|---|---|
| `/franka/right/joint_states` | JointState | 팔 7관절 현재값 (도달 확인용) |
| `/hand/right/joint_states` | JointState | 손 16관절 현재값 |
| `/paxini/right/ft` | Float32MultiArray[12] | 촉각 4손가락×3축 (BEST_EFFORT) |

---

## 6. 자주 하는 실수 / 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| 토픽 안 보임 | `ROS_DOMAIN_ID=9` 미설정 → 모든 터미널 export |
| 파시니 값 안 뜸 | BEST_EFFORT 필요: `--qos-reliability best_effort` / paxini writer 꺼짐 |
| aligned_depth 없음 | 카메라 launch에 `align_depth.enable:=true` |
| 제어권 잡혀도 로봇 안 움직임 | require_control=true인데 우리가 제어권 없이 보냄 → SequenceClient가 자동 처리 (정상이면 OK) |
| 로봇 명령 무시됨 | 토픽 이름 확인 (`/franka/right/q_target` 맞나) / move_group 그룹명 mismatch |
| 매 단계 멈춤(timeout) | joint_states 피드백 안 옴 → 구독 토픽/JointState 파싱 확인 |
| 손 안 쥠 | 서보 안 켜짐 → `/hand/right/cmd_servo` true 확인 |
| 재실행 시 바로 시작됨 | 이전 DONE latched 남음 → arbiter 재시작 (제어 PC 담당자) |

---

## 7. ⚠️ 아직 검증 안 된 것 (메인 PC 첫 실행 시 확인)

```
1. MoveIt 그룹/프레임명 — 우리 executor(fr3_arm) vs dual-arm MoveIt
2. q_target 스트리밍 — 우리는 목표 1회 발행 (제어PC가 클램프로 따라감, 부드럽지 않으면 연속발행 필요)
3. 서보 켜기 순서 — 기본 버전 (손 튀면 조정)
4. 카메라 align_depth 발행 여부
5. require_control 하에서 SequenceClient 제어권 획득 정상인지
```

문의: 제어 PC 담당자 / 상세: Dual_Arm_Hand_Ctrl/docs_dev/USAGE_GUIDE.md
