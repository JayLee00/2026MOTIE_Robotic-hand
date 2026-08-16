# 실행 명령어 정리 (현재 세팅 — dual 모델 / 호스트 / Docker 미사용)

> 이 PC(cy) 기준. MoveIt 은 dex_soldering dual 모델의 **right_arm** 사용, 실행은 호스트 py3.10.
> 비전(SAM3/파지계산)은 grasp_fruit(py3.12) subprocess, ROS(카메라·시퀀스·로봇)는 시스템 py3.10.

---

## 0. 공통 — 환경 소싱 (⚠️ 매 터미널)

```bash
conda deactivate          # ★필수★ which python3 → /usr/bin/python3 여야 함 (안 그러면 rclpy 에러)
source /opt/ros/humble/setup.bash
source /home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/fr_ws/install/setup.bash                                              # franka_description
source /home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/dex_ros/isaac-ros/kistar_ws/install/setup.bash # dual 모델(빌드됨)
source /home/cy/motie_ws/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash                   # sequence_client (시나리오1만)
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
cd /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp
```

> **제어 PC 쪽** (별도 PC): shm(RT) + control_pc.launch(require_control:=true) + 카메라 + Franka unlock/FCI.

---

## 1. 비전만 테스트 (로봇 X) — 카메라 → SAM3 → PCA → 좌표 PNG

2단계 (카메라는 py3.10, SAM3/파지는 grasp_fruit py3.12).

```bash
# ── 1) 카메라 → NPZ (ROS 소싱 상태) ──
/usr/bin/python3 scripts/ros_camera_grab.py --stem shot

# ── 2) SAM3 + PCA → PNG (grasp_fruit, ROS 오염 제거) ──
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u LD_LIBRARY_PATH \
  /home/user/miniconda3/envs/grasp_fruit/bin/python scripts/run_pipeline.py \
    --input data/raw/shot_000.npz \
    --query orange \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --overlay_name shot_result
```
**결과**: `data/outputs/shot_result.png` (x,y,z + PCA 화살표), `..._topdown_summary.json`
**옵션**: `--query`(물체명), `--overlay_name`(사진 이름), `--x_offset`/`--y_offset`(오프셋 임시 override)

---

## 2. MoveIt 띄우기 (dual 모델, right_arm) — 터미널 A

```bash
# (0번 소싱 후)
ros2 launch scripts/launch_moveit.py
#   RViz 없이:  ros2 launch scripts/launch_moveit.py use_rviz:=false
#   → "You can start planning now!" 뜨면 OK
```
- move_group + joint_state_merger + robot_state_publisher + 테이블/scene/카메라 TF + RViz
- 손 노드 제외, 오른팔 trajectory_bridge

## 3. 상태 relay (선택 — merger가 이미 /joint_states_relay 제공하므로 보통 불필요)

```bash
/usr/bin/python3 scripts/franka_joint_state_relay.py
```

---

## 4. 시나리오1 전체 (Pick + 시퀀스 handoff) — 터미널 B ★메인★

> 사전: 터미널 A의 move_group 실행 중 + 제어 PC 준비 + 로봇 unlock.

```bash
# (0번 소싱 — Dual_Arm 소싱 포함)
/usr/bin/python3 scripts/run_scenario1_host.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot \
    --overlay_name real_robot_1
#   Query> orange   → 제어권 획득 → 카메라 → SAM3 → 파지 → 로봇 pick(쥔 채 유지) → DONE → Inhand 차례
#   exit/quit/q     → Pick 취소(abort)
```
**자주 쓰는 옵션**:
| 옵션 | 용도 |
|---|---|
| `--execute_robot` | 로봇 실행 (없으면 비전+파지계산까지만) |
| `--disable_collision` | 테이블 충돌박스 무시 (파지 하강이 collision으로 막힐 때) |
| `--overlay_name NAME` | 파지 사진 이름 (data/outputs/NAME.png) |
| `--speed_factor 0.05` | 더 느리게 (기본 0.1) |

## 5. 로봇 단독 실행 (summary JSON 이미 있을 때)

```bash
/usr/bin/python3 scripts/send_to_robot_host.py \
    --summary_json data/outputs/shot_result_topdown_summary.json \
    --execute_robot            # 실제로는 파일 있으면 바로 실행
#   pick+place: --place    /   collision off: --disable_collision
```

---

## 6. 검증 (터미널 C)

```bash
# 시퀀스 상태 (Pick DONE 확인)
ros2 topic echo /sequence_state --qos-durability transient_local --qos-reliability reliable
#   → seq_id=1, state=2(DONE), owner=0  이면 성공 (Inhand 차례)

# 팔 실제 움직이나
ros2 topic echo /franka/right/joint_states --field position

# move_group 서비스 확인
ros2 service list | grep compute_ik
```

---

## 7. 파지 튜닝 (설정 파일)

| 파일 | 파라미터 | 용도 |
|---|---|---|
| `configs/arm.yaml` | `ee_correction.yaw_deg` | 손 회전(어느 양옆 잡나) |
| | `ee_correction.x/y_offset_m` | 손바닥 XY 이동 (접촉 위치) |
| | `grasp_z_offset_m` | 파지 깊이 |
| | `pointcloud.top_z_pct` | 중심 계산 상위 Z% (기본 30) |
| `configs/hand.yaml` | `abduction_deg` / `finger_bend_deg` | 손가락 벌림/굽힘 |
| `configs/fruits.yaml` | 물체별 override | orange/lemon 등 개별 오프셋 |

---

## 8. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `No module named 'rclpy'` | conda 켜져 있음 → **`conda deactivate`** (which python3 = /usr/bin/python3) |
| `grasp_fruit` import 깨짐 | grasp_fruit 실행 시 `env -u PYTHONPATH -u AMENT_PREFIX_PATH -u LD_LIBRARY_PATH` 안 붙임 |
| move_group 없음 / `/compute_ik` 없음 | 터미널 A의 `launch_moveit.py` 안 떠 있음 |
| `Cartesian NN% < 90%` | 1순위 직선계획 실패 → OMPL 폴백 (정상). OMPL OK면 진행. 자꾸 뜨면 `--disable_collision` or 물체를 로봇 쪽으로 |
| `IK failed code=-31` / `collision scene이 차단` | 테이블 충돌박스가 파지 막음 → **`--disable_collision`** |
| RViz `model 'right_arm'... 'dual_fr3_kistar' expected` | (해결됨) grasp.py disp.model_id='' |
| 카메라 프레임 수신 실패 | 제어 PC 카메라 미실행 / ROS_DOMAIN_ID 불일치 |
| move_group 중복 / RViz 충돌 | 다른 분 dual launch와 동시 실행 → 하나만 (또는 namespace:=) |

---

## 파일 위치 요약
```
scripts/ros_camera_grab.py       카메라 → NPZ (py3.10)
scripts/run_pipeline.py          비전 오프라인 (grasp_fruit)
scripts/run_topdown_grasp.py     파지 좌표 계산 (grasp_fruit)
scripts/launch_moveit.py         MoveIt dual/right_arm (py3.10)
scripts/franka_joint_state_relay.py  상태 relay
scripts/run_scenario1_host.py    시나리오1 오케스트레이터 (py3.10)
scripts/send_to_robot_host.py    로봇 단독 실행 (py3.10)
configs/{arm,hand,fruits}.yaml   파지 튜닝






#
ros2 launch scripts/launch_moveit.py

#
 /usr/bin/python3 /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp/scripts/franka_joint_state_relay.py

## NO GUI
/usr/bin/python3 scripts/run_scenario1_host.py     --calibration configs/calibration/extrinsic_20260612_170053.json     --execute_robot

## GUI
/usr/bin/python3 scripts/run_scenario1_host.py     --calibration configs/calibration/extrinsic_20260612_170053.json     --execute_robot     --gui

/usr/bin/python3 scripts/run_scenario1_host.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot --gui --yes

## QWEN (자연어 지시로 자동 선택 — SAM3 후보 → Qwen2-VL 판단)
/usr/bin/python3 scripts/run_scenario1_host.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot \
    --instruction "pick the kiwi on the dark table"
#   Query> kiwi  → 캡처 → SAM3 → Qwen 선택(결과 창 Enter) → 파지 → 로봇 pick

## QWEN 무정지 (한 줄로 처음부터 끝까지 — 프롬프트/Enter/y/n 전부 생략)
/usr/bin/python3 scripts/run_scenario1_host.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot \
    --query kiwi \
    --instruction "pick the kiwi on the dark table" \
    --yes
#   실패 시 즉시 abort 종료 (재입력 루프 없음)

## 초기자세복귀
/usr/bin/python3 scripts/go_home.py

source ~/.bashrc
gohome


# 사진
/usr/bin/python3 scripts/ros_camera_grab.py --stem name_here





# UPDATE
ros2 launch scripts/launch_moveit.py

/usr/bin/python3 scripts/franka_joint_state_relay.py

/usr/bin/python3 scripts/run_scenario1_host.py    
 --calibration configs/calibration/extrinsic_20260612_170053.json     --execute_robot --query kiwi     --instruction "pick the kiwi on the dark table" --yes     --disable_collision



## 시작할때
conda deactivate          # ★필수★ (ⓔ base 없어져야 함)
source /opt/ros/humble/setup.bash
source /home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/fr_ws/install/setup.bash
source /home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/dex_ros/isaac-ros/kistar_ws/install/setup.bash
source /home/cy/motie_ws/Dual_Arm_Hand_Ctrl/ros2/install/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
cd /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp


 cy  ~   ros2 launch franka_kistar_bringup realsense_front.launch.py 


 cy  ~   ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py     joint_state_mode:=direct     robot_ip:=192.168.0.100     use_rviz:=true
