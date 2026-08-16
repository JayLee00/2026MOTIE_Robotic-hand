# IK 세팅 계획 — local dex_ros(kistar 모델) 활용

> MoveIt(move_group)으로 파지 좌표 → 팔 관절각 7개(IK) + 경로계획을 수행.
> 실제 구동은 제어 PC(q_target). grasp.py가 move_group의 `/compute_ik`·`/move_action` 호출.
> **원칙: 우리 것(`kistar_ws_sja` 복사본, `Topdown_Grasp`, Docker)만 수정. dex_ros 원본은 안 건드림.**

작성 기준: 2026-07-08. 현재 git baseline = 커밋 `2aacfa8` (arm.yaml=`fr3_arm`/`base`, 원본 launch/relay).

---

## ★ 실제 적용된 해법 (검증 완료 2026-07-08) — Docker/빌드 없이 호스트

> 아래 A/B 플랜을 검토하다, **dex_soldering/dex_ros의 기존 빌드**가 완전+일관(로봇명 fr3, base→fr3_link0,
> group fr3_arm)임을 확인 → **빌드도 Docker도 SRDF수정도 불필요**. 3개 워크스페이스 source 만으로 move_group 동작.

**환경 (source 3개, 빌드 없음):**
```bash
source /opt/ros/humble/setup.bash
source /home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/fr_ws/install/setup.bash                                              # franka_description 1.3.0
source /home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/dex_ros/isaac-ros/kistar_ws/install/setup.bash # kistar 모델(이미 빌드됨)
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
```

**우리 쪽 수정 (딱 3개 파일):**
1. `configs/arm.yaml`: `ref_frame: base → fr3_link0` (group fr3_arm, ee fr3_link8 는 baseline 유지)
   - `/compute_ik` 실측: base=응답없음, world=transform실패, **fr3_link0=성공**
2. `scripts/franka_joint_state_relay.py`: `/franka/right/joint_states`(name=fr3_joint1..7) → `/joint_states` QoS브리지
   - ⚠️ `/joint_states_r` 는 이름이 `fr3_r_joint1..`(접두사)라 URDF와 안 맞음 → 쓰면 안 됨
3. `scripts/send_to_robot_host.py`: robot_executor 를 `docker exec` 대신 **호스트 /usr/bin/python3(3.10)** 로 직접 실행

**실행 (터미널 3개):**
```bash
# T1: move_group (3개 소싱 후)
ros2 launch /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp/scripts/launch_moveit.py
# T2: 상태 relay (humble 소싱 후)
/usr/bin/python3 /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp/scripts/franka_joint_state_relay.py
# T3: 로봇 실행 (grasp summary 준비 후) — ⚠️ 시작 즉시 손 서보ON+init 자세 이동함, 로봇 안전상태 확인
python scripts/send_to_robot_host.py --summary_json data/outputs/<stem>_topdown_summary.json
```

**launch_moveit.py 는 baseline 그대로** (이미 franka_kistar_moveit_config·franka_kistar_description 참조 = dex_soldering 패키지명과 일치). **Docker 이미지/컨테이너/kistar_ws_sja 는 불필요** (정리 대상).

---

---

## 조사로 확인된 핵심 사실 (두 플랜 공통)

1. **폴더명 ≠ 패키지명**: 폴더 `franka_kistar_moveit_config` → 실제 `<name>franka_kistar_isaac_moveit_config`
2. **kistar URDF** (`fr3_kistar.urdf.xacro`): 로봇명 `fr3`, root 링크 **`world`**(≠`base`), 손(palm) 포함. `world→fr3_link0`은 identity(0,0,0) = MoveIt의 world가 곧 로봇 베이스.
3. **kistar 정적 SRDF(`fr3_kistar.srdf`)가 깨짐**: 로봇명 `fr3_kistar`·링크 `base`·palm을 팔그룹 끝으로 기대 → URDF와 불일치. (원본 dex_ros는 이 정적 SRDF 대신 franka semantic xacro로 SRDF를 생성해 씀)
4. **description CMakeLists**가 `robots/` 폴더를 install 안 함 → xacro가 `$(find ...)/robots/fr3/*.yaml` 참조하므로 install에 `robots` 추가 필요.
5. **franka_description 1.3.0**(`/home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/fr_ws`)이 kistar xacro와 호환 (매크로 인자 일치).
6. **제어 PC 상태 토픽**: `/joint_states_r`(팔7+손16=23, "RViz/MoveIt 호환용"), best_effort. move_group은 기본 reliable → QoS 브리지 필요.
7. move_group 서비스는 **루트 네임스페이스**(`/compute_ik`, `/move_action`, `/compute_cartesian_path`).
8. **프레임**: summary는 world 좌표. grasp.py `world_to_base()`로 베이스 좌표 변환 후 `frame_id=REF_FRAME`으로 전송. kistar는 base 링크가 없고 world가 그 역할 → `ref_frame: world`. **grasp.py 변환 로직 무변경.**

---

## 공통 코드 수정 (플랜 A·B 동일)

### C1. `kistar_ws_sja/src/franka_kistar_description/CMakeLists.txt`
```cmake
install(
  DIRECTORY urdf meshes robots     # ← robots 추가
  DESTINATION share/${PROJECT_NAME}
)
```

### C2. SRDF 수정 — `kistar_ws_sja/src/franka_kistar_moveit_config/config/fr3_kistar.srdf`
```
1. <robot name="fr3_kistar">          → <robot name="fr3">
2. virtual_joint(world_to_base, base) → 삭제 (URDF에 world→fr3_link0 이미 있음)
3. end_effector(kistar_ee, palm)      → 삭제 (palm이 팔그룹 밖 → 에러)
4. group fr3_manipulator (fr3_link0→fr3_link8) → 유지
(선택) 손 자충돌 방지 필요 시 disable_collisions 추가
```

### C3. `configs/arm.yaml`
```yaml
planning:
  group_name: fr3_manipulator   # kistar SRDF 그룹
  ee_link:    fr3_link8
  ref_frame:  world             # kistar URDF root (= 로봇 베이스)
```

### C4. `scripts/franka_joint_state_relay.py`
`/joint_states_r`(제어 PC 23관절, best_effort) 구독 → `/joint_states`(reliable) 발행. 메시지 통과(name/position 그대로), QoS만 변환.

### C5. `scripts/launch_moveit.py`
config 로드 패키지명 `franka_kistar_moveit_config` → **`franka_kistar_isaac_moveit_config`** (4곳: srdf/kinematics/joint_limits/ompl). URDF는 franka_kistar_description 그대로.

### C6. `configs/paths.yaml`
```yaml
__KISTAR_WS__: /home/cy/motie_ws/kistar_ws_sja
__MOUNT_MAP__: [["/home/cy", "/home/cy"]]
__DOCKER_CONTAINER__: ros2_humble   # (플랜 A만 의미 있음)
```

---

# 플랜 A — Docker 사용 (격리)

## 이미 확보
- 이미지 `grasp_fruit_moveit:latest` (humble+moveit+pick_ik+franka-description)
- 컨테이너 `ros2_humble` (`-v /home/cy:/home/cy`, network=host, domain 9)
- `kistar_ws_sja` 빌드됨

## 단계
1. **공통 수정 C1~C6** 적용
2. 컨테이너 안에서 재빌드:
   ```bash
   docker exec ros2_humble bash -c "source /opt/ros/humble/setup.bash && \
     cd /home/cy/motie_ws/kistar_ws_sja && colcon build"
   ```
3. move_group + relay 실행 (컨테이너 안):
   ```bash
   docker exec -d ros2_humble bash -c "source /opt/ros/humble/setup.bash && \
     source /home/cy/motie_ws/kistar_ws_sja/install/setup.bash && \
     export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 && \
     ros2 launch /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp/scripts/launch_moveit.py"
   docker exec -d ros2_humble bash -c "... && /usr/bin/python3 \
     /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp/scripts/franka_joint_state_relay.py"
   ```
4. **IK 테스트**: `/compute_ik`(fr3_manipulator, world 프레임) → 관절각 확인
5. **전체 연결**: grasp summary → `send_to_robot.py`(docker exec) → `robot_executor.py` → grasp.py IK + `/franka/right/q_target`

## 장단점
- ✅ 호스트/다른 사용자 환경 안 건드림 (격리)
- ✅ 원본 설계(docker_runner exec)와 그대로 호환
- ⚠️ 컨테이너·마운트·exec 레이어로 복잡, 디버깅 한 겹 더

---

# 플랜 B — Docker 없이 호스트에서 직접

호스트 ROS2 Humble(`/opt/ros/humble`)에 MoveIt을 설치해 move_group을 **호스트에서** 실행.

## 사전 준비
1. **호스트에 MoveIt 이미 설치돼 있음** (확인 2026-07-08: `ros-humble-moveit*` 28개,
   `moveit_ros_move_group`/`moveit_core`/`moveit_kinematics` + `move_group` 실행파일 존재).
   → **새로 설치할 것 없음. 전역 설치 단점 해당 없음** (이미 있는 걸 사용).
   franka_description 은 우리 kistar_ws_sja 안 1.3.0 사용.
2. **kistar_ws_sja 를 호스트에서 빌드**:
   ```bash
   source /opt/ros/humble/setup.bash
   cd /home/cy/motie_ws/kistar_ws_sja
   colcon build
   ```
   (dex_ros 원본 아님 — 우리 복사본. install/build는 kistar_ws_sja 안에만 생김)

## 단계
1. **공통 수정 C1~C5** 적용 (C6의 DOCKER_CONTAINER는 무시)
2. 호스트에서 kistar_ws_sja 빌드 (위)
3. **move_group + relay 실행** (호스트, 터미널 각각):
   ```bash
   # 터미널 A — move_group
   source /opt/ros/humble/setup.bash
   source /home/cy/motie_ws/kistar_ws_sja/install/setup.bash
   export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
   ros2 launch /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp/scripts/launch_moveit.py

   # 터미널 B — joint_states relay
   source /opt/ros/humble/setup.bash
   export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
   /usr/bin/python3 /home/user/prime/ChanukHwang/RobotAgentSystem/skill-set/grasp/scripts/franka_joint_state_relay.py
   ```
4. **IK 테스트**: `/compute_ik`(fr3_manipulator, world) → 관절각 확인
5. **robot_executor 호스트 실행 경로 추가** (핵심 차이):
   - 현재 `send_to_robot.py`는 `docker exec`로 컨테이너 안에서 실행 → **호스트 직접 실행 버전 필요**
   - 새 스크립트 `scripts/send_to_robot_host.py`(또는 `robot_executor` 직접 호출):
     ```bash
     source /opt/ros/humble/setup.bash
     source /home/cy/motie_ws/kistar_ws_sja/install/setup.bash
     export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
     /usr/bin/python3 scripts/robot_executor.py --summary_json <json> --mode grasp ...
     ```
   - 비전(grasp_fruit, py3.12)이 이 호스트 스크립트를 subprocess로 호출하도록 pipeline 연결
     (rclpy/moveit_msgs는 `/usr/bin/python3`=3.10 + humble에서 동작)

## 장단점
- ✅ 컨테이너 레이어 없음 → 단순, 디버깅 쉬움, `ros2 topic/service` 바로 접근
- ✅ py 버전 문제 없음 (호스트 py3.10 = Humble native)
- ⚠️ **`ros-humble-moveit`가 시스템 전역 설치** → 공유 머신·다른 사용자에 영향 (추가 설치라 보통 무해하나, 격리는 아님)
- ⚠️ `send_to_robot.py`(docker exec 전제)를 호스트 실행 버전으로 추가/수정 필요
- ⚠️ move_group을 호스트 domain 9에서 실행 → 다른 분이 같은 domain에서 move_group 돌리면 충돌 (Docker든 호스트든 동일 이슈)

---

## 두 플랜 비교 요약

| 항목 | A (Docker) | B (호스트) |
|---|---|---|
| 격리 (공유 머신 보호) | ✅ | ✅ (moveit 이미 설치됨 → 새 설치 0) |
| 단순함/디버깅 | △ (레이어 1겹) | ✅ (컨테이너 없음) |
| 코드 변경량 | 공통 C1~C6 | 공통 C1~C5 + send_to_robot 호스트 버전 |
| py 버전 이슈 | 없음(컨테이너 3.10) | 없음(`/usr/bin/python3` 3.10) |
| 원본 설계 호환 | ✅ (docker_runner) | 새 실행 경로 필요 |

> **결론(2026-07-08)**: moveit이 호스트에 이미 설치돼 있어 B의 "전역 설치" 단점이 사라짐.
> B가 더 단순하고 격리도 유지 → **플랜 B 권장.** 단 `send_to_robot.py`의 호스트 실행 경로만 추가하면 됨.

---

## 공통 리스크 / 미검증
- 손 충돌 모델 (kistar URDF에 손 포함 → 자충돌 가능, 필요시 SRDF disable_collisions)
- `/joint_states_r` 조인트 이름 ↔ URDF 23관절 이름 일치 여부 (relay 후 move_group 경고 확인)
- KDL 솔버 성공률 (특이자세 실패 시 pick_ik/trac_ik 전환)
- q_target 단발 vs 연속 스트리밍 (제어 PC 0.2rad/msg 클램프)
- move_group joint_states QoS 매칭 (relay가 reliable로 재발행하므로 OK 예상, 실측 확인)
