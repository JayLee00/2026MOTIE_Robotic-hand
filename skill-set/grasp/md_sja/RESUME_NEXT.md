# 이어서 하기 (RESUME) — Topdown_Grasp + 파시니 촉각

> 마지막 세션까지의 상태와, 바로 이어서 할 명령어 모음. 위에서부터 순서대로.

---

## A. 현재 상태 (한눈에)

| 항목 | 상태 | 비고 |
|---|---|---|
| 비전 (카메라→SAM3→PCA) | ✅ 동작 | orange는 "brown ball"로도 검출됨 |
| 파시니 촉각 데이터 | ✅ 수신 확인 | Franka PC에서, 누르면 값 변함 |
| 절전 모드 | ✅ 차단됨 | 다시 안 잠 |
| 카메라 USB -71 | ⚠️ 하드웨어 의심 | 케이블 바꿔도 지속 → 포트/카메라 점검 필요 |
| 네트워크 (kist↔Franka) | ⚠️ 미완 | subnet 불일치, kist를 0동네로 바꿔야 |
| git 커밋 | ⚠️ 안 됨 | orange 설정, x_offset 기능 미커밋 |
| orange PCA 끄기 | ⏳ 미구현 | 둥근 물체 각도 튐 해결책 |
| 촉각 파이프라인 통합 | ⏳ 다음 단계 | 접촉 감지부터 |

---

## B. 시스템 켜는 순서 (매번 처음부터)

### B-1. 카메라 (RealSense) — kist PC
```bash
# 안 되면(errno=5) 리셋 먼저
/home/kist/rs_env/bin/python -c "import pyrealsense2 as rs,time; rs.context().query_devices()[0].hardware_reset(); time.sleep(8); print('reset done')"
# 사진 테스트
cd /home/kist/GW/Topdown_Grasp
/home/kist/rs_env/bin/python scripts/capture_realsense_once.py --stem test
# 확인: data/raw/test_000_rgb.png
```

### B-2. 네트워크 (⚠️ 아직 안 고침 — 로봇/촉각을 kist에서 쓰려면 필수)
문제: kist=192.168.82.50, Franka PC=192.168.0.100 (다른 동네)
해결: **kist PC의 USB Ethernet을 192.168.0.50으로** (Franka는 공용이라 못 건드림)
```
kist PC: 설정 → Network → USB Ethernet 톱니 → IPv4 → Manual
  Address: 192.168.0.50 / Netmask: 255.255.255.0 → Apply → 껐다 켜기
확인: ping -c 3 192.168.0.100
```

### B-3. 로봇 스택 (MoveIt) — kist PC
```bash
docker restart ros2_humble    # 절전 후 죽었으면
docker exec -it ros2_humble bash -c "
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
  export PATH=/usr/sbin:/usr/bin:/sbin:/bin:/opt/ros/humble/bin
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  source /opt/ros/humble/setup.bash
  source /home/kist/GW/Dex_ROS_GWB/isaac-ros/kistar_ws/install/setup.bash
  ros2 launch franka_kistar_bringup fr3_interactive_pose_control.launch_GWB.py \
    gui:=false use_fake_joint_states:=false execute_mode:=direct_franka_topic reference_frame:=base
"
# gui:=false 중요 (gui:=true면 RViz가 GPU로 죽음). Ctrl+C 하지 말 것.
# 확인(새 터미널): export ROS_DOMAIN_ID=9; source /opt/ros/jazzy/setup.bash; ros2 topic list | grep -E "franka|hand"
```

### B-4. Franka PC 로봇 브리지 (Franka PC에서)
```bash
export ROS_DOMAIN_ID=9; export ROS_LOCALHOST_ONLY=0
ros2 launch franka_kistar_bringup robot_execution_pc.launch.py robot_ip:=172.16.0.1
# + Franka Desk(172.16.0.1): 로봇 파란불/잠금해제/에러없음(에러면 Recovery)
```

### B-5. 파시니 촉각 (Franka PC에서, 터미널 3개)
```bash
# 터미널1: writer (무부하 상태로 — --calibrate가 0점화함)
paxini
#   → 값 나오는거 확인 후 그대로 둠
# 터미널2: SHM→ROS 브리지
nd2
# 터미널3: 확인 (BEST_EFFORT 필수!)
ros2 topic echo /paxini/ft_r --qos-reliability best_effort
#   → 안 나오면: pkill -f paxini_writer; pkill -f shm_state_publisher; ps로 확인 후 paxini→nd2 순서로 다시
#   → ipcs -m 에서 0x7951 nattch=2 면 정상 연결
# /paxini/ft_r = Float32MultiArray [4손가락][3축]=12개
```

---

## C. 작업별 이어서 하기

### C-1. 비전만 테스트 (로봇 없이, 네트워크 불필요)
```bash
cd /home/kist/GW/Topdown_Grasp
/home/kist/miniconda3/envs/grasp_fruit/bin/python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json
# Query> orange   (안 잡히면 brown ball / orange fruit)
# → PCA(alpha=..), pose 출력. 결과: data/outputs/interactive_XXX_topdown_*.png/json
# 로봇까지: 위에 --execute_robot --place --disable_collision 추가 + z/x offset:
#   --z_offset 0.16 --x_offset 0.04
```

### C-2. git 커밋 (지금까지 작업 저장)
```bash
cd /home/kist/GW/Topdown_Grasp
git status   # 변경: fruits.yaml, paths.yaml, pipeline_core.py, run_topdown_grasp.py
# ⚠️ paths.yaml은 머신전용 → 커밋 제외 권장:
git add configs/fruits.yaml scripts/pipeline_core.py scripts/run_topdown_grasp.py
git commit -m "orange offset + --x_offset/--y_offset CLI"
```

### C-3. orange PCA 끄기 (둥근 물체 각도 튐 해결) — 미구현
계획: fruits.yaml에 `use_pca_yaw: false` 지원 추가.
- `run_topdown_grasp.py` build_topdown_pose에서 fruits override에 use_pca_yaw 읽기
- false면 `alpha_final = 0.0` (PCA 회전 무시, 고정 -45°만)
- 위치: build_topdown_pose ~L140 (alpha_final 계산 직후) + fruits.yaml 블록 ~L437

### C-4. 촉각 파이프라인 통합 (다음 큰 단계)
전제: **B-2 네트워크 먼저** (kist가 Franka의 /paxini/ft_r 받아야).
로드맵:
1. **데이터 이해** — /paxini/ft_r 12개가 어느 손가락/축인지, baseline vs 접촉값 매핑 (라이브 모니터 스크립트)
2. **접촉 감지** — 힘>임계값 = 닿음. grasp.py에 `/paxini/ft_r` 구독(BEST_EFFORT) 추가
3. **파지 성공/실패 판정** — 쥔 후 접촉 없으면 헛잡음 → 재시도
4. **적응형 쥐기** — 고정 90° 대신 닿을때까지 (hand.yaml finger_bend 고정 → 촉각조건)
5. **slip 감지** — 들 때 힘 급감 → 더 세게
통합 지점: `scripts/utils/grasp.py`의 grasp/lift 단계 + `_setup_ros`에 파시니 구독 추가.

---

## D. 알려진 문제 & 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| 카메라 errno=5 / -71 | USB 신호(포트/카메라 HW) | 리셋(B-1) → 안되면 파란 3.0포트 재연결 → HW점검 |
| ping 안 됨 | subnet 불일치 | B-2 (kist를 0.50으로) |
| ros2 topic 텅 빔 | ROS_DOMAIN_ID 안 맞음 | export ROS_DOMAIN_ID=9 |
| 토픽 있는데 node list 빔 | daemon 캐시 | ros2 daemon stop |
| dual_fr3_kistar 충돌/RViz죽음 | 좀비노드 / GPU | docker restart + gui:=false |
| 파시니 값 0.0 | writer 없거나 순서꼬임 | paxini 먼저→nd2, pkill 후 재시작 |
| 파시니 echo 값 안뜸 | QoS RELIABLE | --qos-reliability best_effort |
| PC가 잠 | 절전 | 이미 masked됨 (systemctl) |

---

## E. 핵심 경로 / 명령 요약
```
비전 python:  /home/kist/miniconda3/envs/grasp_fruit/bin/python
카메라 python: /home/kist/rs_env/bin/python
캘리브:       configs/calibration/extrinsic_20260612_170053.json
로봇 WS:      /home/kist/GW/Dex_ROS_GWB/isaac-ros/kistar_ws
컨테이너:     ros2_humble (Humble), 호스트=Jazzy(모니터링)
파시니:       Franka PC(prime@prime) ~/Dual_Arm_Hand_Ctrl, `paxini`+`nd2`
파시니 토픽:  /paxini/ft_r (Float32MultiArray, BEST_EFFORT, [4][3])
설정 파일:    configs/{arm,hand,fruits,paths}.yaml
```
