# 1차 목표 실로봇 테스트 절차 (사용자 실행용)

> 목표: Control PC에서 SHM/통신/제어 준비 + 오렌지 파지 상태에서, **Current PC의 단일 명령어 하나**로
> parent_pose → child_pose → **T_act 실행 → release & retract**까지.
>
> ⚠ 안전: 첫 팔 동작은 **속도 스케일 낮게**, E-stop 손 위에. 배치 동작은 파지 물체 충돌 미고려(§9-H)라 여유·저속.
> ⚠ 약어: **[제어]**=Control PC, **[현재]**=Current PC host(conda), **[현재/도커]**=Current PC `docker/run.sh` 컨테이너 안.
>
> **아키텍처 메모(중요):** front_cam은 이제 **산업부 PC**에 연결되어 ROS2 domain 9로 `/front_cam/front/...`를 계속
> 발행한다. 오케스트레이터(컨테이너, domain 9, `--network host`)가 그 토픽을 **직접 subscribe**한다 —
> host 캡처 서비스 불필요. ⚠ 산업부 PC의 realsense는 **`align_depth.enable:=true`**로 띄워
> `/front_cam/front/aligned_depth_to_color/image_raw`가 나와야 한다(color 픽셀 마스크가 depth를 1:1 인덱싱).
> 카메라→world는 **오프라인 extrinsic(tf.txt·URDF, 카메라가 base에 고정이라 상수)** 사용.
>
> **런타임 인터페이스 메모:** 팔 구동은 MoveIt→`/fr3_r_arm_controller` FollowJointTrajectory→trajectory_receiver→SHM
> (런타임 네임스페이스 개편에서 **변경 없음**). 이 경로는 신규 **sequence 제어권 게이트**(`/franka/{side}/q_target`
> 스트리밍용)의 영향을 받지 **않는다**. hand q는 `/joint_states_relay`(merger가 `/joint_states_r|l`→`right_` remap +
> counts→rad, **변경 없음**)에서 읽는다.

---

## 실행 순서

1. **[제어]** Dual_Arm RT 런타임 — 터미널 1 (SHM 0x7951 생성)
   * 런타임 교체(2026-07-05): 공동 런타임 /home/prime/Dual_Arm_Hand_Ctrl 사용 (DUAL_ARM_HAND_CTRL.md)
   * (크기 다른 세그먼트 잔재 시: 관련 프로세스 종료 후  ipcrm -M 0x7951)
   ```bash
   cd /home/prime/Dual_Arm_Hand_Ctrl
   sudo FRANKA_ARM_R_IP=172.16.0.1 FRANKA_ARM_L_IP=172.17.0.1 \
     ./build/test/Dual_Arm_Hand_Imp_Ctrl_V1_0 enp1s0f0 enp1s0f1
   ```
2. **[제어]** trajectory_receiver(ROS2 브리지) — 터미널 2
   ```bash
   cd /home/prime/Dual_Arm_Hand_Ctrl/ros2
   source /opt/ros/humble/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=9
   ros2 launch trajectory_receiver dual_arm.launch.py
   ```
3. **[현재/도커]** 디지털트윈(**direct** = 실로봇 추종) — 터미널 A
   ```bash
   cd ~/place-object && docker/run.sh
   ```
   ```bash
   ros2 launch franka_kistar_bringup dual_fr3_kistar_planning_pc_v2.launch.py \
       joint_state_mode:=direct use_rviz:=true strict_urdf_snapshot:=false
   ```
3b. **[산업부 PC]** front RealSense 발행 (ROS2 domain 9) — **이미 발행 중이면 생략**.
   먼저 현재 PC에서 확인: `ros2 topic hz /front_cam/front/aligned_depth_to_color/image_raw`
   → ~15Hz로 나오면(=산업부 PC가 이미 align_depth로 띄워둠) 이 단계 **불필요**. 안 나올 때만:
   ```bash
   export ROS_DOMAIN_ID=9 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   ros2 launch franka_kistar_bringup realsense_front.launch.py align_depth.enable:=true
   ```
4. **[현재]** 모델 서비스 기동 — 터미널 B (계속 떠 있음; Molmo 로딩 ~1분, ~30GB VRAM). 카메라는 서비스가 아니라
   ROS 토픽이므로 여기서 안 뜬다.
  * grid 서비스(:8815)가 뜨면서 **이미지 그리드 브라우저 탭이 자동으로 열립니다**(비어 있다가 실행 중 채워짐).
   ```bash
   cd ~/place-object && bash vision_pipeline/run_services.sh
   ```
5. **[제어]** Hand GUI로 **물체(오렌지) 파지** — 터미널 3 (파지 상태 유지)
   ```bash
   cd ~/place-object/kistar_hand_gui && python3 kistar_hand_gui.py
   ```
6. **[현재/도커]** **단일 명령어** — 터미널 C
   ```bash
   cd ~/place-object && docker/run.sh
   ```
   ```bash
   bash -c \
     'python3 -u -m vision_pipeline.run scenario=fruit hand_pc=true' \
     2>&1 | tee vision_pipeline/test_logs/D_run.log
   ```

---

# 2차 목표 실로봇 테스트 절차 (Release & Retract, 사용자 실행용)

> 목표: 1차(파지→T_act 도달)에 이어 **T_act 도달 후 물체를 놓고(Release) 팔을 복귀(Retract)**까지, **단일 명령어 하나**로.
> 1차 대비 바뀌는 곳은 **② 브리지 런치(control_pc)**, **②b PaXini writer 추가**, **⑤ GUI 주의** 3곳뿐(표시함). 나머지는 동일.
>
> ⚠ 안전: 첫 팔 동작은 **속도 스케일 낮게**, E-stop 손 위에. 배치·하강은 파지 물체 충돌 미고려(§9-H)라 여유·저속.
> Release는 손을 **Voltage 모드**로 전환하므로 손 주변 안전 확보.

## 실행 순서

1. **[제어]** Dual_Arm RT 런타임 — 터미널 1 (SHM 0x7951 생성)
   * (크기 다른 세그먼트 잔재 시: 관련 프로세스 종료 후  ipcrm -M 0x7951)
   ```bash
   cd /home/prime/Dual_Arm_Hand_Ctrl
   sudo FRANKA_ARM_R_IP=172.16.0.1 FRANKA_ARM_L_IP=172.17.0.1 \
     ./build/test/Dual_Arm_Hand_Imp_Ctrl_V1_0 enp1s0f0 enp1s0f1
   ```
2. **[제어]** trajectory_receiver(ROS2 브리지) — 터미널 2. ⚠ **1차와 다름**: `dual_arm.launch.py`(팔만) 대신
   **`control_pc.launch.py`** — 기존 팔 경로(`/fr3_r_arm_controller` + `/joint_states_r`)를 그대로 포함하는 상위집합이며,
   추가로 `shm_state_publisher`(→ `/paxini/right/ft`) + `hand_target_receiver`(→ `/hand/right/{cmd_mode,cmd_servo,q_target}`)를
   띄운다. 기본값 `require_control:=false`라 손 q_target이 게이트되지 않는다.
   ```bash
   cd /home/prime/Dual_Arm_Hand_Ctrl/ros2
   source /opt/ros/humble/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=9
   ros2 launch trajectory_receiver control_pc.launch.py
   ```
2b. **[제어]** PaXini writer — 터미널 4. ⚠ **1차엔 없던 단계.** 이게 있어야 `/paxini/right/ft`에 실제 힘값이 나온다.
   없거나 값이 0이면 파이프라인이 **"tactile triggers DISABLED" 를 로그로 크게 알리고** T_act(Case 2)에서만 release한다
   (조용히 넘어가지 않음). ⚠ **이 writer는 시작 순간을 무조건 무부하 0점으로 tare 한다**(별도 인자 불필요 — 보정이 기본,
   건너뛰려면 `--no-calibrate`. `--calibrate`는 인자로 없으니 붙이면 에러). no-load(Retract) 판정이 이 0점에 의존하므로
   **Step 5 파지 전, 손끝 무접촉 상태**에서 실행할 것 (파지 후 실행하면 파지력이 0으로 tare되어 liveness 검사에서 걸리고
   트리거가 비활성된다).
   ```bash
   cd /home/prime/Dual_Arm_Hand_Ctrl
   python3 ./tools/paxini_writer.py --hand r
   ```
3. **[현재/도커]** 디지털트윈(**direct** = 실로봇 추종) — 터미널 A
   ```bash
   cd ~/place-object && docker/run.sh
   ```
   ```bash
   ros2 launch franka_kistar_bringup dual_fr3_kistar_planning_pc_v2.launch.py \
       joint_state_mode:=direct use_rviz:=true strict_urdf_snapshot:=false
   ```
3b. **[산업부 PC]** front RealSense 발행 (ROS2 domain 9) — **이미 발행 중이면 생략**.
   먼저 현재 PC에서 확인: `ros2 topic hz /front_cam/front/aligned_depth_to_color/image_raw`
   → ~15Hz로 나오면(=산업부 PC가 이미 align_depth로 띄워둠) 이 단계 **불필요**. 안 나올 때만:
   ```bash
   export ROS_DOMAIN_ID=9 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   ros2 launch franka_kistar_bringup realsense_front.launch.py align_depth.enable:=true
   ```
4. **[현재]** 모델 서비스 기동 — 터미널 B (계속 떠 있음; Molmo 로딩 ~1분, ~30GB VRAM). 카메라는 서비스가 아니라
   ROS 토픽이므로 여기서 안 뜬다.
  * grid 서비스(:8815)가 뜨면서 **이미지 그리드 브라우저 탭이 자동으로 열립니다**(비어 있다가 실행 중 채워짐).
   ```bash
   cd ~/place-object && bash vision_pipeline/run_services.sh
   ```
5. **[제어]** Hand GUI로 **물체(오렌지) 파지** — 터미널 3 (파지 상태 유지). ⚠ **파지 성립 후 GUI를 만지지 말 것**:
   Release 때 파이프라인이 손을 Voltage로 전환하는데, 그 시점에 GUI가 Position target(counts)을 다시 쓰면 duty로
   재해석되어 **폭주** 위험(파지 target은 SHM에 유지되므로 방치하면 됨).
   ```bash
   cd ~/place-object/kistar_hand_gui && python3 kistar_hand_gui.py
   ```
6. **[현재/도커]** **단일 명령어** — 터미널 C. T_act 도달 후 release+retract까지 자동 진행된다.
   ```bash
   cd ~/place-object && docker/run.sh
   ```
   ```bash
   bash -c \
     'python3 -u -m vision_pipeline.run scenario=fruit hand_pc=true' \
     2>&1 | tee vision_pipeline/test_logs/D_run.log
   ```