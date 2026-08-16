# tools/moveit — MoveIt 디지털 트윈 (move_group)

이 PC 가 **유일한 move_group 소유자**다. 구 분산환경에서는 산업부 PC 가 트윈을 띄웠고,
Current PC 는 절대 띄우지 않는 규칙이었다. 통합 후에도 규칙은 같다: **트윈은 정확히 1개.**

> `move_group` 이 2개면 `/move_action` 이 중복되어 **모든 arm move 가 실패**한다.
> place skill 서버는 preflight 에서 이를 검사하고 즉시 중단한다.

## 기동

```bash
tools/moveit/launch_twin.sh          # 이미 떠 있으면 거부한다
```

`run_fruit_demo.sh` 는 `--twin auto`(기본)로 move_group 이 없을 때만 이 launch 를 대신 띄운다.
이미 떠 있으면 그대로 재사용하고, 2개 이상이면 실행을 중단한다.

## launch 체인

```
dual_fr3_kistar_moveit.launch.py        (얇은 래퍼: use_camera=false, camera_view=true 강제)
  └─ dual_fr3_kistar_planning_pc_v2.launch.py   (실제 move_group + RViz + static TF + 테이블)
```

주요 인자 — `joint_state_mode:=direct`(실로봇 관절 추종), `robot_ip:=192.168.0.100`,
`use_rviz:=true`. 전체 목록은 v2 launch 파일의 `DeclareLaunchArgument` 참조:
[`.../franka_kistar_bringup/launch/dual_fr3_kistar_planning_pc_v2.launch.py`](../ros2/dex_ros/isaac-ros/kistar_ws/src/franka_kistar_bringup/launch/dual_fr3_kistar_planning_pc_v2.launch.py)

이 PC 는 실제 하드웨어를 잡지 않는다(`ros2_control:=false`, `use_fake_hardware:=true`).
실제 구동은 Control PC 담당이며, 이 트윈은 계획·충돌검사·시각화를 제공한다.

## 설정 패키지

`franka_kistar_moveit_config` — SRDF, kinematics, joint_limits, OMPL, controllers YAML.
