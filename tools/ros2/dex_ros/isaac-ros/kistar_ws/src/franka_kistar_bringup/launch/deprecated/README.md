# `launch/deprecated/` — 보관용(deprecated) launch 파일

여기 있는 launch 파일들은 **현재 워크플로우에서 사용되지 않는 레거시**입니다.
2026-07-14에 `launch/` 최상위에서 `git mv`로 이동했습니다(삭제가 아니라 보관 — git 히스토리 보존).

현행(active) launch는 상위 `../` 에 있는 4개뿐입니다:
`dual_fr3_kistar_planning_pc_v2` · `dual_fr3_kistar_moveit` · `dual_fr3_kistar_all` · `realsense_front`.

전체 배경/역할/의존 관계는 [`../README.md`](../README.md) 참고.

---

## 목록

| 파일 | 계열 | 내용 |
|---|---|---|
| `dual_fr3_kistar_planning_pc.launch.py` | dual v1 | 양팔 MoveIt 플래닝 v1(non-bracket) — `_v2`로 대체됨 |
| `fr3_kistar.launch.py` | single | FR3+KISTAR 기본 bringup(ros2_control + RViz, MoveIt 無) |
| `fr3_default.launch.py` | single | Franka 제공 demo 개작 — 저장소 참조 0 |
| `fr3_kistar_moveit_real.launch.py` | single | MoveIt + JointTrajectoryController 실로봇 실행 |
| `fr3_kistar_moveit_bringup.launch.py` | single | MoveIt + Shared-Memory 브릿지(P2P 모션) |
| `fr3_kistar_moveit_planning_pc.launch.py` | single(분산) | 플래닝 PC — trajectory topic publish |
| `fr3_interactive_pose_control.launch.py` | single | CUI EE-pose 입력 데모 (위 planning_pc include) |
| `moveit_planning_pc.launch.py` | single(분산) | 범용 MoveIt 플래닝 PC |
| `robot_execution_pc.launch.py` | single(분산) | 실행측 PC (ros2_control + trajectory_subscriber) |
| `realsense_multi.launch.py` | camera | 다중 RealSense 헬퍼 — `realsense_front`로 대체됨 |

---

## ⚠️ 되살리기(revive) 시 주의

이 파일들은 **패키지에 설치되지 않습니다.** `setup.py`의 설치 규칙이
`glob('launch/*.launch.py')`(비재귀)라서 `deprecated/` 하위는 install 공간에 복사되지 않습니다.
따라서 `ros2 launch franka_kistar_bringup <파일>` (짧은 이름) 은 동작하지 않습니다.

다시 사용하려면:
1. `setup.py`의 data_files에 `launch/deprecated/*.launch.py` 설치 규칙을 추가하고,
2. 아래 **내부 상호 include** 경로에 `deprecated`를 삽입해야 합니다
   (현재는 `FindPackageShare("franka_kistar_bringup")/launch/<file>` 로 install 공간을 가리킴):
   - `fr3_interactive_pose_control.launch.py` → `fr3_kistar_moveit_planning_pc.launch.py`
   - `fr3_kistar_moveit_real.launch.py` → `realsense_multi.launch.py`

## 이 이동에 맞춰 갱신된 툴링 (v1 참조)

- `test/test_validator.py` — `LAUNCH_PY` 경로를 `launch/deprecated/`로 변경(테스트 통과 확인)
- `scripts/lint_launch_args.sh` — 기본 `LAUNCH_FILE`을 `launch/deprecated/`로 변경(lint OK 확인)
- `scripts/measure_launch_time.py` — docstring을 현행 `_v2` 기준으로 갱신
