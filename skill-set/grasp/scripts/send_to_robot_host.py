#!/usr/bin/env python3
"""
robot_executor.py 를 **호스트에서 직접** 실행 (Docker 미사용 — 플랜 B).

기존 send_to_robot.py 는 `docker exec` 로 컨테이너 안에서 robot_executor 를 돌린다.
이 버전은 MoveIt 을 호스트에서 띄운 환경(dex_soldering kistar_ws + fr_ws 오버레이)에
맞춰, robot_executor 를 **시스템 python3(3.10, ROS Humble native)** 로 직접 실행한다.
(grasp_fruit conda(3.12)에서는 rclpy import 불가하므로 반드시 /usr/bin/python3 사용)

전제 — 별도 터미널에서 미리 실행돼 있어야 함:
  1. move_group :  (아래 3개 소싱 후) ros2 launch scripts/launch_moveit.py
  2. relay      :  /usr/bin/python3 scripts/franka_joint_state_relay.py
     소싱: /opt/ros/humble + fr_ws/install + dex_soldering kistar_ws/install

사용:
  # grasp only
  python scripts/send_to_robot_host.py --summary_json data/outputs/scene_topdown_summary.json
  # pick + place
  python scripts/send_to_robot_host.py --summary_json ... --place
  # base 이동 후 임시 (collision off)
  python scripts/send_to_robot_host.py --summary_json ... --disable_collision
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from utils.arm import PLACE_Z_DESCENT_M

# ── 호스트 MoveIt 환경 (플랜 B). 다른 머신이면 인자로 덮어쓰기 ──────────────────
DEFAULT_ROS_SETUP = "/opt/ros/humble/setup.bash"
DEFAULT_FRANKA_WS = "/home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/fr_ws/install/setup.bash"                                                  # franka_description 1.3.0
DEFAULT_KISTAR_WS = "/home/user/prime/ChanukHwang/RobotAgentSystem/tools/ros2/dex_ros/isaac-ros/kistar_ws/install/setup.bash"     # kistar 모델(빌드됨)
DEFAULT_PYTHON    = "/usr/bin/python3"   # ROS Humble native (3.10) — rclpy/moveit_msgs 동작
DEFAULT_DOMAIN_ID = 9


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary_json", required=True,
                   help="Path to topdown_summary.json")
    p.add_argument("--place", action="store_true",
                   help=f"pick+place 모드. 하강 거리는 arm.yaml ({PLACE_Z_DESCENT_M} m)")
    p.add_argument("--execute_mode", default="direct_franka_topic",
                   choices=["trajectory_forwarder", "direct_franka_topic"])
    p.add_argument("--speed_factor",    type=float, default=0.1)
    p.add_argument("--approach_offset", type=float, default=0.10)
    p.add_argument("--disable_collision", action="store_true",
                   help="MoveIt collision 검사 비활성화 (base 이동 후 임시)")
    # 호스트 환경 오버라이드
    p.add_argument("--ros_setup", default=DEFAULT_ROS_SETUP)
    p.add_argument("--franka_ws", default=DEFAULT_FRANKA_WS)
    p.add_argument("--kistar_ws", default=DEFAULT_KISTAR_WS)
    p.add_argument("--python",    default=DEFAULT_PYTHON)
    p.add_argument("--domain_id", type=int, default=DEFAULT_DOMAIN_ID)
    return p.parse_args()


def main():
    args = parse_args()
    mode = "place" if args.place else "grasp"

    summary = str(Path(args.summary_json).resolve())
    if not Path(summary).exists():
        print(f"[ERROR] summary_json 없음: {summary}")
        sys.exit(1)
    executor = str(SCRIPTS / "robot_executor.py")

    extra = (
        f"--mode {mode} "
        f"--execute_mode {args.execute_mode} "
        f"--speed_factor {args.speed_factor} "
        f"--approach_offset {args.approach_offset}"
    )
    if args.place:
        extra += f" --place_z_descent {PLACE_Z_DESCENT_M}"
    if args.disable_collision:
        extra += " --disable_collision"

    # conda(PYTHONPATH/PYTHONHOME 등) 오염을 제거한 뒤 ROS 3개 워크스페이스 소싱 →
    # 시스템 python3(3.10)로 robot_executor 실행. (호출자가 grasp_fruit conda 여도 안전)
    bash = (
        "unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER && "
        f"source {args.ros_setup} && "
        f"source {args.franka_ws} && "
        f"source {args.kistar_ws} && "
        f"export ROS_DOMAIN_ID={args.domain_id} && "
        "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && "
        "export ROS_LOCALHOST_ONLY=0 && "
        f"exec {args.python} {executor} --summary_json {summary} {extra}"
    )

    print(f"[send_to_robot_host] mode={mode}  (호스트 직접 실행, Docker 미사용)")
    print(f"  summary : {summary}")
    print(f"  python  : {args.python}")
    print(f"  kistar  : {args.kistar_ws}")

    rc = subprocess.run(["bash", "-c", bash]).returncode
    if rc == 0:
        print(f"\n[send_to_robot_host] {mode} 완료.")
    else:
        print(f"\n[send_to_robot_host] 종료 코드: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
