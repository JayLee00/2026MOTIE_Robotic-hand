# -----------------------------
# dual_fr3_kistar_moveit.launch.py
# MoveIt-ONLY entry point (split workflow, step 1 of 2).
#
# Thin wrapper around dual_fr3_kistar_planning_pc_v2.launch.py that:
#   - NEVER starts the RealSense driver (use_camera is forced to false), so
#     this launch + RViz stay light;
#   - forces camera_view:=true, so RViz uses fr3_kistar_camera.rviz. The
#     FrontRGB/FrontDepth/FrontCloud displays sit idle (near-zero cost) until
#     the camera is launched separately — then RViz simply renders the streams.
#
# Companion launches:
#   step 2 (camera, separate terminal/process):
#     ros2 launch franka_kistar_bringup realsense_front.launch.py
#   all-in-one convenience:
#     ros2 launch franka_kistar_bringup dual_fr3_kistar_all.launch.py
#
# All CLI args are forwarded verbatim to the v2 launch (joint_state_mode,
# robot_ip, use_rviz, front_camera_*, table_*, ...) — see the v2 file for the
# full list. Passing use_camera/camera_view here is overridden by this wrapper.
#
# Usage:
#   ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py \
#       joint_state_mode:=direct robot_ip:=192.168.0.100 use_rviz:=true
#
#   GUI sliders -> REAL fingers (fake mode + joint_state_publisher_gui +
#   hand_gui_bridge; needs ROS_DOMAIN_ID=9 and the robot RT loop):
#   ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py \
#       joint_state_mode:=fake use_joint_state_gui:=true real_hand:=true hand_side:=right
# -----------------------------

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

# Wrapper-enforced overrides. camera_view keeps the camera displays in RViz so a
# separately-launched realsense_front.launch.py shows up without an RViz restart.
_OVERRIDES = {
    "use_camera": "false",
    "camera_view": "true",
}


def _include_v2(context, *args, **kwargs):
    # Forward every CLI arg the operator passed (context.launch_configurations
    # holds exactly those), then apply the wrapper overrides on top.
    forwarded = dict(context.launch_configurations)
    dropped = {k: forwarded[k] for k in _OVERRIDES if k in forwarded and forwarded[k] != _OVERRIDES[k]}
    forwarded.update(_OVERRIDES)

    actions = []
    if dropped:
        actions.append(
            LogInfo(
                msg=(
                    "[dual_fr3_kistar_moveit] overriding "
                    + str(dropped)
                    + " -> "
                    + str(_OVERRIDES)
                    + " (this wrapper is MoveIt-only; launch the camera via "
                    "realsense_front.launch.py or use dual_fr3_kistar_all.launch.py)."
                )
            )
        )
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("franka_kistar_bringup"),
                        "launch",
                        "dual_fr3_kistar_planning_pc_v2.launch.py",
                    ]
                )
            ),
            launch_arguments=forwarded.items(),
        )
    )
    return actions


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_include_v2)])
