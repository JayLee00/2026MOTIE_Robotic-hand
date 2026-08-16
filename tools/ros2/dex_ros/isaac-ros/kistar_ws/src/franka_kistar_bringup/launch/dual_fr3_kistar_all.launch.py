# -----------------------------
# dual_fr3_kistar_all.launch.py
# All-in-one convenience entry point: MoveIt stack + front RealSense together.
#
# Composition (identical node set to the split workflow, one command):
#   1. dual_fr3_kistar_moveit.launch.py semantics — includes the v2 planning
#      launch with use_camera:=false + camera_view:=true (RViz uses the camera
#      rviz variant but the driver is NOT started inside the MoveIt include);
#   2. realsense_front.launch.py — the RealSense driver as its own include, so
#      it can be killed/restarted independently in the split workflow without
#      touching MoveIt.
#
# NOTE: running everything under one launch does not reduce total CPU load —
# the RealSense driver + pointcloud rendering cost the same. If RViz feels
# heavy, prefer the split workflow (launch MoveIt and camera in separate
# terminals) or drop streams: camera_depth:=false / pointcloud:=false.
#
# Camera stream selection accepts BOTH spellings (docs use camera_rgb/depth):
#   camera_rgb   / enable_color  -> RealSense color stream
#   camera_depth / enable_depth  -> RealSense depth stream
#   pointcloud                   -> depth pointcloud
# All other CLI args are forwarded verbatim to the v2 planning launch
# (joint_state_mode, robot_ip, use_rviz, front_camera_*, table_*, ...).
#
# Usage:
#   ros2 launch franka_kistar_bringup dual_fr3_kistar_all.launch.py \
#       joint_state_mode:=direct robot_ip:=192.168.0.100 use_rviz:=true \
#       camera_rgb:=true camera_depth:=true
# -----------------------------

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

# MoveIt include is display-only for the camera; the driver runs as the
# separate realsense_front include below.
_MOVEIT_OVERRIDES = {
    "use_camera": "false",
    "camera_view": "true",
}

# realsense_front.launch.py args, with the doc-facing camera_rgb/camera_depth
# aliases mapped onto the driver's enable_color/enable_depth.
_CAMERA_ARG_ALIASES = {
    "camera_rgb": "enable_color",
    "camera_depth": "enable_depth",
    "enable_color": "enable_color",
    "enable_depth": "enable_depth",
    "pointcloud": "pointcloud",
    "serial": "serial",
    "camera_namespace": "camera_namespace",
    "camera_name": "camera_name",
    "base_frame_id": "base_frame_id",
}


def _include_all(context, *args, **kwargs):
    cli_args = dict(context.launch_configurations)

    # Split camera-facing args out; everything (camera aliases included — they
    # are declared no-ops in v2 when use_camera:=false) is forwarded to v2.
    camera_args = {}
    for src, dst in _CAMERA_ARG_ALIASES.items():
        if src in cli_args:
            camera_args[dst] = cli_args[src]

    moveit_args = dict(cli_args)
    moveit_args.update(_MOVEIT_OVERRIDES)

    launch_dir = PathJoinSubstitution(
        [FindPackageShare("franka_kistar_bringup"), "launch"]
    )

    return [
        LogInfo(
            msg=(
                "[dual_fr3_kistar_all] MoveIt (camera_view rviz) + realsense_front "
                "camera_args=" + str(camera_args)
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [launch_dir, "dual_fr3_kistar_planning_pc_v2.launch.py"]
                )
            ),
            launch_arguments=moveit_args.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([launch_dir, "realsense_front.launch.py"])
            ),
            launch_arguments=camera_args.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_include_all)])
