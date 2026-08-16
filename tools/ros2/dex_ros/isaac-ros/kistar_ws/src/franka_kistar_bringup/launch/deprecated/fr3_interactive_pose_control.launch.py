"""
Interactive Pose Control Launch File

PC1 (Planning Computer)에서 실행:
- MoveIt planning
- pose_commander (CUI pose input + user confirmation)
- trajectory_forwarder (trajectory → /trajectory_commands topic)
- RViz (조건부 - gui:=true일 때)

Usage:
  ros2 launch franka_kistar_bringup fr3_interactive_pose_control.launch.py
  ros2 launch franka_kistar_bringup fr3_interactive_pose_control.launch.py gui:=true
  ros2 launch franka_kistar_bringup fr3_interactive_pose_control.launch.py gui:=false

Author: Chanyoung Ahn
Date: 2025
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    gui = LaunchConfiguration("gui")
    planning_time = LaunchConfiguration("planning_time")
    end_effector_link = LaunchConfiguration("end_effector_link")

    # Include fr3_kistar_moveit_planning_pc.launch.py
    # (이미 MoveIt + trajectory_forwarder + RViz 포함)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("franka_kistar_bringup"),
                "launch",
                "fr3_kistar_moveit_planning_pc.launch.py"
            ])
        ]),
        launch_arguments={
            "use_rviz": gui,  # gui 인자에 따라 RViz on/off
        }.items()
    )

    # Pose Commander Node
    pose_commander = Node(
        package="franka_kistar_bringup",
        executable="pose_commander.py",
        name="pose_commander",
        output="screen",
        emulate_tty=True,  # Enable terminal input for pose entry
        parameters=[
            {
                "gui": gui,
                "planning_group": "fr3_arm",
                "end_effector_link": end_effector_link,
                "planning_time": planning_time,
                "reference_frame": "world",
            }
        ],
    )

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Enable RViz GUI (true/false)"
        ),
        DeclareLaunchArgument(
            "planning_time",
            default_value="5.0",
            description="MoveIt planning timeout (seconds)"
        ),
        DeclareLaunchArgument(
            "end_effector_link",
            default_value="fr3_link8",
            description="End-effector link name"
        ),

        # Nodes
        moveit_launch,
        pose_commander,
    ])
