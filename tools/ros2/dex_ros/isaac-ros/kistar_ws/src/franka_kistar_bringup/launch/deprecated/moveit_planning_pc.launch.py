"""
MoveIt Planning PC Launch File (PC1 - 192.168.1.38)

이 launch 파일은 Planning Computer에서 실행됩니다.
- MoveIt move_group (planning만)
- RViz (visualization)
- TrajectoryForwarder (Action → Topic)

생성된 trajectory를 /trajectory_commands topic으로 publish하여
Robot PC (192.168.1.250)로 전송합니다.

Usage:
  ros2 launch franka_kistar_bringup moveit_planning_pc.launch.py

Author: Chanyoung Ahn
Date: 2025
"""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # LaunchConfigurations
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")

    # Frames
    world_frame = LaunchConfiguration("world_frame")
    robot_base_x = LaunchConfiguration("robot_base_x")
    robot_base_y = LaunchConfiguration("robot_base_y")
    robot_base_z = LaunchConfiguration("robot_base_z")
    robot_base_roll = LaunchConfiguration("robot_base_roll")
    robot_base_pitch = LaunchConfiguration("robot_base_pitch")
    robot_base_yaw = LaunchConfiguration("robot_base_yaw")

    # Robot Description (URDF/xacro)
    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_kistar_description"),
        "urdf",
        "fr3_kistar.urdf.xacro",
    )

    robot_description_config = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            franka_xacro_file,
            " robot_ip:=dummy",  # Planning only, no real robot
            " use_fake_hardware:=true",
            " fake_sensor_commands:=false",
            " ros2_control:=false",  # No controller on PC1
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_config, value_type=str)
    }

    # Semantic (SRDF)
    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "fr3",
        "fr3.srdf.xacro",
    )

    robot_description_semantic_config = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            franka_semantic_xacro_file,
            " hand:=false",
            " ee_id:=none",
        ]
    )

    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            robot_description_semantic_config, value_type=str
        )
    }

    # Planning configs
    kinematics_yaml = load_yaml("franka_kistar_moveit_config", "config/kinematics.yaml")

    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": "default_planner_request_adapters/AddTimeOptimalParameterization "
            "default_planner_request_adapters/ResolveConstraintFrames "
            "default_planner_request_adapters/FixWorkspaceBounds "
            "default_planner_request_adapters/FixStartStateBounds "
            "default_planner_request_adapters/FixStartStateCollision "
            "default_planner_request_adapters/FixStartStatePathConstraints",
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl_planning_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/ompl_planning.yaml"
    )
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

    # MoveIt Controller (TrajectoryForwarder가 이 action을 받음)
    moveit_simple_controllers_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/fr3_controllers.yaml"
    )
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_simple_controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # Static TFs
    world_to_base_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_tf",
        arguments=[
            "--x", robot_base_x,
            "--y", robot_base_y,
            "--z", robot_base_z,
            "--roll", robot_base_roll,
            "--pitch", robot_base_pitch,
            "--yaw", robot_base_yaw,
            "--frame-id", world_frame,
            "--child-frame-id", "base",
        ],
        output="screen",
    )

    base_to_fr3_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_fr3_link0_tf",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "base",
            "--child-frame-id", "fr3_link0",
        ],
        output="screen",
    )

    # Robot State Publisher (Planning only - fake joint states)
    robot_rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=namespace,
        parameters=[
            robot_description,
            {"publish_robot_description": True},
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("tf", "/tf"),
            ("tf_static", "/tf_static"),
            ("robot_description", "/robot_description"),
        ],
        output="screen",
    )

    # Joint State Publisher (Planning only - fake states)
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        namespace=namespace,
        parameters=[
            robot_description,
            {"use_sim_time": use_sim_time},
        ],
    )

    # MoveIt move_group
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
        ],
    )

    # RViz
    moveit_rviz_path = os.path.join(
        get_package_share_directory("franka_kistar_moveit_config"),
        "rviz",
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        namespace=namespace,
        arguments=["-d", moveit_rviz_path],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            kinematics_yaml,
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
        condition=IfCondition(use_rviz),
    )

    # Trajectory Forwarder (핵심: Action → Topic)
    trajectory_forwarder = Node(
        package="franka_kistar_bringup",
        executable="trajectory_forwarder.py",
        name="trajectory_forwarder",
        namespace=namespace,
        output="screen",
        parameters=[
            {
                "action_name": "/fr3_arm_controller/follow_joint_trajectory",
                "topic_name": "/trajectory_commands",
            }
        ],
    )

    # Launch args
    launch_args = [
        DeclareLaunchArgument(
            "namespace", default_value="", description="Namespace for the robot."
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("robot_base_x", default_value="0.066"),
        DeclareLaunchArgument("robot_base_y", default_value="-0.122"),
        DeclareLaunchArgument("robot_base_z", default_value="0.099"),
        DeclareLaunchArgument("robot_base_roll", default_value="0.785"),
        DeclareLaunchArgument("robot_base_pitch", default_value="0.0"),
        DeclareLaunchArgument("robot_base_yaw", default_value="0.0"),
    ]

    return LaunchDescription(
        launch_args
        + [
            # TFs
            world_to_base_tf,
            base_to_fr3_tf,
            # Publishers
            robot_rsp,
            joint_state_publisher,
            # MoveIt
            run_move_group_node,
            # Trajectory Forwarder (핵심 노드)
            trajectory_forwarder,
            # RViz
            rviz_node,
        ]
    )
