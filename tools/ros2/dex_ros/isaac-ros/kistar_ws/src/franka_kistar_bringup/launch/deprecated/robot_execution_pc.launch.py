"""
Robot Execution PC Launch File (PC2 - 192.168.1.250)

이 launch 파일은 Robot Computer에서 실행됩니다.
- TrajectorySubscriber (Topic → Action)
- JointTrajectoryController (ros2_control)
- Hardware Interface (libfranka)

PC1 (192.168.1.38)에서 전송한 trajectory를 수신하여 실제 로봇을 제어합니다.

Usage (Real Robot):
  ros2 launch franka_kistar_bringup robot_execution_pc.launch.py robot_ip:=172.16.0.1

Usage (Fake Hardware for testing):
  ros2 launch franka_kistar_bringup robot_execution_pc.launch.py use_fake_hardware:=true

Author: Chanyoung Ahn
Date: 2026
"""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import UnlessCondition, IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # LaunchConfigurations
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")

    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")

    load_gripper = LaunchConfiguration("load_gripper")

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
            " robot_ip:=",
            robot_ip,
            " use_fake_hardware:=",
            use_fake_hardware,
            " fake_sensor_commands:=",
            fake_sensor_commands,
            " ros2_control:=true",  # Controller enabled on PC2
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_config, value_type=str)
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

    # Robot State Publisher
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

    # ROS2 Control Node (Controller Manager)
    ros2_controllers_path = os.path.join(
        get_package_share_directory("franka_kistar_moveit_config"),
        "config",
        "fr3_ros_controllers.yaml",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=namespace,
        parameters=[robot_description, ros2_controllers_path],
        remappings=[("joint_states", "franka/joint_states")],
        output="screen",
    )

    # Load Controllers
    load_controllers = []
    for controller in ["fr3_arm_controller", "joint_state_broadcaster"]:
        load_controllers.append(
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "controller_manager",
                    "spawner",
                    controller,
                    "--controller-manager-timeout",
                    "60",
                    "--controller-manager",
                    PathJoinSubstitution([namespace, "controller_manager"]),
                ],
                output="screen",
            )
        )

    # Joint State Publisher (merge joint states)
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        namespace=namespace,
        parameters=[
            {
                "source_list": ["franka/joint_states"],
                "rate": 30,
            }
        ],
    )

    # Franka Robot State Broadcaster (only for real robot)
    franka_robot_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=["franka_robot_state_broadcaster"],
        output="screen",
        condition=UnlessCondition(use_fake_hardware),
    )

    # Trajectory Subscriber (핵심: Topic → Action → Controller)
    trajectory_subscriber = Node(
        package="franka_kistar_bringup",
        executable="trajectory_subscriber.py",
        name="trajectory_subscriber",
        namespace=namespace,
        output="screen",
        parameters=[
            {
                "topic_name": "/trajectory_commands",
                "action_name": "/fr3_arm_controller/follow_joint_trajectory",
                "timeout_sec": 60.0,
            }
        ],
    )

    # Gripper (optional)
    gripper_launch_file = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [FindPackageShare("franka_gripper"), "launch", "gripper.launch.py"]
                )
            ]
        ),
        launch_arguments={
            "robot_ip": robot_ip,
            "use_fake_hardware": use_fake_hardware,
            "namespace": namespace,
        }.items(),
        condition=IfCondition(load_gripper),
    )

    # Launch args
    launch_args = [
        DeclareLaunchArgument(
            "namespace", default_value="", description="Namespace for the robot."
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "robot_ip", default_value="172.16.0.1", description="Robot IP"
        ),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
        DeclareLaunchArgument("load_gripper", default_value="false"),
        # Frames
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
            # Robot State
            robot_rsp,
            # ROS2 Control
            ros2_control_node,
            joint_state_publisher,
            franka_robot_state_broadcaster,
            # Trajectory Subscriber (핵심 노드)
            trajectory_subscriber,
            # Gripper
            gripper_launch_file,
        ]
        + load_controllers
    )
