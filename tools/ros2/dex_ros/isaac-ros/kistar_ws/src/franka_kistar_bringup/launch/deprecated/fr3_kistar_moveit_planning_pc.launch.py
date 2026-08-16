# -----------------------------
# fr3_kistar_moveit_planning_pc.launch.py
# PC1 (Planning Computer) - MoveIt Planning + Trajectory Publishing
#
# fr3_kistar_moveit_real.launch.py와 동일한 설정 (RViz, collision, planning scene 등)
# 하지만 execution은 /trajectory_commands topic으로 publish
# Test real robot - ROS2 - other PC
#
# Usage:
# ros2 launch franka_kistar_bringup fr3_kistar_moveit_planning_pc.launch.py
# -----------------------------

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    Shutdown,
)
from launch.conditions import UnlessCondition, IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import IfElseSubstitution, TextSubstitution
from launch.actions import TimerAction


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # -----------------------------
    # LaunchConfigurations
    # -----------------------------
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")

    use_rviz = LaunchConfiguration("use_rviz")
    use_fake_joint_states = LaunchConfiguration("use_fake_joint_states")
    rviz_source = LaunchConfiguration("rviz_source")  # kistar | moveit
    rviz_config = LaunchConfiguration("rviz_config")

    # world/base/table TF args
    world_frame = LaunchConfiguration("world_frame")
    robot_ip = LaunchConfiguration("robot_ip")

    robot_base_x = LaunchConfiguration("robot_base_x")
    robot_base_y = LaunchConfiguration("robot_base_y")
    robot_base_z = LaunchConfiguration("robot_base_z")
    robot_base_roll = LaunchConfiguration("robot_base_roll")
    robot_base_pitch = LaunchConfiguration("robot_base_pitch")
    robot_base_yaw = LaunchConfiguration("robot_base_yaw")

    table_frame = LaunchConfiguration("table_frame")
    table_x = LaunchConfiguration("table_x")
    table_y = LaunchConfiguration("table_y")
    table_z = LaunchConfiguration("table_z")
    table_roll = LaunchConfiguration("table_roll")
    table_pitch = LaunchConfiguration("table_pitch")
    table_yaw = LaunchConfiguration("table_yaw")

    ttable_frame = LaunchConfiguration("ttable_frame")
    ttable_x = LaunchConfiguration("ttable_x")
    ttable_y = LaunchConfiguration("ttable_y")
    ttable_z = LaunchConfiguration("ttable_z")

    front_camera_link = LaunchConfiguration("front_camera_link")
    side_camera_link = LaunchConfiguration("side_camera_link")
    profile_frame = LaunchConfiguration("profile_frame")

    # -----------------------------
    # Robot Description (URDF/xacro)
    # -----------------------------
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
            " robot_ip:=", robot_ip,  # Planning only, no real robot
            " use_fake_hardware:=true",
            " fake_sensor_commands:=true",
            " ros2_control:=false",  # No controller on PC1
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_config, value_type=str)
    }

    # -----------------------------
    # Semantic (SRDF) - Use custom fr3_kistar.srdf
    # -----------------------------
    franka_semantic_file = os.path.join(
        get_package_share_directory("franka_kistar_moveit_config"),
        "config",
        "fr3_kistar.srdf",
    )

    with open(franka_semantic_file, "r") as f:
        robot_description_semantic_content = f.read()

    robot_description_semantic = {
        "robot_description_semantic": robot_description_semantic_content
    }

    # -----------------------------
    # Planning configs
    # -----------------------------
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
        "publish_robot_description_semantic": True,
        # Real-time joint state tracking parameters
        "publish_robot_state": True,
        "publish_robot_state_frequency": 50.0,  # 50 Hz - real-time tracking
        "monitor_dynamics": False,  # Don't monitor velocity/acceleration (reduces latency)
    }

    # -----------------------------
    # Tables (URDF/xacro)
    # -----------------------------
    table_xacro_path = PathJoinSubstitution(
        [
            FindPackageShare("franka_kistar_bringup"),
            "urdf",
            "table.urdf.xacro",
        ]
    )

    table_xacro_cmd = Command([FindExecutable(name="xacro"), " ", table_xacro_path])
    table_description = {
        "robot_description": ParameterValue(table_xacro_cmd, value_type=str)
    }

    ttable_xacro_path = PathJoinSubstitution(
        [
            FindPackageShare("franka_kistar_bringup"),
            "urdf",
            "ttable.urdf.xacro",
        ]
    )
    ttable_xacro_cmd = Command([FindExecutable(name="xacro"), " ", ttable_xacro_path])
    ttable_description = {
        "robot_description": ParameterValue(ttable_xacro_cmd, value_type=str)
    }

    # -----------------------------
    # Static TFs (global /tf)
    # -----------------------------
    world_to_base_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_tf",
        arguments=[
            "--x",
            robot_base_x,
            "--y",
            robot_base_y,
            "--z",
            robot_base_z,
            "--roll",
            robot_base_roll,
            "--pitch",
            robot_base_pitch,
            "--yaw",
            robot_base_yaw,
            "--frame-id",
            world_frame,
            "--child-frame-id",
            "base",
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    base_to_fr3_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_fr3_link0_tf",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "base",
            "--child-frame-id",
            "fr3_link0",
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    world_to_table_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_table_tf",
        arguments=[
            "--x",
            table_x,
            "--y",
            table_y,
            "--z",
            table_z,
            "--roll",
            table_roll,
            "--pitch",
            table_pitch,
            "--yaw",
            table_yaw,
            "--frame-id",
            world_frame,
            "--child-frame-id",
            table_frame,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    world_to_ttable_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_ttable_tf",
        arguments=[
            "--x",
            ttable_x,
            "--y",
            ttable_y,
            "--z",
            ttable_z,
            "--roll",
            table_roll,
            "--pitch",
            table_pitch,
            "--yaw",
            table_yaw,
            "--frame-id",
            world_frame,
            "--child-frame-id",
            ttable_frame,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    profile_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="profile_tf",
        arguments=[
            "--x",
            "0.025",
            "--y",
            "0.032",
            "--z",
            "0.0",
            "--roll",
            "0.",
            "--pitch",
            "0.",
            "--yaw",
            "0.",
            "--frame-id",
            world_frame,
            "--child-frame-id",
            profile_frame,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    # Camera TF
    front_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="front_camera_tf",
        arguments=[
            "--x",
            "0.12",
            "--y",
            "0.02",
            "--z",
            "0.88",
            "--roll",
            "0.",
            "--pitch",
            "0.8727",
            "--yaw",
            "0.",
            "--frame-id",
            world_frame,
            "--child-frame-id",
            front_camera_link,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    side_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="side_camera_tf",
        arguments=[
            "--x",
            "0.55",
            "--y",
            "0.82",
            "--z",
            "0.6",
            "--roll",
            "0.",
            "--pitch",
            "0.4363",
            "--yaw",
            "-1.5708",
            "--frame-id",
            world_frame,
            "--child-frame-id",
            side_camera_link,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    # Marker TFs
    def marker_tf(name, x, y, z, parent, child):
        return Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=name,
            arguments=[
                "--x",
                x,
                "--y",
                y,
                "--z",
                z,
                "--roll",
                "0.",
                "--pitch",
                "0.",
                "--yaw",
                "0.",
                "--frame-id",
                parent,
                "--child-frame-id",
                child,
            ],
            remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
            output="screen",
        )

    sw_wall_tf = marker_tf(
        "sw_wall_tf", "-0.5", "0.", "0.", table_frame, "sw_wall_frame"
    )

    # -----------------------------
    # robot_state_publisher (robot + tables)
    # -----------------------------
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
    # PC2가 연결된 경우 use_fake_joint_states:=false로 비활성화
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        namespace=namespace,
        parameters=[
            robot_description,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(use_fake_joint_states),
    )

    table_rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="table_state_publisher",
        namespace="table",
        parameters=[
            table_description,
            {"publish_robot_description": True},
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("tf", "/tf"),
            ("tf_static", "/tf_static"),
            ("robot_description", "/table_description"),
        ],
        output="screen",
    )

    ttable_rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ttable_state_publisher",
        namespace="ttable",
        parameters=[
            ttable_description,
            {"publish_robot_description": True},
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("tf", "/tf"),
            ("tf_static", "/tf_static"),
            ("robot_description", "/ttable_description"),
        ],
        output="screen",
    )

    # -----------------------------
    # MoveIt move_group
    # -----------------------------
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

    # -----------------------------
    # RViz config selection
    # -----------------------------
    kistar_rviz_path = PathJoinSubstitution(
        [
            FindPackageShare("franka_kistar_bringup"),
            "rviz",
            rviz_config,
        ]
    )

    moveit_rviz_path = os.path.join(
        get_package_share_directory("franka_kistar_moveit_config"),
        "rviz",
        "moveit.rviz",
    )

    rviz_cfg = IfElseSubstitution(
        PythonExpression(["'", rviz_source, "' == 'kistar'"]),
        kistar_rviz_path,
        TextSubstitution(text=moveit_rviz_path),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        namespace=namespace,
        arguments=["-d", rviz_cfg],
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

    # -----------------------------
    # Planning Scene (collision objects)
    # -----------------------------
    scene_boxes_node = Node(
        package="franka_kistar_bringup",
        executable="planning_scene_static_boxes.py",
        output="screen",
        parameters=[
            {
                "world_frame": LaunchConfiguration("world_frame"),
                "table_frame": LaunchConfiguration("table_frame"),
                "ttable_frame": LaunchConfiguration("ttable_frame"),
                "profile_frame": LaunchConfiguration("profile_frame"),
                "table_size": LaunchConfiguration("table_size"),
                "ttable_size": LaunchConfiguration("ttable_size"),
            }
        ],
    )

    scene_boxes_delayed = TimerAction(period=3.0, actions=[scene_boxes_node])

    # -----------------------------
    # Trajectory Forwarder (핵심: Action → Topic)
    # -----------------------------
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

    # -----------------------------
    # Launch args
    # -----------------------------
    launch_args = [
        DeclareLaunchArgument(
            "namespace", default_value="", description="Namespace for the robot."
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "use_fake_joint_states",
            default_value="false",
            description="Use fake joint_state_publisher (false when PC2 is connected)"
        ),
        DeclareLaunchArgument(
            "rviz_source", default_value="kistar", description="kistar or moveit"
        ),
        DeclareLaunchArgument("rviz_config", default_value="fr3_kistar.rviz"),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="172.16.0.1",
            description="Robot IP address (planning only, no actual connection)"
        ),
        # frames
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("robot_base_x", default_value="0.066"),
        DeclareLaunchArgument("robot_base_y", default_value="-0.122"),
        DeclareLaunchArgument("robot_base_z", default_value="0.099"),
        DeclareLaunchArgument("robot_base_roll", default_value="0.785"),
        DeclareLaunchArgument("robot_base_pitch", default_value="0.0"),
        DeclareLaunchArgument("robot_base_yaw", default_value="0.0"),
        # table pose
        DeclareLaunchArgument("table_frame", default_value="table_link"),
        DeclareLaunchArgument("table_x", default_value="0.0"),
        DeclareLaunchArgument("table_y", default_value="0.032"),
        DeclareLaunchArgument("table_z", default_value="0.0"),
        DeclareLaunchArgument("ttable_frame", default_value="ttable_link"),
        DeclareLaunchArgument("ttable_x", default_value="0.6"),
        DeclareLaunchArgument("ttable_y", default_value="0.0"),
        DeclareLaunchArgument("ttable_z", default_value="0.205"),
        DeclareLaunchArgument("table_roll", default_value="0.0"),
        DeclareLaunchArgument("table_pitch", default_value="0.0"),
        DeclareLaunchArgument("table_yaw", default_value="0.0"),
        DeclareLaunchArgument("table_size", default_value="[1.2, 1.8, 0.05]"),
        DeclareLaunchArgument("ttable_size", default_value="[0.5, 0.8, 0.03]"),
        DeclareLaunchArgument("front_camera_link", default_value="front_camera_link"),
        DeclareLaunchArgument("side_camera_link", default_value="side_camera_link"),
        DeclareLaunchArgument("profile_frame", default_value="profile_frame"),
    ]

    return LaunchDescription(
        launch_args
        + [
            # TFs
            world_to_base_tf,
            base_to_fr3_tf,
            world_to_table_tf,
            world_to_ttable_tf,
            profile_tf,
            front_camera_tf,
            side_camera_tf,
            sw_wall_tf,
            scene_boxes_delayed,
            # RSP
            table_rsp,
            ttable_rsp,
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
