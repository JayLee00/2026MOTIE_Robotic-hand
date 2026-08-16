# -----------------------------
# fr3_kistar_moveit_bringup.launch.py
# For Point2Point Motion Generator with Shared Memory Bridge!
# Cannot generate obstacle avoidance trajectory with this setup..
# Test real robot - direct 
# -----------------------------
    
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown
from launch.conditions import IfCondition
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

    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")

    bridge = LaunchConfiguration("bridge")  # isaac | real
    arm_side = LaunchConfiguration("arm_side")
    command_rate_hz = LaunchConfiguration("command_rate_hz")
    resample_dt = LaunchConfiguration("resample_dt")

    load_gripper = LaunchConfiguration("load_gripper")
    ee_id = LaunchConfiguration("ee_id")

    use_rviz = LaunchConfiguration("use_rviz")
    rviz_source = LaunchConfiguration("rviz_source")  # kistar | moveit
    rviz_config = LaunchConfiguration("rviz_config")

    # world/base/table TF args (from fr3_kistar.launch.py)
    world_frame = LaunchConfiguration("world_frame")

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

    camera_frame = LaunchConfiguration("camera_frame")

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
            " robot_ip:=",
            robot_ip,
            " use_fake_hardware:=",
            use_fake_hardware,
            " fake_sensor_commands:=",
            fake_sensor_commands,
            # MoveIt+execution을 위한 ros2_control interface 생성(URDF내)
            " ros2_control:=true",
            # 필요하면 추가 인자도 여기에 계속 붙이면 됨 (arm_prefix 등)
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_config, value_type=str)
    }

    # -----------------------------
    # Semantic (SRDF)
    # -----------------------------
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

    # -----------------------------
    # Planning configs
    # -----------------------------
    kinematics_yaml = load_yaml("franka_kistar_moveit_config", "config/kinematics.yaml")
    joint_limits_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/joint_limits.yaml"
    )
    robot_description_planning = {"robot_description_planning": joint_limits_yaml}

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

    totg_params = {
        "time_optimal_trajectory_generation.resample_dt": ParameterValue(
            resample_dt, value_type=float
        ),
        "time_optimal_trajectory_generation.path_tolerance": 0.1,
        "time_optimal_trajectory_generation.min_angle_change": 0.001,
    }

    moveit_simple_controllers_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/fr3_controllers.yaml"
    )
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_simple_controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 3.0,
        "trajectory_execution.allowed_goal_duration_margin": 2.0,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # -----------------------------
    # Tables (URDF/xacro) - from fr3_kistar.launch.py
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
    # Static TFs (global /tf) - from fr3_kistar.launch.py
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

    world_to_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_camera_tf",
        arguments=[
            "--x",
            table_x,
            "--y",
            table_y,
            "--z",
            "0.9",
            "--roll",
            "0.",
            "--pitch",
            "0.",
            "--yaw",
            "0.",
            "--frame-id",
            world_frame,
            "--child-frame-id",
            camera_frame,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    # AprilTag marker TFs (ttable -> marker_i)
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

    ttable_to_marker0_tf = marker_tf(
        "ttable_to_marker0_tf", "0.175", "0.325", "0.", ttable_frame, "marker_0_frame"
    )
    ttable_to_marker1_tf = marker_tf(
        "ttable_to_marker1_tf", "0.175", "-0.325", "0.", ttable_frame, "marker_1_frame"
    )
    ttable_to_marker2_tf = marker_tf(
        "ttable_to_marker2_tf", "-0.175", "0.325", "0.", ttable_frame, "marker_2_frame"
    )
    ttable_to_marker3_tf = marker_tf(
        "ttable_to_marker3_tf", "-0.175", "-0.325", "0.", ttable_frame, "marker_3_frame"
    )

    # -----------------------------
    # robot_state_publisher (robot + tables) (global /tf)
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
            robot_description_planning,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            totg_params,
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
            robot_description_planning,
            ompl_planning_pipeline_config,
            kinematics_yaml,
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
        condition=IfCondition(use_rviz),
    )

    # -----------------------------
    # Bridges (isaac / real)
    # -----------------------------
    isaac_bridge_node = Node(
        package="franka_kistar_isaac_moveit",
        executable="isaac_moveit_bridge",
        namespace=namespace,
        name="fr3_arm_controller",
        output="screen",
        condition=IfCondition(PythonExpression(["'", bridge, "' == 'isaac'"])),
    )

    real_bridge_node = Node(
        package="franka_kistar_isaac_moveit",
        executable="real_moveit_bridge",
        namespace=namespace,
        name="fr3_arm_controller",
        output="screen",
        parameters=[
            {
                "arm_side": arm_side,
                "command_rate_hz": command_rate_hz,
                "publish_dummy_hand_joints": True,  # 너 코드 유지
            }
        ],
        condition=IfCondition(PythonExpression(["'", bridge, "' == 'real'"])),
    )

    scene_boxes_node = Node(
        package="franka_kistar_bringup",
        executable="planning_scene_static_boxes.py",  # setup.py entry point 이름
        output="screen",
        parameters=[
            {
                "world_frame": LaunchConfiguration("world_frame"),
                "table_frame": LaunchConfiguration("table_frame"),
                "ttable_frame": LaunchConfiguration("ttable_frame"),
                "table_size": LaunchConfiguration("table_size"),
                "ttable_size": LaunchConfiguration("ttable_size"),
            }
        ],
    )

    # (중요) TF/MoveGroup 뜨는 시간 주려고 2초 지연 추천
    scene_boxes_delayed = TimerAction(period=3.0, actions=[scene_boxes_node])

    # -----------------------------
    # Gripper (optional)
    # -----------------------------
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

    # -----------------------------
    # Launch args
    # -----------------------------
    launch_args = [
        DeclareLaunchArgument(
            "namespace", default_value="", description="Namespace for the robot."
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "robot_ip", default_value="172.16.0.3", description="Robot IP"
        ),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
        DeclareLaunchArgument(
            "bridge", default_value="real", description="isaac or real"
        ),
        DeclareLaunchArgument(
            "arm_side", default_value="right", description="left or right"
        ),
        DeclareLaunchArgument("command_rate_hz", default_value="100.0"),
        DeclareLaunchArgument("resample_dt", default_value="0.01"),
        DeclareLaunchArgument("load_gripper", default_value="false"),
        DeclareLaunchArgument("ee_id", default_value="franka_hand"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_source", default_value="kistar", description="kistar or moveit"
        ),
        DeclareLaunchArgument("rviz_config", default_value="fr3_kistar.rviz"),
        # frames
        DeclareLaunchArgument("world_frame", default_value="world"),
        # world -> base (same defaults as fr3_kistar.launch.py)
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
        DeclareLaunchArgument("camera_frame", default_value="camera_link"),
    ]

    return LaunchDescription(
        launch_args
        + [
            # TFs
            world_to_base_tf,
            base_to_fr3_tf,
            world_to_table_tf,
            world_to_ttable_tf,
            world_to_camera_tf,
            ttable_to_marker0_tf,
            ttable_to_marker1_tf,
            ttable_to_marker2_tf,
            ttable_to_marker3_tf,
            # RSP
            table_rsp,
            ttable_rsp,
            robot_rsp,
            # MoveIt
            run_move_group_node,
            scene_boxes_delayed,
            # RViz
            rviz_node,
            # Gripper + Bridges
            real_bridge_node,
        ]
    )
