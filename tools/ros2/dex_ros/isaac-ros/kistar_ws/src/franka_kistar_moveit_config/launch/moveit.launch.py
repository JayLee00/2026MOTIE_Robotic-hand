import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
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

import yaml


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    with open(absolute_file_path, "r") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    robot_ip_parameter_name = "robot_ip"
    use_fake_hardware_parameter_name = "use_fake_hardware"
    fake_sensor_commands_parameter_name = "fake_sensor_commands"
    namespace_parameter_name = "namespace"
    load_gripper_parameter_name = "load_gripper"
    ee_id_parameter_name = "ee_id"
    bridge_parameter_name = "bridge"
    arm_side_parameter_name = "arm_side"
    command_rate_hz_parameter_name = "command_rate_hz"
    resample_dt_parameter_name = "resample_dt"
    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    namespace = LaunchConfiguration(namespace_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    ee_id = LaunchConfiguration(ee_id_parameter_name)
    bridge = LaunchConfiguration(bridge_parameter_name)
    arm_side = LaunchConfiguration(arm_side_parameter_name)
    command_rate_hz = LaunchConfiguration(command_rate_hz_parameter_name)

    resample_dt = LaunchConfiguration(resample_dt_parameter_name)

    # Command-line arguments
    db_arg = DeclareLaunchArgument(
        "db", default_value="False", description="Database flag"
    )

    # planning_context
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
            " ros2_control:=true",
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_config, value_type=str)
    }

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

    kinematics_yaml = load_yaml("franka_kistar_moveit_config", "config/kinematics.yaml")
    joint_limits_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/joint_limits.yaml"
    )
    robot_description_planning = {"robot_description_planning": joint_limits_yaml}
    # Planning Functionality
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
    # Trajectory Execution Functionality
    moveit_simple_controllers_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/fr3_controllers.yaml"
    )
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_simple_controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager"
        "/MoveItSimpleControllerManager",
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

    # Start the actual move_group node/action server
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
            # joint_limits_yaml,
            ompl_planning_pipeline_config,
            totg_params,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
        ],
    )

    # RViz
    rviz_base = os.path.join(
        get_package_share_directory("franka_kistar_moveit_config"), "rviz"
    )
    rviz_full_config = os.path.join(rviz_base, "moveit.rviz")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_full_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_planning,
            ompl_planning_pipeline_config,
            kinematics_yaml,
            # joint_limits_yaml,
        ],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=namespace,
        output="both",
        parameters=[robot_description],
    )

    isaac_bridge_node = Node(
        package="franka_kistar_isaac_moveit",
        executable="isaac_moveit_bridge",
        namespace=namespace,
        name="fr3_arm_controller",  # 중요: 노드 이름을 fr3_arm_controller로 해서
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
                "publish_dummy_hand_joints": True,  # TODO
            }
        ],
        condition=IfCondition(PythonExpression(["'", bridge, "' == 'real'"])),
    )

    robot_arg = DeclareLaunchArgument(
        robot_ip_parameter_name, description="Hostname or IP address of the robot."
    )

    namespace_arg = DeclareLaunchArgument(
        namespace_parameter_name,
        default_value="",
        description="Namespace for the robot.",
    )

    bridge_arg = DeclareLaunchArgument(
        bridge_parameter_name,
        default_value="isaac",
        description="Bridge type (isaac or real).",
    )

    arm_side_arg = DeclareLaunchArgument(
        arm_side_parameter_name,
        default_value="right",
        description="Arm side for real bridge (left or right).",
    )

    command_rate_hz_arg = DeclareLaunchArgument(
        command_rate_hz_parameter_name,
        default_value="100.0",
        description="Command streaming rate for real bridge (Hz).",
    )

    resample_dt_arg = DeclareLaunchArgument(
        resample_dt_parameter_name,
        default_value="0.01",
        description="TOTG resample dt (sec). Smaller -> more traj points (e.g., 0.01 ~ 100Hz).",
    )

    load_gripper_arg = DeclareLaunchArgument(
        load_gripper_parameter_name,
        default_value="true",
        description="Whether to load the gripper or not (true or false)",
    )

    ee_id_arg = DeclareLaunchArgument(
        ee_id_parameter_name,
        default_value="franka_hand",
        description="The end-effector id to use. Available options: none, franka_hand, cobot_pump",
    )

    use_fake_hardware_arg = DeclareLaunchArgument(
        use_fake_hardware_parameter_name,
        default_value="false",
        description="Use fake hardware",
    )

    fake_sensor_commands_arg = DeclareLaunchArgument(
        fake_sensor_commands_parameter_name,
        default_value="false",
        description="Fake sensor commands. Only valid when '{}' is true".format(
            use_fake_hardware_parameter_name
        ),
    )

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
            use_fake_hardware_parameter_name: use_fake_hardware,
            "namespace": namespace,
        }.items(),
        condition=IfCondition(load_gripper),
    )

    return LaunchDescription(
        [
            robot_arg,
            namespace_arg,
            bridge_arg,
            arm_side_arg,
            command_rate_hz_arg,
            resample_dt_arg,
            load_gripper_arg,
            ee_id_arg,
            use_fake_hardware_arg,
            fake_sensor_commands_arg,
            db_arg,
            rviz_node,
            robot_state_publisher,
            run_move_group_node,
            gripper_launch_file,
            isaac_bridge_node,
        ]
    )
