import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # 1) ros2_control 하드웨어 타입 (mock vs isaac)
    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",  # <- IsaacSim이랑 붙일 때는 isaac
        description="ROS2 control hardware interface type [mock_components, isaac]",
    )

    # 2) use_sim_time
    use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock if true",
    )

    # 3) MoveIt config: panda → fr3_kistar
    moveit_config = (
        MoveItConfigsBuilder(
            "fr3_kistar",  # 로봇 이름
            package_name="franka_kistar_moveit_config",  # 네 MoveIt 설정 패키지 이름
        )
        .robot_description(
            file_path="config/fr3_kistar_isaac.urdf.xacro",  # <- 이 xacro를 네가 만들어야 함
            mappings={
                # URDF 안에서 ros2_control hardware를 바꾸고 싶다면 이런 식으로 전달
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        .robot_description_semantic(file_path="config/fr3_kistar.srdf")
        .trajectory_execution(file_path="config/fr3_kistar_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # 4) move_group node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # 5) RViz
    rviz_config_file = os.path.join(
        get_package_share_directory("franka_kistar_moveit_config"),
        "rviz",
        "fr3_kistar_moveit.rviz",  # <- panda용 말고, 네가 새로 만든 rviz 파일 이름
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    # 6) static TF: world → base (또는 fr3_link0)
    world2robot_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher_world_to_robot",
        output="log",
        arguments=[
            "0.0",
            "0.0",
            "0.0",  # x y z
            "0.0",
            "0.0",
            "0.0",  # r p y
            "world",
            "base",  # parent, child (네 URDF 기준으로 맞춰)
        ],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # 7) robot_state_publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    # 8) ros2_control: panda → fr3_kistar
    ros2_controllers_path = os.path.join(
        get_package_share_directory("franka_kistar_moveit_config"),
        "config",
        "fr3_kistar_isaac_ros2_controllers.yaml",  # <- 네가 새로 만들 파일
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            ros2_controllers_path,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    # 9) 컨트롤러 스포너들: 이름을 controllers.yaml이랑 맞춰야 함
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    fr3_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["fr3_arm_controller", "-c", "/controller_manager"],
    )

    # 손 컨트롤러는 나중에 붙일 거면 이렇게:
    # kistar_hand_controller_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["kistar_hand_controller", "-c", "/controller_manager"],
    # )

    return LaunchDescription(
        [
            ros2_control_hardware_type,
            use_sim_time,
            rviz_node,
            world2robot_tf_node,
            robot_state_publisher,
            move_group_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            fr3_arm_controller_spawner,
            # kistar_hand_controller_spawner,
        ]
    )
