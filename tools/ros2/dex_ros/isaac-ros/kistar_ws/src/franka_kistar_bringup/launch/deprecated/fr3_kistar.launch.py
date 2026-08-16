import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, FindExecutable


def generate_robot_nodes(context):
    namespace = LaunchConfiguration('namespace').perform(context)

    # -----------------------------
    # Robot xacro -> robot_description
    # -----------------------------
    robot_urdf_path = os.path.join(
        get_package_share_directory("franka_kistar_description"),
        "urdf",
        "fr3_kistar.urdf.xacro",
    )

    robot_xacro_cmd = Command([
        FindExecutable(name='xacro'), ' ',
        robot_urdf_path,
        ' ros2_control:=', LaunchConfiguration('enable_ros2_control'),
        ' arm_prefix:=', LaunchConfiguration('arm_prefix'),
        ' robot_ip:=', LaunchConfiguration('robot_ip'),
        ' use_fake_hardware:=', LaunchConfiguration('use_fake_hardware'),
        ' fake_sensor_commands:=', LaunchConfiguration('fake_sensor_commands'),
        ' gazebo:=false',
    ])

    robot_description = {
        'robot_description': ParameterValue(robot_xacro_cmd, value_type=str)
    }

    controllers_yaml = LaunchConfiguration('controllers_yaml').perform(context)

    # -----------------------------
    # Table xacro -> table_description
    # -----------------------------
    table_xacro_path = PathJoinSubstitution([
        FindPackageShare("franka_kistar_bringup"),
        "urdf",
        "table.urdf.xacro",
    ])

    table_xacro_cmd = Command([
        FindExecutable(name='xacro'), ' ',
        table_xacro_path,
    ])

    table_description = {
        'robot_description': ParameterValue(table_xacro_cmd, value_type=str)
    }

    # -----------------------------
    # Static TFs
    # world -> base
    # base  -> fr3_link0  (mount orientation)
    # world -> table_link (table pose)
    # -----------------------------
    world_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_base_tf',
        arguments=[
            '--x', LaunchConfiguration('robot_base_x'),
            '--y', LaunchConfiguration('robot_base_y'),
            '--z', LaunchConfiguration('robot_base_z'),
            '--roll',  LaunchConfiguration('robot_base_roll'),
            '--pitch', LaunchConfiguration('robot_base_pitch'),
            '--yaw',   LaunchConfiguration('robot_base_yaw'),
            '--frame-id', LaunchConfiguration('world_frame'),
            '--child-frame-id', 'base',
        ],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static')],
        output='screen',
    )

    base_to_fr3_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_fr3_link0_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'base',
            '--child-frame-id', 'fr3_link0',
        ],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static')],
        output='screen',
    )

    world_to_table_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_table_tf',
        arguments=[
            '--x', LaunchConfiguration('table_x'),
            '--y', LaunchConfiguration('table_y'),
            '--z', LaunchConfiguration('table_z'),
            '--roll',  LaunchConfiguration('table_roll'),
            '--pitch', LaunchConfiguration('table_pitch'),
            '--yaw',   LaunchConfiguration('table_yaw'),
            '--frame-id', LaunchConfiguration('world_frame'),
            '--child-frame-id', LaunchConfiguration('table_frame'),  # default: table_link
        ],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static')],
        output='screen',
    )

    # -----------------------------
    # robot_state_publisher: Robot
    # publish /robot_description (for RViz RobotModel)
    # -----------------------------
    robot_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        parameters=[robot_description, {'publish_robot_description': True}],
        remappings=[
            ('tf', '/tf'),
            ('tf_static', '/tf_static'),
            ('robot_description', '/robot_description'),
        ],
        output='screen',
    )

    # -----------------------------
    # robot_state_publisher: Table
    # publish /table_description (for RViz RobotModel 2nd)
    # -----------------------------
    table_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='table_state_publisher',
        namespace='table',
        parameters=[table_description, {'publish_robot_description': True}],
        remappings=[
            ('tf', '/tf'),
            ('tf_static', '/tf_static'),
            ('robot_description', '/table_description'),
        ],
        output='screen',
    )

    # -----------------------------
    # ros2_control (optional)
    # NOTE: real robot 미연결이면 터짐 -> 기본 false 권장
    # -----------------------------
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        namespace=namespace,
        parameters=[controllers_yaml, robot_description],
        output='screen',
        on_exit=Shutdown(),
        condition=IfCondition(LaunchConfiguration('enable_ros2_control')),
    )

    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=['joint_state_broadcaster'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_ros2_control')),
    )

    franka_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=['franka_robot_state_broadcaster'],
        parameters=[{'arm_id': LaunchConfiguration('arm_id').perform(context)}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_ros2_control')),
    )

    # -----------------------------
    # joint_state_publisher (when ros2_control is OFF)
    # -----------------------------
    jsp_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        namespace=namespace,
        parameters=[robot_description, {'use_robot_description': True}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_joint_state_gui')),
    )

    jsp = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        parameters=[robot_description, {'use_robot_description': True}],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_joint_state_gui')),
    )

    rviz_config_path = PathJoinSubstitution([
        FindPackageShare('franka_kistar_bringup'),
        'rviz',
        LaunchConfiguration('rviz_config')
    ]).perform(context)

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        namespace=namespace,
        arguments=['-d', rviz_config_path],
        parameters=[robot_description],  # robot은 기존대로
        output='screen',
    )

    return [
        world_to_base_tf,
        world_to_table_tf,
        base_to_fr3_tf,
        table_rsp,
        robot_rsp,
        rviz_node,
        ros2_control_node,
        jsb_spawner,
        franka_state_spawner,
        jsp_gui,
        jsp,
    ]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument('arm_id', default_value='fr3'),
        DeclareLaunchArgument('arm_prefix', default_value=''),
        DeclareLaunchArgument('namespace', default_value=''),

        DeclareLaunchArgument('robot_ip', default_value='172.16.0.3'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false'),
        DeclareLaunchArgument('fake_sensor_commands', default_value='false'),

        DeclareLaunchArgument(
            'controllers_yaml',
            default_value=PathJoinSubstitution([
                FindPackageShare('franka_bringup'),
                'config',
                'controllers.yaml'
            ])
        ),

        # ---- frames ----
        DeclareLaunchArgument('world_frame', default_value='world'),

        # ---- robot base pose (world -> base) ----
        DeclareLaunchArgument('robot_base_x', default_value='0.066'),
        DeclareLaunchArgument('robot_base_y', default_value='-0.122'),
        DeclareLaunchArgument('robot_base_z', default_value='0.099'),
        DeclareLaunchArgument('robot_base_roll',  default_value='0.785'),
        DeclareLaunchArgument('robot_base_pitch', default_value='0.0'),
        DeclareLaunchArgument('robot_base_yaw',   default_value='0.0'),

        # ---- table pose (world -> table_link) ----
        DeclareLaunchArgument('table_frame', default_value='table_link'),
        DeclareLaunchArgument('table_x', default_value='0.0'),
        DeclareLaunchArgument('table_y', default_value='0.0'),
        DeclareLaunchArgument('table_z', default_value='0.0'),
        DeclareLaunchArgument('table_roll',  default_value='0.0'),
        DeclareLaunchArgument('table_pitch', default_value='0.0'),
        DeclareLaunchArgument('table_yaw',   default_value='0.0'),

        # ---- switches ----
        DeclareLaunchArgument('enable_ros2_control', default_value='false'),  # <- 현실적으로 기본 false 추천
        DeclareLaunchArgument('use_joint_state_gui', default_value='true'),

        DeclareLaunchArgument(
            'rviz_config',
            default_value='fr3_kistar.rviz',
            description='RViz config filename inside franka_kistar_bringup/rviz'
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=generate_robot_nodes)])
