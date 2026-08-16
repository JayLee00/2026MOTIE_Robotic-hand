"""
Grasp_fruit MoveIt 론치 — dual 모델, **right_arm 동작 전용** (손 노드 제외).

dual_fr3_kistar_{planning_pc_v2, moveit, all}.launch.py 구성을 참고해 재작성.
- 로봇 모델 : dual_fr3_kistar_v2 (좌우 팔 + 손 포함). group: left_arm/right_arm/both_arms.
              손은 모델에 그대로 두되(수동), 계획/실행은 right_arm 만 사용(arm.yaml group_name=right_arm).
- 손 노드   : hand_gui_bridge / joint_state_publisher_gui(hand sliders) — **제외** (원본도 기본 OFF).
- 실행      : 팔 트래젝토리 실행이 필요하면 trajectory_bridge_right 유지(오른팔만). 왼팔 bridge 제외.
              (우리 grasp.py 는 /franka/right/q_target 로 직접 실행 — bridge 없이도 동작)
- 기능 유지 : RViz + 테이블/ttable TF·RSP + scene_boxes(충돌박스) + 카메라 TF + realsense(옵션) + world TF.

현재 자세: joint_state_merger.py 가 /joint_states_l+/joint_states_r → /joint_states_relay (dual 모델명으로
           remap, 손가락 encoder→rad 스케일). move_group/RSP 가 /joint_states → /joint_states_relay 로 구독.

전제 소싱: /opt/ros/humble + fr_ws/install(franka_description) + dex_soldering kistar_ws/install
"""

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

_DESC   = 'franka_kistar_description'
_MOVEIT = 'franka_kistar_moveit_config'
_BRINGUP = 'franka_kistar_bringup'


def _load_yaml(pkg, rel):
    with open(os.path.join(get_package_share_directory(pkg), rel)) as f:
        return yaml.safe_load(f)


def _load_file(pkg, rel):
    with open(os.path.join(get_package_share_directory(pkg), rel)) as f:
        return f.read()


def _tf(name, x, y, z, roll, pitch, yaw, parent, child):
    return Node(
        package='tf2_ros', executable='static_transform_publisher', name=name,
        arguments=['--x', x, '--y', y, '--z', z,
                   '--roll', roll, '--pitch', pitch, '--yaw', yaw,
                   '--frame-id', parent, '--child-frame-id', child],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static')], output='screen')


def generate_launch_description():
    ns          = LaunchConfiguration('namespace')
    use_rviz    = LaunchConfiguration('use_rviz')
    use_camera  = LaunchConfiguration('use_camera')
    world_frame = LaunchConfiguration('world_frame')
    table_frame = LaunchConfiguration('table_frame')
    ttable_frame = LaunchConfiguration('ttable_frame')
    profile_frame = LaunchConfiguration('profile_frame')
    front_camera_link = LaunchConfiguration('front_camera_link')
    side_camera_link = LaunchConfiguration('side_camera_link')

    # ── Robot description (dual v2, ros2_control 비활성) ─────────────────────────
    dual_xacro = os.path.join(get_package_share_directory(_DESC), 'urdf', 'dual_fr3_kistar_v2.urdf.xacro')
    robot_description = {'robot_description': ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', dual_xacro,
                 ' ros2_control:=false', ' use_fake_hardware:=true', ' fake_sensor_commands:=true']),
        value_type=str)}
    robot_description_semantic = {'robot_description_semantic':
                                  _load_file(_MOVEIT, 'config/dual_fr3_kistar_v2.srdf')}
    kinematics_yaml = _load_yaml(_MOVEIT, 'config/dual_v2_kinematics.yaml')
    robot_description_planning = {'robot_description_planning':
                                  _load_yaml(_MOVEIT, 'config/dual_v2_joint_limits.yaml')}

    ompl_config = {'move_group': {
        'planning_plugin': 'ompl_interface/OMPLPlanner',
        'request_adapters':
            'default_planner_request_adapters/AddTimeOptimalParameterization '
            'default_planner_request_adapters/ResolveConstraintFrames '
            'default_planner_request_adapters/FixWorkspaceBounds '
            'default_planner_request_adapters/FixStartStateBounds '
            'default_planner_request_adapters/FixStartStateCollision '
            'default_planner_request_adapters/FixStartStatePathConstraints',
        'start_state_max_bounds_error': 0.1,
    }}
    ompl_config['move_group'].update(_load_yaml(_MOVEIT, 'config/dual_v2_ompl_planning.yaml'))

    # TOTG 시간 파라미터화: /move_action 계획(HOME 관절계획, OMPL 폴백)을
    # 100Hz(10ms) 로 리샘플 → 부드러운 실행. (Cartesian 서비스에는 adapter 가
    # 적용되지 않으므로 grasp.py _add_timestamps 가 같은 역할을 담당)
    totg_params = {
        'time_optimal_trajectory_generation.resample_dt': 0.01,
        'time_optimal_trajectory_generation.path_tolerance': 0.1,
        'time_optimal_trajectory_generation.min_angle_change': 0.001,
    }

    moveit_controllers = {
        'moveit_simple_controller_manager': _load_yaml(_MOVEIT, 'config/dual_v2_controllers.yaml'),
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }
    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.execution_duration_monitoring': False,
        'trajectory_execution.allowed_execution_duration_scaling': 10.0,
        'trajectory_execution.allowed_goal_duration_margin': 5.0,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }
    planning_scene_monitor = {
        'publish_planning_scene': True, 'publish_geometry_updates': True,
        'publish_state_updates': True, 'publish_transforms_updates': True,
        'publish_robot_description_semantic': True, 'publish_robot_state': True,
        'publish_robot_state_frequency': 50.0, 'monitor_dynamics': False,
    }

    # ── 테이블 URDF ─────────────────────────────────────────────────────────────
    table_desc = {'robot_description': ParameterValue(Command([FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([FindPackageShare(_BRINGUP), 'urdf', 'table.urdf.xacro'])]), value_type=str)}
    ttable_desc = {'robot_description': ParameterValue(Command([FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([FindPackageShare(_BRINGUP), 'urdf', 'ttable.urdf.xacro'])]), value_type=str)}

    # ── Static TFs (원본 dual launch 값) ────────────────────────────────────────
    world_to_base = _tf('world_to_base_tf', '0', '0', '0', '0', '0', '0', world_frame, 'base')
    world_to_table = _tf('world_to_table_tf', LaunchConfiguration('table_x'), LaunchConfiguration('table_y'),
                         LaunchConfiguration('table_z'), LaunchConfiguration('table_roll'),
                         LaunchConfiguration('table_pitch'), LaunchConfiguration('table_yaw'),
                         world_frame, table_frame)
    world_to_ttable = _tf('world_to_ttable_tf', LaunchConfiguration('ttable_x'), LaunchConfiguration('ttable_y'),
                          LaunchConfiguration('ttable_z'), LaunchConfiguration('table_roll'),
                          LaunchConfiguration('table_pitch'), LaunchConfiguration('table_yaw'),
                          world_frame, ttable_frame)
    profile_tf = _tf('profile_tf', '0.025', '-0.0', '0.0', '0.', '0.', '0.', world_frame, profile_frame)
    front_camera_tf = _tf('front_camera_tf',
                          LaunchConfiguration('front_camera_x'), LaunchConfiguration('front_camera_y'),
                          LaunchConfiguration('front_camera_z'), LaunchConfiguration('front_camera_roll'),
                          LaunchConfiguration('front_camera_pitch'), LaunchConfiguration('front_camera_yaw'),
                          world_frame, front_camera_link)
    side_camera_tf = _tf('side_camera_tf', '0.55', '-0.7', '0.6', '0.', '0.4363', '1.5708',
                         world_frame, side_camera_link)
    sw_wall_tf = _tf('sw_wall_tf', '-0.5', '0.', '0.', '0.', '0.', '0.', table_frame, 'sw_wall_frame')

    # ── RealSense front (옵션) ──────────────────────────────────────────────────
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    realsense_front = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare(_BRINGUP), 'launch', 'realsense_front.launch.py'])),
        launch_arguments={'enable_color': 'true', 'enable_depth': 'true'}.items(),
        condition=IfCondition(use_camera))

    # ── Robot / table state publishers ──────────────────────────────────────────
    robot_rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', namespace=ns,
        parameters=[robot_description, {'publish_robot_description': True}],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static'),
                    ('robot_description', '/robot_description'),
                    ('/joint_states', '/joint_states_relay')],
        output='screen')
    table_rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='table_state_publisher', namespace='table',
        parameters=[table_desc, {'publish_robot_description': True}],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static'),
                    ('robot_description', '/table_description')], output='screen')
    ttable_rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='ttable_state_publisher', namespace='ttable',
        parameters=[ttable_desc, {'publish_robot_description': True}],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static'),
                    ('robot_description', '/ttable_description')], output='screen')

    # ── Joint state merger (direct): /joint_states_l+/joint_states_r → /joint_states_relay ─
    joint_state_relay = Node(
        package=_BRINGUP, executable='joint_state_merger.py', name='joint_state_relay',
        namespace=ns, output='screen',
        parameters=[{
            'input_topics': ['/joint_states_l', '/joint_states_r'],
            'output_topic': '/joint_states_relay', 'publish_rate': 50.0,
            'joint_name_remap': 'fr3_l_:left_fr3_,fr3_r_:right_fr3_,l_:left_,r_:right_',
            'joint_position_scale': ('index_:3.834951969714103e-4,thumb_:3.834951969714103e-4,'
                                     'middle_:3.834951969714103e-4,ring_:3.834951969714103e-4'),
        }])
    joint_states_observer = Node(
        package=_BRINGUP, executable='joint_states_observer.py', name='joint_states_observer',
        namespace=ns, output='screen', parameters=[{'mode': 'direct'}],
        condition=IfCondition(LaunchConfiguration('joint_states_observer')))

    # ── move_group (dual 모델, /joint_states → /joint_states_relay) ──────────────
    move_group_node = Node(
        package='moveit_ros_move_group', executable='move_group', namespace=ns, output='screen',
        parameters=[robot_description, robot_description_semantic, robot_description_planning,
                    kinematics_yaml, ompl_config, totg_params, trajectory_execution,
                    moveit_controllers, planning_scene_monitor],
        remappings=[('/joint_states', '/joint_states_relay')])

    # ── Trajectory bridge (오른팔만; 왼팔 bridge 제외) ──────────────────────────
    trajectory_bridge_right = Node(
        package=_BRINGUP, executable='trajectory_bridge.py', name='trajectory_bridge_right',
        namespace=ns, output='screen',
        parameters=[{'moveit_action': '/right_arm_controller/follow_joint_trajectory',
                     'robot_action': '/fr3_r_arm_controller/follow_joint_trajectory',
                     'joint_name_remap': 'right_fr3_:fr3_r_,right_:r_'}])

    # ── Planning scene collision boxes (테이블/카메라 박스) ─────────────────────
    scene_boxes = Node(
        package=_BRINGUP, executable='planning_scene_static_boxes.py', output='screen',
        parameters=[{
            'world_frame': world_frame, 'table_frame': table_frame, 'ttable_frame': ttable_frame,
            'profile_frame': profile_frame,
            'table_size': LaunchConfiguration('table_size'), 'ttable_size': LaunchConfiguration('ttable_size'),
            'side_camera_frame': side_camera_link, 'side_camera_size': [0.15, 0.15, 0.6],
            'side_camera_z_offset': -0.30, 'timeout_sec': 60.0,
        }])
    scene_boxes_delayed = RegisterEventHandler(
        OnProcessStart(target_action=move_group_node, on_start=[scene_boxes]))
# 
    # ── RViz (move_group 기동 5s 후) ────────────────────────────────────────────
    # rviz_cfg = os.path.join(get_package_share_directory(_BRINGUP), 'rviz', 'fr3_kistar.rviz')
    rviz_cfg = os.path.join(get_package_share_directory(_BRINGUP), 'rviz', 'fr3_kistar_camera.rviz')
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz', namespace=ns,
        arguments=['-d', rviz_cfg],
        parameters=[robot_description, robot_description_semantic, robot_description_planning,
                    ompl_config, kinematics_yaml],
        output='screen', condition=IfCondition(use_rviz))
    rviz_delayed = RegisterEventHandler(
        OnProcessStart(target_action=move_group_node,
                       on_start=[TimerAction(period=5.0, actions=[rviz_node])]))

    args = [
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_camera', default_value='false'),
        DeclareLaunchArgument('joint_states_observer', default_value='true'),
        DeclareLaunchArgument('world_frame', default_value='world'),
        DeclareLaunchArgument('table_frame', default_value='table_link'),
        DeclareLaunchArgument('table_x', default_value='0.0'),
        DeclareLaunchArgument('table_y', default_value='0.032'),
        DeclareLaunchArgument('table_z', default_value='0.0'),
        DeclareLaunchArgument('table_roll', default_value='0.0'),
        DeclareLaunchArgument('table_pitch', default_value='0.0'),
        DeclareLaunchArgument('table_yaw', default_value='0.0'),
        DeclareLaunchArgument('ttable_frame', default_value='ttable_link'),
        DeclareLaunchArgument('ttable_x', default_value='0.5'),
        DeclareLaunchArgument('ttable_y', default_value='0.0'),
        DeclareLaunchArgument('ttable_z', default_value='0.205'),
        DeclareLaunchArgument('table_size', default_value='[1.2, 1.8, 0.05]'),
        DeclareLaunchArgument('ttable_size', default_value='[0.5, 0.8, 0.03]'),
        DeclareLaunchArgument('profile_frame', default_value='profile_frame'),
        DeclareLaunchArgument('front_camera_link', default_value='front_camera_link'),
        DeclareLaunchArgument('front_camera_x', default_value='0.12'),
        DeclareLaunchArgument('front_camera_y', default_value='0.02'),
        DeclareLaunchArgument('front_camera_z', default_value='0.75'),
        DeclareLaunchArgument('front_camera_roll', default_value='0.'),
        DeclareLaunchArgument('front_camera_pitch', default_value='0.8727'),
        DeclareLaunchArgument('front_camera_yaw', default_value='0.'),
        DeclareLaunchArgument('side_camera_link', default_value='side_camera_link'),
    ]

    return LaunchDescription(args + [
        # TFs
        world_to_base, world_to_table, world_to_ttable, profile_tf,
        front_camera_tf, side_camera_tf, sw_wall_tf, realsense_front,
        # RSP
        table_rsp, ttable_rsp, robot_rsp,
        # Joint states
        joint_state_relay, joint_states_observer,
        # MoveIt
        move_group_node, scene_boxes_delayed,
        # 오른팔 trajectory bridge
        trajectory_bridge_right,
        # RViz (지연)
        rviz_delayed,
    ])
