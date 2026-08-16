# -----------------------------
# dual_fr3_kistar_planning_pc.launch.py
# PC1 (Planning Computer) - Dual-arm MoveIt Planning + Real-time Tracking
#
# - Single MoveIt instance with left_arm / right_arm / both_arms groups
# - Two trajectory forwarders (left/right)
# - Joint state merger for real-time robot pose tracking
# - Collision checking between both arms is automatic
#
# Usage:
#   # Offline planning (fake joint states):
#   ros2 launch franka_kistar_bringup dual_fr3_kistar_planning_pc.launch.py
#
#   # Real robot tracking via topics (Franka PC runs its own franka_hardware on DOMAIN_ID=9):
#   ROS_DOMAIN_ID=9 ros2 launch franka_kistar_bringup dual_fr3_kistar_planning_pc.launch.py joint_state_mode:=direct
# -----------------------------

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import Shutdown as ShutdownEvent
from launch.substitutions import (
    Command,
    FindExecutable,
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


import hashlib

_VALID_MODES = ("fake", "direct", "merger")
_PLURAL_UNSET_SENTINEL = "__UNSET__"

# Populated by generate_launch_description() before _check_urdf_snapshot_runtime runs.
_urdf_snapshot_statuses = {}


def _read_text(path):
    with open(path, "r") as fh:
        return fh.read()


def _sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_snapshot_or_none(urdf_path):
    """Return (status, urdf_text_or_None) for a snapshot path.

    status is one of:
      'ok'        — snapshot file + sidecar present and hash matches.
      'mismatch'  — both files present but the hash on disk differs from the sidecar.
      'missing'   — either file is absent.
    """
    sidecar = urdf_path + ".sha256"
    if not (os.path.isfile(urdf_path) and os.path.isfile(sidecar)):
        return "missing", None
    expected = _read_text(sidecar).split()[0].strip()
    actual = _sha256_of_file(urdf_path)
    if actual != expected:
        return "mismatch", None
    return "ok", _read_text(urdf_path)


def _check_urdf_snapshot_runtime(context, *args, **kwargs):
    """Runtime OpaqueFunction action: react to the snapshot status that was already
    decided at generate_launch_description() time, gated by the strict_urdf_snapshot arg.
    """
    strict = LaunchConfiguration("strict_urdf_snapshot").perform(context).lower() in (
        "1", "true", "yes", "on",
    )
    actions = []
    bad = [(name, status) for name, status in _urdf_snapshot_statuses.items() if status != "ok"]
    if not bad:
        actions.append(LogInfo(msg="[urdf_snapshot] OK — generated/*.urdf hashes match sidecars."))
        return actions
    for name, status in bad:
        msg = (
            f"[urdf_snapshot] {name} snapshot status='{status}' — "
            "regenerate with scripts/regenerate_urdf.sh."
        )
        actions.append(LogInfo(msg=msg))
    if strict:
        actions.append(
            LogInfo(
                msg=(
                    "[FATAL] strict_urdf_snapshot=true and snapshot(s) are stale or missing. "
                    "Run: ./isaac-ros/kistar_ws/src/franka_kistar_bringup/scripts/regenerate_urdf.sh"
                )
            )
        )
        actions.append(
            EmitEvent(event=ShutdownEvent(reason="URDF snapshot validation failed"))
        )
    else:
        actions.append(
            LogInfo(
                msg=(
                    "[urdf_snapshot] strict_urdf_snapshot=false — proceeding with live xacro "
                    "fallback (the launch already loaded robot_description via Command([xacro,...]))."
                )
            )
        )
    return actions


def _validate_args(context, *args, **kwargs):
    """Read-only validator: rejects unknown/empty canonical and any plural-spelling use.

    Inserted as the first action of the LaunchDescription so that EmitEvent(Shutdown) is
    on the event bus before any conditional node is visited. Downstream substitutions
    read 'joint_state_mode' directly; this OpaqueFunction does not mutate context.
    """
    canonical = LaunchConfiguration("joint_state_mode").perform(context)
    plural = LaunchConfiguration("joint_states_mode").perform(context)
    robot_ip = LaunchConfiguration("robot_ip").perform(context)

    actions = [
        LogInfo(
            msg=(
                "[joint_state_args] canonical='"
                + canonical
                + "' plural='"
                + plural
                + "' robot_ip='"
                + robot_ip
                + "'"
            )
        )
    ]
    reasons = []

    plural_was_passed = plural != _PLURAL_UNSET_SENTINEL
    if plural_was_passed:
        reasons.append(
            "DEPRECATED plural 'joint_states_mode' was passed (='"
            + plural
            + "'). Use 'joint_state_mode' (singular)."
        )
    # Empty-string canonical is REJECTED — it would otherwise silently produce no
    # joint state source (all three IfConditions evaluate False) and hang.
    if canonical not in _VALID_MODES:
        reasons.append(
            "Invalid joint_state_mode value '"
            + canonical
            + "'. Allowed: "
            + str(list(_VALID_MODES))
            + ". Empty string is rejected (was a silent-hang surface)."
        )
    if reasons:
        for r in reasons:
            actions.append(LogInfo(msg="[FATAL] " + r))
        actions.append(
            EmitEvent(event=ShutdownEvent(reason="joint_state arg validation failed"))
        )
    return actions


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
    rviz_source = LaunchConfiguration("rviz_source")
    rviz_config = LaunchConfiguration("rviz_config")

    world_frame = LaunchConfiguration("world_frame")

    # Table TF args
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
    # Robot Description (Dual-arm URDF)
    # PR-2 B.1: prefer pre-compiled snapshot (generated/*.urdf) when available and the
    # SHA256 sidecar matches; otherwise fall back to live xacro Command (original
    # behavior). _check_urdf_snapshot_runtime decides whether stale snapshots are a
    # FATAL Shutdown (strict_urdf_snapshot=true, the default) or a LogWarn fall-back
    # (strict_urdf_snapshot=false). Regenerate snapshots with:
    #     ./isaac-ros/kistar_ws/src/franka_kistar_bringup/scripts/regenerate_urdf.sh
    # -----------------------------
    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_kistar_description"),
        "urdf",
        "dual_fr3_kistar.urdf.xacro",
    )
    franka_urdf_snapshot = os.path.join(
        get_package_share_directory("franka_kistar_description"),
        "urdf",
        "generated",
        "dual_fr3_kistar.urdf",
    )

    _dual_status, _dual_text = _load_snapshot_or_none(franka_urdf_snapshot)
    # Track snapshot statuses for the runtime OpaqueFunction action below.
    _urdf_snapshot_statuses["dual_fr3_kistar"] = _dual_status

    if _dual_status == "ok":
        robot_description = {
            "robot_description": ParameterValue(_dual_text, value_type=str)
        }
    else:
        # Live xacro fallback. The runtime OpaqueFunction will Shutdown if strict
        # and the snapshot is stale/missing; otherwise it logs a warning here.
        robot_description_config = Command(
            [
                FindExecutable(name="xacro"),
                " ",
                franka_xacro_file,
                " ros2_control:=false",
                " use_fake_hardware:=true",
                " fake_sensor_commands:=true",
            ]
        )
        robot_description = {
            "robot_description": ParameterValue(robot_description_config, value_type=str)
        }

    # -----------------------------
    # Semantic (SRDF) - Dual-arm
    # -----------------------------
    franka_semantic_file = os.path.join(
        get_package_share_directory("franka_kistar_moveit_config"),
        "config",
        "dual_fr3_kistar.srdf",
    )

    with open(franka_semantic_file, "r") as f:
        robot_description_semantic_content = f.read()

    robot_description_semantic = {
        "robot_description_semantic": robot_description_semantic_content
    }

    # -----------------------------
    # Planning configs (Dual-arm)
    # -----------------------------
    kinematics_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/dual_kinematics.yaml"
    )

    joint_limits_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/dual_joint_limits.yaml"
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
        "franka_kistar_moveit_config", "config/dual_ompl_planning.yaml"
    )
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

    moveit_simple_controllers_yaml = load_yaml(
        "franka_kistar_moveit_config", "config/dual_controllers.yaml"
    )
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_simple_controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": True,
        # Real-robot safety-limited execution can run far slower than planned MoveIt
        # duration; disabling execution_duration_monitoring makes MoveIt wait for the
        # controller (trajectory_bridge → robot) to deliver the real terminal status
        # instead of guessing from plan duration. Scaling + margin remain as defensive
        # defaults if monitoring is re-enabled later.
        "trajectory_execution.execution_duration_monitoring": False,
        "trajectory_execution.allowed_execution_duration_scaling": 10.0,
        "trajectory_execution.allowed_goal_duration_margin": 5.0,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description_semantic": True,
        "publish_robot_state": True,
        "publish_robot_state_frequency": 50.0,
        "monitor_dynamics": False,
    }

    # -----------------------------
    # Tables (URDF/xacro)
    # -----------------------------
    table_xacro_path = PathJoinSubstitution(
        [FindPackageShare("franka_kistar_bringup"), "urdf", "table.urdf.xacro"]
    )
    table_xacro_cmd = Command([FindExecutable(name="xacro"), " ", table_xacro_path])
    table_description = {
        "robot_description": ParameterValue(table_xacro_cmd, value_type=str)
    }

    ttable_xacro_path = PathJoinSubstitution(
        [FindPackageShare("franka_kistar_bringup"), "urdf", "ttable.urdf.xacro"]
    )
    ttable_xacro_cmd = Command([FindExecutable(name="xacro"), " ", ttable_xacro_path])
    ttable_description = {
        "robot_description": ParameterValue(ttable_xacro_cmd, value_type=str)
    }

    # -----------------------------
    # Static TFs
    # Note: arm base offsets are embedded in the URDF (base → left/right_fr3_link0).
    # Only world → base TF is needed here.
    # -----------------------------
    world_to_base_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_tf",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", world_frame,
            "--child-frame-id", "base",
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    world_to_table_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_table_tf",
        arguments=[
            "--x", table_x, "--y", table_y, "--z", table_z,
            "--roll", table_roll, "--pitch", table_pitch, "--yaw", table_yaw,
            "--frame-id", world_frame,
            "--child-frame-id", table_frame,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    world_to_ttable_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_ttable_tf",
        arguments=[
            "--x", ttable_x, "--y", ttable_y, "--z", ttable_z,
            "--roll", table_roll, "--pitch", table_pitch, "--yaw", table_yaw,
            "--frame-id", world_frame,
            "--child-frame-id", ttable_frame,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    profile_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="profile_tf",
        arguments=[
            "--x", "0.025", "--y", "-0.0", "--z", "0.0",
            "--roll", "0.", "--pitch", "0.", "--yaw", "0.",
            "--frame-id", world_frame,
            "--child-frame-id", profile_frame,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    front_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="front_camera_tf",
        arguments=[
            "--x", "0.12", "--y", "0.02", "--z", "0.88",
            "--roll", "0.", "--pitch", "0.8727", "--yaw", "0.",
            "--frame-id", world_frame,
            "--child-frame-id", front_camera_link,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    side_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="side_camera_tf",
        arguments=[
            "--x", "0.55", "--y", "0.82", "--z", "0.6",
            "--roll", "0.", "--pitch", "0.4363", "--yaw", "-1.5708",
            "--frame-id", world_frame,
            "--child-frame-id", side_camera_link,
        ],
        remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
        output="screen",
    )

    def marker_tf(name, x, y, z, parent, child):
        return Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=name,
            arguments=[
                "--x", x, "--y", y, "--z", z,
                "--roll", "0.", "--pitch", "0.", "--yaw", "0.",
                "--frame-id", parent, "--child-frame-id", child,
            ],
            remappings=[("tf", "/tf"), ("tf_static", "/tf_static")],
            output="screen",
        )

    sw_wall_tf = marker_tf("sw_wall_tf", "-0.5", "0.", "0.", table_frame, "sw_wall_frame")

    # -----------------------------
    # Robot State Publisher
    # -----------------------------
    # joint_states topic: "fake" mode uses joint_states (relative), non-fake uses /joint_states_relay
    joint_states_topic = IfElseSubstitution(
        PythonExpression(["'", LaunchConfiguration("joint_state_mode"), "' == 'fake'"]),
        TextSubstitution(text="joint_states"),
        TextSubstitution(text="/joint_states_relay"),
    )

    # The PROTECTION against the robot PC's fake /joint_states reaching planning consumers
    # is the destination rewrite to /joint_states_relay below — not the leading slash on the
    # remap source. With namespace="", the relative form "joint_states" already resolves to
    # /joint_states and the rewrite still works. The absolute source "/joint_states" is a
    # future defense against namespaced launches; do not strip the slash on stylistic grounds.
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
            ("/joint_states", joint_states_topic),
        ],
        output="screen",
    )

    # Joint State source: "fake" | "direct" | "merger"
    #   fake:   joint_state_publisher (offline planning, no robot)
    #   direct: relay robot's /joint_states with validation → /joint_states_relay
    #   merger: merge /left/joint_states + /right/joint_states → /joint_states_relay
    joint_state_mode = LaunchConfiguration("joint_state_mode")

    # Defense-in-depth: each IfCondition checks mode-membership before equality.
    # If the validator's Shutdown is ever lost (future refactor), bogus values still
    # match no conditional node — no silent partial launch.
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        namespace=namespace,
        parameters=[
            robot_description,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(
            PythonExpression(
                [
                    "'", joint_state_mode, "' in ['fake', 'direct', 'merger'] and ",
                    "'", joint_state_mode, "' == 'fake'",
                ]
            )
        ),
    )

    # direct mode: merge /joint_states_l + /joint_states_r from robot PC,
    # remap joint names (fr3_l_ → left_fr3_, l_ → left_, etc.)
    joint_state_relay = Node(
        package="franka_kistar_bringup",
        executable="joint_state_merger.py",
        name="joint_state_relay",
        namespace=namespace,
        output="screen",
        parameters=[
            {
                "input_topics": ["/joint_states_l", "/joint_states_r"],
                "output_topic": "/joint_states_relay",
                "publish_rate": 50.0,
                "joint_name_remap": "fr3_l_:left_fr3_,fr3_r_:right_fr3_,l_:left_,r_:right_",
            }
        ],
        condition=IfCondition(
            PythonExpression(
                [
                    "'", joint_state_mode, "' in ['fake', 'direct', 'merger'] and ",
                    "'", joint_state_mode, "' == 'direct'",
                ]
            )
        ),
    )

    # merger mode: merge arbitrary left/right topics (custom setup)
    joint_state_merger = Node(
        package="franka_kistar_bringup",
        executable="joint_state_merger.py",
        name="joint_state_merger",
        namespace=namespace,
        output="screen",
        parameters=[
            {
                "input_topics": ["/left/joint_states", "/right/joint_states"],
                "output_topic": "/joint_states_relay",
                "publish_rate": 50.0,
                "joint_name_remap": "",
            }
        ],
        condition=IfCondition(
            PythonExpression(
                [
                    "'", joint_state_mode, "' in ['fake', 'direct', 'merger'] and ",
                    "'", joint_state_mode, "' == 'merger'",
                ]
            )
        ),
    )

    # Observability-only diagnostic (default-on in direct/merger modes).
    # DDS multi-subscriber semantics mean this node CANNOT block other subscribers from
    # receiving /joint_states — it is an early-warning surface, not a safety control.
    # Real protection is the destination rewrite on robot_rsp/move_group remaps above.
    joint_states_observer = Node(
        package="franka_kistar_bringup",
        executable="joint_states_observer.py",
        name="joint_states_observer",
        namespace=namespace,
        output="screen",
        parameters=[
            {"mode": joint_state_mode},
        ],
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    LaunchConfiguration("joint_states_observer"),
                    "' == 'true' and '",
                    joint_state_mode,
                    "' in ['direct', 'merger']",
                ]
            )
        ),
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
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
        ],
        # See robot_rsp note above on why the absolute-form source matters.
        remappings=[("/joint_states", joint_states_topic)],
    )

    # -----------------------------
    # RViz
    # -----------------------------
    kistar_rviz_path = PathJoinSubstitution(
        [FindPackageShare("franka_kistar_bringup"), "rviz", rviz_config]
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
                # Generous wait budget — move_group's first-load can take >20s with
                # large SRDFs; the script's wait_for_service handles the actual
                # readiness, this just prevents the "Giving up" path from firing early.
                "timeout_sec": 60.0,
            }
        ],
    )

    # Start scene_boxes only after move_group's process starts; its internal
    # wait_for_service handles the remaining latency until /apply_planning_scene is up.
    scene_boxes_delayed = RegisterEventHandler(
        OnProcessStart(target_action=run_move_group_node, on_start=[scene_boxes_node])
    )

    # -----------------------------
    # Trajectory Bridges (left + right)
    # Remap URDF joint names → robot joint names, forward to robot's
    # trajectory_receiver action server (which writes to SHM → RTOS → motor)
    # -----------------------------
    trajectory_bridge_left = Node(
        package="franka_kistar_bringup",
        executable="trajectory_bridge.py",
        name="trajectory_bridge_left",
        namespace=namespace,
        output="screen",
        parameters=[
            {
                "moveit_action": "/left_arm_controller/follow_joint_trajectory",
                "robot_action": "/fr3_l_arm_controller/follow_joint_trajectory",
                "joint_name_remap": "left_fr3_:fr3_l_,left_:l_",
            }
        ],
    )

    trajectory_bridge_right = Node(
        package="franka_kistar_bringup",
        executable="trajectory_bridge.py",
        name="trajectory_bridge_right",
        namespace=namespace,
        output="screen",
        parameters=[
            {
                "moveit_action": "/right_arm_controller/follow_joint_trajectory",
                "robot_action": "/fr3_r_arm_controller/follow_joint_trajectory",
                "joint_name_remap": "right_fr3_:fr3_r_,right_:r_",
            }
        ],
    )

    # -----------------------------
    # Launch args
    # -----------------------------
    launch_args = [
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        # CANONICAL singular spelling. Matches docs/run/run_real_moveit.md and the operator
        # command. The legacy plural is detected and rejected by _validate_args — silent
        # fake-mode fallback on the real robot is no longer possible.
        DeclareLaunchArgument(
            "joint_state_mode",
            default_value="fake",
            description=(
                "fake | direct | merger. Canonical SINGULAR spelling. "
                "The legacy plural 'joint_states_mode' is detected and rejected (does NOT "
                "silently fall back to 'fake'). Empty string is also rejected."
            ),
        ),
        # Stowaway: legacy plural is declared ONLY so the validator can detect & reject it.
        # Lint script asserts this is the unique occurrence of the plural literal.
        DeclareLaunchArgument(
            "joint_states_mode",  # __plural_rejected_sentinel__
            default_value="__UNSET__",
            description=(
                "DEPRECATED — use joint_state_mode (singular). Any non-default value "
                "triggers loud Shutdown."
            ),
        ),
        # Operator's documented command at docs/run/run_real_moveit.md:31 passes robot_ip;
        # declare it so it is logged at startup rather than silently dropped (the launch
        # file does not route to libfranka; the robot side runs on ROS_DOMAIN_ID=9).
        DeclareLaunchArgument(
            "robot_ip",
            default_value="",
            description=(
                "Robot PC IP. INFORMATIONAL ONLY on planning PC (the robot lives on "
                "ROS_DOMAIN_ID=9). Declared so passing it does not silently disappear; "
                "the validator logs the received value at startup."
            ),
        ),
        DeclareLaunchArgument(
            "joint_states_observer",
            default_value="true",
            description=(
                "Observability only — DiagnosticStatus WARN if /joint_states has "
                "publishers in direct/merger mode. Cannot block subscribers "
                "(DDS multi-sub semantics). Default ON in direct/merger."
            ),
        ),
        DeclareLaunchArgument(
            "strict_urdf_snapshot",
            default_value="true",
            description=(
                "PR-2 B.1: if true (default + CI), the launch FATAL-Shuts down when the "
                "pre-compiled generated/*.urdf hash differs from its sidecar. If false, "
                "the launch logs a WARN and falls back to live xacro Command (field "
                "recovery; documented in docs/run/run_real_moveit.md)."
            ),
        ),
        DeclareLaunchArgument("rviz_source", default_value="kistar", description="kistar or moveit"),
        DeclareLaunchArgument("rviz_config", default_value="fr3_kistar.rviz"),
        # frames
        DeclareLaunchArgument("world_frame", default_value="world"),
        # table pose
        DeclareLaunchArgument("table_frame", default_value="table_link"),
        DeclareLaunchArgument("table_x", default_value="0.0"),
        DeclareLaunchArgument("table_y", default_value="0.032"),
        DeclareLaunchArgument("table_z", default_value="0.0"),
        DeclareLaunchArgument("ttable_frame", default_value="ttable_link"),
        DeclareLaunchArgument("ttable_x", default_value="0.5"),
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

    # OpaqueFunction action — runs first so EmitEvent(Shutdown) reaches the event bus
    # before any conditional node is visited. Read-only (no context mutation).
    validate_args = OpaqueFunction(function=_validate_args)
    check_urdf_snapshot = OpaqueFunction(function=_check_urdf_snapshot_runtime)

    # PR-2 B.5: RViz starts 5s after move_group process start. The extra delay is not
    # for AC4 (RViz never contributed to headless_ready_max) — it gives move_group time
    # to publish the SRDF/planning_group list before the MotionPlanning panel renders
    # its dropdown. Without this delay operators see "No planning group" in the panel
    # even though move_group is healthy.
    rviz_after_move_group = RegisterEventHandler(
        OnProcessStart(
            target_action=run_move_group_node,
            on_start=[TimerAction(period=5.0, actions=[rviz_node])],
        )
    )

    return LaunchDescription(
        launch_args
        + [
            # FIRST: validate launch args (rejects empty/bogus mode + plural spelling).
            validate_args,
            # SECOND: snapshot status emit (no-op when OK; FATAL when strict + stale).
            check_urdf_snapshot,
            # TFs
            world_to_base_tf,
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
            # Joint States (fake / direct relay / merger — mutually exclusive)
            joint_state_publisher,
            joint_state_relay,
            joint_state_merger,
            # Observability (default-on in direct/merger; cannot block subscribers).
            joint_states_observer,
            # MoveIt
            run_move_group_node,
            # Trajectory Bridges (left + right)
            trajectory_bridge_left,
            trajectory_bridge_right,
            # RViz starts after move_group is ready (UX-only; zero AC4 cost).
            rviz_after_move_group,
        ]
    )
