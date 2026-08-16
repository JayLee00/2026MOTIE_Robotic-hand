
# Standalone launch for the front D435i (serial 846112071515).
#
# Usage:
#   ros2 launch franka_kistar_bringup realsense_front.launch.py
#
# The driver prefixes base_frame_id with camera_name, so the default
# "camera_link" yields a TF root of "front_camera_link" — chaining the camera
# under the world -> front_camera_link static TF published by
# dual_fr3_kistar_planning_pc_v2.launch.py. Do NOT pass "front_camera_link"
# here: it would become "front_front_camera_link" and disconnect the tree.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # rs_launch.py needs serial_no as a quoted string literal (matches
    # realsense_multi.launch.py's f"'{serial}'"), so resolve it here rather
    # than forwarding the LaunchConfiguration substitution directly.
    serial_value = LaunchConfiguration("serial").perform(context)

    include_rs = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])
        ),
        launch_arguments={
            "serial_no": f"'{serial_value}'",
            "camera_namespace": LaunchConfiguration("camera_namespace"),
            "camera_name": LaunchConfiguration("camera_name"),
            "base_frame_id": LaunchConfiguration("base_frame_id"),
            "publish_tf": "true",
            "tf_publish_rate": "0.0",
            "pointcloud.enable": LaunchConfiguration("pointcloud"),
            "enable_color": LaunchConfiguration("enable_color"),
            "enable_depth": LaunchConfiguration("enable_depth"),
            # realsense-ros v4.5x dropped color_width/color_fps-style args; profiles
            # must be passed as WxHxFPS strings or the driver silently runs its
            # defaults (color 1280x720@30 + depth 848x480@30 — the "heavy RViz"
            # culprit). Keep both streams at VGA/30fps.
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
            # align_depth publishes /aligned_depth_to_color/image_raw (depth warped
            # into the color frame). Off by default in realsense-ros, so forward it
            # explicitly or the topic never appears.
            "align_depth.enable": LaunchConfiguration("align_depth"),
            "initial_reset": "true",
            "clip_distance": "1.3",
            # --- Depth-loss optimization for the molded fiber fruit tray ---
            # The glossy white pulp tray defeats the D435i IR stereo: with depth
            # auto-exposure the driver ran the exposure up to 8500us, saturating
            # the projected IR pattern on the specular surface, so ~62-76% of the
            # tray's depth pixels dropped out (measured inside a SAM3 mask of the
            # tray). A per-mask parameter sweep found the minimum-loss operating
            # point: cap the depth exposure at 1500us (removes the specular IR
            # saturation) and enable the temporal filter (accumulates the
            # flickering-but-real returns across frames). This cut the tray's
            # depth loss from ~62% to <1% with accurate, low-noise values
            # (tray median 591mm, temporal std 0.7mm; orange-anchor unchanged at
            # ~730mm). Laser power is left at the default 150 (raising it
            # over-saturates the white surface -> worse), and hole_filling is
            # deliberately NOT enabled: it fabricates depth and wedges the
            # post-processing thread in this realsense-ros build.
            "depth_module.enable_auto_exposure": LaunchConfiguration("depth_auto_exposure"),
            "depth_module.exposure": LaunchConfiguration("depth_exposure"),
            "temporal_filter.enable": LaunchConfiguration("temporal_filter"),
        }.items(),
    )
    return [include_rs]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("enable_color", default_value="true"),
        DeclareLaunchArgument("enable_depth", default_value="true"),
        DeclareLaunchArgument("pointcloud", default_value="true"),
        DeclareLaunchArgument("align_depth", default_value="true"),
        DeclareLaunchArgument("color_profile", default_value="640x480x30"),
        DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
        # Depth-loss optimization for the white molded fiber tray (see launch_setup).
        # Override at runtime, e.g. depth_exposure:=2000, to re-tune for a scene.
        DeclareLaunchArgument("depth_auto_exposure", default_value="false"),
        DeclareLaunchArgument("depth_exposure", default_value="1500"),
        DeclareLaunchArgument("temporal_filter", default_value="true"),
        DeclareLaunchArgument("serial", default_value="_846112071515"),
        DeclareLaunchArgument("camera_namespace", default_value="front_cam"),
        DeclareLaunchArgument("camera_name", default_value="front"),
        # Driver prefixes camera_name: "camera_link" -> root frame "front_camera_link".
        DeclareLaunchArgument("base_frame_id", default_value="camera_link"),
        OpaqueFunction(function=launch_setup),
    ])
