
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def rs(serial, ns, name, base_frame, enable_pointcloud="true"):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])
        ),
        launch_arguments={
            "serial_no": f"'{serial}'",
            "camera_namespace": ns,     # 토픽 분리
            "camera_name": name,        # frame prefix 분리
            "base_frame_id": base_frame,  # 로봇 TF에 붙일 마운트 프레임
            "publish_tf": "true",
            "tf_publish_rate": "0.0",
            "pointcloud.enable": enable_pointcloud,
            "enable_color": "true",
            "enable_depth": "true",
            "color_width": "640",
            "color_height": "480",
            "color_fps": "15",
            "depth_width": "640",
            "depth_height": "480",
            "depth_fps": "15",
            "initial_reset": "true",
            "clip_distance": "1.3",
            # "publish_tf": "false",
        }.items(),
    )

def generate_launch_description():
    return LaunchDescription([
        rs("_846112071515", "front_cam", "front", "camera_link", "true"),
        rs("_033422070379", "side_cam", "side", "camera_link", "true"),
    ])
