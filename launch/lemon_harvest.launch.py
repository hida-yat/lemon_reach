import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_yolo = LaunchConfiguration("use_yolo")
    params = [{"use_sim_time": use_sim_time}]

    # yolo_ros (2D検出 + detect_3d_node で base_link 座標の3D位置)
    yolo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("yolo_bringup"),
                "launch",
                "yolov8.launch.py",
            )
        ),
        launch_arguments={
            "use_3d": "True",
            "input_depth_topic": "/realsense/depth/image_rect_raw",
            "input_depth_info_topic": "/realsense/color/camera_info",
            "target_frame": "base_link",
            # Isaac Simの深度は 32FC1 [m]
            "depth_image_units_divisor": "1",
        }.items(),
        condition=IfCondition(use_yolo),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "use_yolo",
            default_value="true",
            description="Whether to launch yolo_ros together",
        ),
        yolo,
        Node(
            package="lemon_reach",
            executable="lemon_target_node",
            parameters=params,
            output="screen",
        ),
        Node(
            package="lemon_reach",
            executable="lemon_approach_node",
            parameters=params,
            output="screen",
        ),
        Node(
            package="lemon_reach",
            executable="arm_home_node",
            parameters=params,
            output="screen",
        ),
        Node(
            package="lemon_reach",
            executable="lemon_harvest_manager_node",
            parameters=params,
            output="screen",
        ),
    ])
