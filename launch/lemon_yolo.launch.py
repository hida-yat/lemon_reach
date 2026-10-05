import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    手首の D435 でLemonを検出する yolo_ros を起動する (lemon_harvest.launch.py とは別に起動)。

    セグメンテーションモデルを使うと、detect_3d_node は深度をマスク内の画素だけから
    取るので、葉や背景の深度が混ざりにくい。3D位置はカメラ座標のまま出させ、
    base_link への変換は lemon_target_node が撮影時刻のTFで行う
    (detect_3d_node は最新のTFで変換するため)。

    lemon_ellipse_node が検出ごとにマスクへ楕円を当てはめて短辺・長辺を求め、
    /lemon_ellipse/lemons に配信する (lemon_target の入力なので常に起動する)。
    短辺・長辺は YOLO の debug 画像に重ねた /lemon_ellipse/dbg_image と、
    RViz 用の /lemon_ellipse/markers で表示する。
    """

    yolo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("yolo_bringup"),
                "launch",
                "yolo.launch.py",
            )
        ),
        launch_arguments={
            "model": LaunchConfiguration("model"),
            "device": LaunchConfiguration("device"),
            "threshold": LaunchConfiguration("threshold"),
            "input_image_topic": "/realsense/color/image_raw",
            "use_3d": "True",
            "input_depth_topic": "/realsense/depth/image_rect_raw",
            "input_depth_info_topic": "/realsense/color/camera_info",
            "target_frame": "d435_color_optical_frame",
            # Isaac Simの深度は 32FC1 [m]
            "depth_image_units_divisor": "1",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "model",
            default_value=os.path.expanduser(
                "~/devel/lemon-seg/runs/segment/train/weights/best.pt"
            ),
            description="YOLO segmentation model (class 'lemon')",
        ),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("threshold", default_value="0.5"),
        DeclareLaunchArgument(
            "publish_debug_image",
            default_value="true",
            description="Publish the ellipse overlay on /lemon_ellipse/dbg_image",
        ),
        yolo,
        Node(
            package="lemon_reach",
            executable="lemon_ellipse_node",
            parameters=[{
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
            }],
            output="screen",
        ),
    ])
