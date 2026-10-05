import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """
    収穫に使うノードを起動する。YOLO は含まないので、先に別途起動しておく
    (例: ros2 launch lemon_reach lemon_yolo.launch.py)。
    """

    use_sim_time = LaunchConfiguration("use_sim_time")
    params = [{"use_sim_time": use_sim_time}]
    target_params = [{
        "use_sim_time": use_sim_time,
        "score_threshold": ParameterValue(
            LaunchConfiguration("score_threshold"), value_type=float
        ),
    }]
    # 指先をレモン中心よりどれだけ奥まで入れるか。approach はその先 3 cm まで
    # 直線で入れられることを確かめる (insert_margin >= grasp_depth + 0.03)
    approach_params = [{
        "use_sim_time": use_sim_time,
        "insert_margin": ParameterValue(
            LaunchConfiguration("insert_margin"), value_type=float
        ),
    }]

    # lemon_grasp_node は Servo を同じプロセスで動かすのでロボットモデルが要る
    moveit_config = MoveItConfigsBuilder(
        "piper", package_name="piper_with_gripper_moveit"
    ).to_moveit_configs()
    servo_yaml = os.path.join(
        get_package_share_directory("lemon_grasp"), "config", "servo.yaml"
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "score_threshold",
            default_value="0.5",
            description="Minimum YOLO score used by lemon_target",
        ),
        DeclareLaunchArgument(
            "grasp_depth",
            default_value="0.03",
            description="How far the fingertips go beyond the lemon center [m]",
        ),
        DeclareLaunchArgument(
            "insert_margin",
            default_value="0.06",
            description="Straight insertion checked beyond the lemon center [m] "
                        "(keep >= grasp_depth + 0.03)",
        ),
        Node(
            package="lemon_reach",
            executable="lemon_target_node",
            parameters=target_params,
            output="screen",
        ),
        Node(
            package="lemon_reach",
            executable="lemon_approach_node",
            parameters=approach_params,
            output="screen",
        ),
        Node(
            package="lemon_grasp",
            executable="lemon_grasp_node",
            name="lemon_grasp",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
                servo_yaml,
                {
                    "use_sim_time": use_sim_time,
                    "grasp_depth": ParameterValue(
                        LaunchConfiguration("grasp_depth"), value_type=float
                    ),
                },
            ],
            # PoseTracking が購読する target_pose をノードの名前空間に入れる
            remappings=[("target_pose", "/lemon_grasp/target_pose")],
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
