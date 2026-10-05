#!/usr/bin/env python3

import threading

from rclpy.action import ActionClient

from builtin_interfaces.msg import Duration
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes, PlanningSceneComponents
from moveit_msgs.srv import GetMotionPlan, GetPlanningScene, GetPositionIK
from std_srvs.srv import Empty


class MoveItHelper:
    """
    MoveIt (move_group) の計画・実行を同期的に呼ぶためのヘルパー。

    Service / Action の完了をスレッドで待つので、ノードは
    MultiThreadedExecutor + ReentrantCallbackGroup で spin すること。
    """

    def __init__(self, node, callback_group):

        self.node = node

        node.declare_parameter("planning_group", "arm")
        # 再求解の回数と1回あたりの計画時間
        node.declare_parameter("max_plan_attempts", 3)
        node.declare_parameter("planning_time", 5.0)
        node.declare_parameter("moveit_num_planning_attempts", 10)
        node.declare_parameter("velocity_scaling", 0.1)
        node.declare_parameter("acceleration_scaling", 0.1)
        # 計画前にoctomapを消し、settle秒待って現在の視野で作り直させる
        node.declare_parameter("clear_octomap", True)
        node.declare_parameter("octomap_settle_sec", 1.0)
        # 起動直後はMoveItのサービスが見つかるまで時間がかかる
        node.declare_parameter("service_timeout", 5.0)
        node.declare_parameter("execute_timeout", 60.0)
        node.declare_parameter("ik_timeout", 0.05)

        self.planning_group = node.get_parameter("planning_group").value
        self.max_plan_attempts = node.get_parameter("max_plan_attempts").value
        self.planning_time = node.get_parameter("planning_time").value
        self.moveit_num_planning_attempts = (
            node.get_parameter("moveit_num_planning_attempts").value
        )
        self.velocity_scaling = node.get_parameter("velocity_scaling").value
        self.acceleration_scaling = (
            node.get_parameter("acceleration_scaling").value
        )
        self.clear_octomap = node.get_parameter("clear_octomap").value
        self.octomap_settle_sec = node.get_parameter("octomap_settle_sec").value
        self.service_timeout = node.get_parameter("service_timeout").value
        self.execute_timeout = node.get_parameter("execute_timeout").value
        self.ik_timeout = node.get_parameter("ik_timeout").value

        self.plan_client = node.create_client(
            GetMotionPlan, "/plan_kinematic_path",
            callback_group=callback_group
        )
        self.scene_client = node.create_client(
            GetPlanningScene, "/get_planning_scene",
            callback_group=callback_group
        )
        self.ik_client = node.create_client(
            GetPositionIK, "/compute_ik",
            callback_group=callback_group
        )
        self.clear_octomap_client = node.create_client(
            Empty, "/clear_octomap",
            callback_group=callback_group
        )
        self.execute_client = ActionClient(
            node, ExecuteTrajectory, "/execute_trajectory",
            callback_group=callback_group
        )

    # --------------------------------------------------
    # 同期呼び出し
    # --------------------------------------------------

    @staticmethod
    def wait_future(future, timeout):

        event = threading.Event()
        future.add_done_callback(lambda _: event.set())

        if not event.wait(timeout):
            return None

        return future.result()

    def call(self, client, request, timeout):

        if not client.wait_for_service(timeout_sec=self.service_timeout):
            self.node.get_logger().error(f"{client.srv_name} not available")
            return None

        return self.wait_future(client.call_async(request), timeout)

    # --------------------------------------------------
    # MoveIt
    # --------------------------------------------------

    def get_joint_positions(self):
        """MoveItが認識している現在の関節角 {name: position}"""

        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ROBOT_STATE

        response = self.call(self.scene_client, request, self.service_timeout)

        if response is None:
            return None

        js = response.scene.robot_state.joint_state
        return dict(zip(js.name, js.position))

    def compute_ik(self, frame_id, link, position, orientation, seed):
        """
        link を frame_id での position / orientation (x, y, z, w) に置く関節角
        {name: position} を、seed {name: position} を初期値にして解く。
        衝突は見ない (Lemonや枝のボクセルに入る姿勢も解く)。解けなければ None
        """

        request = GetPositionIK.Request()

        ik = request.ik_request
        ik.group_name = self.planning_group
        ik.ik_link_name = link
        ik.avoid_collisions = False
        ik.timeout = Duration(
            sec=int(self.ik_timeout), nanosec=int(self.ik_timeout % 1.0 * 1e9)
        )
        ik.robot_state.joint_state.name = list(seed.keys())
        ik.robot_state.joint_state.position = [float(v) for v in seed.values()]

        pose = ik.pose_stamped
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = (float(v) for v in orientation)

        response = self.call(self.ik_client, request, self.service_timeout)

        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        js = response.solution.joint_state
        return dict(zip(js.name, js.position))

    def plan_and_execute(self, constraints, label):
        """
        constraints (moveit_msgs/Constraints) へ max_plan_attempts 回まで
        計画を試し、成功したら実行する。

        @return (success, message)
        """

        for attempt in range(1, self.max_plan_attempts + 1):

            if self.clear_octomap:
                self.reset_octomap()

            trajectory, code = self.plan(constraints)

            if trajectory is not None:
                self.node.get_logger().info(
                    f"[{label}] Plan succeeded (attempt {attempt})"
                )
                break

            self.node.get_logger().warn(
                f"[{label}] Plan attempt {attempt}/{self.max_plan_attempts} "
                f"failed (error_code={code})"
            )

        else:
            return (
                False,
                f"unreachable after {self.max_plan_attempts} attempts"
            )

        if not self.execute(trajectory):
            return False, "execution failed"

        return True, "reached"

    def reset_octomap(self):
        """
        移動中に取り込まれた腕自身の点などの古いボクセルで開始姿勢が
        衝突扱いになるので、octomapを消して現在の視野で作り直させる
        """

        if self.call(
            self.clear_octomap_client, Empty.Request(), self.service_timeout
        ) is None:
            self.node.get_logger().warn("Failed to clear octomap")
            return

        threading.Event().wait(self.octomap_settle_sec)

    def plan(self, constraints):

        request = GetMotionPlan.Request()

        req = request.motion_plan_request
        req.group_name = self.planning_group
        req.start_state.is_diff = True
        req.goal_constraints.append(constraints)
        req.num_planning_attempts = self.moveit_num_planning_attempts
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.acceleration_scaling

        response = self.call(
            self.plan_client, request, self.planning_time + 5.0
        )

        if response is None:
            return None, "timeout"

        res = response.motion_plan_response

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, res.error_code.val

        return res.trajectory, res.error_code.val

    def execute(self, trajectory):

        if not self.execute_client.wait_for_server(
            timeout_sec=self.service_timeout
        ):
            self.node.get_logger().error("/execute_trajectory not available")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        handle = self.wait_future(
            self.execute_client.send_goal_async(goal), self.service_timeout
        )

        if handle is None or not handle.accepted:
            self.node.get_logger().error("Execution goal rejected")
            return False

        result = self.wait_future(
            handle.get_result_async(), self.execute_timeout
        )

        if result is None:
            self.node.get_logger().error("Execution timed out")
            handle.cancel_goal_async()
            return False

        code = result.result.error_code.val

        if code != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().error(
                f"Execution failed (error_code={code})"
            )
            return False

        return True
