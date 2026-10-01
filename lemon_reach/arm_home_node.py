#!/usr/bin/env python3

import rclpy
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from moveit_msgs.msg import Constraints, JointConstraint
from std_srvs.srv import Trigger

from lemon_reach.moveit_helper import MoveItHelper


class ArmHome(Node):
    """
    起動時のアームの関節角をhome姿勢として記録し、そこへ戻す。

    ~/go_home  (Trigger): home姿勢へ計画・実行する
    ~/set_home (Trigger): 現在の姿勢をhome姿勢として記録し直す
    """

    def __init__(self):
        super().__init__("arm_home")

        self.cb_group = ReentrantCallbackGroup()
        self.moveit = MoveItHelper(self, self.cb_group)

        self.declare_parameter(
            "joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        )
        self.declare_parameter("joint_tolerance", 0.01)

        self.joint_names = self.get_parameter("joint_names").value
        self.joint_tolerance = self.get_parameter("joint_tolerance").value

        self.home = None
        self.busy = False

        self.create_service(
            Trigger, "~/go_home", self.go_home_callback,
            callback_group=self.cb_group
        )
        self.create_service(
            Trigger, "~/set_home", self.set_home_callback,
            callback_group=self.cb_group
        )

        # move_groupが立ち上がるまで記録を再試行する (前回の試行と重ならないように)
        self.record_timer = self.create_timer(
            1.0, self.record_on_startup,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

    def record_on_startup(self):

        if self.home is not None:
            return

        if self.record_home():
            self.record_timer.cancel()

    def record_home(self):

        joints = self.moveit.get_joint_positions()

        if joints is None:
            return False

        missing = [n for n in self.joint_names if n not in joints]

        if missing:
            self.get_logger().error(f"Joints not found: {missing}")
            return False

        self.home = {n: joints[n] for n in self.joint_names}

        self.get_logger().info(
            "Home recorded: "
            + ", ".join(f"{n}={v:.3f}" for n, v in self.home.items())
        )

        return True

    def set_home_callback(self, request, response):

        response.success = self.record_home()
        response.message = "recorded" if response.success else "failed"

        return response

    def go_home_callback(self, request, response):

        if self.home is None:
            response.success = False
            response.message = "home is not recorded"
            return response

        if self.busy:
            response.success = False
            response.message = "busy"
            return response

        self.busy = True

        try:
            constraints = Constraints()

            for name, position in self.home.items():
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = position
                jc.tolerance_above = self.joint_tolerance
                jc.tolerance_below = self.joint_tolerance
                jc.weight = 1.0
                constraints.joint_constraints.append(jc)

            success, message = self.moveit.plan_and_execute(
                constraints, "home"
            )

        finally:
            self.busy = False

        response.success = success
        response.message = message

        if success:
            self.get_logger().info("Reached home")
        else:
            self.get_logger().error(f"Go home failed: {message}")

        return response


def main():

    rclpy.init()

    node = ArmHome()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
