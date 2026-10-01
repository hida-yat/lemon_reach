#!/usr/bin/env python3

import math

import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import Pose, PointStamped, TransformStamped
from moveit_msgs.msg import (
    Constraints,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger

from lemon_reach.moveit_helper import MoveItHelper


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m):
    """回転行列 → (x, y, z, w)"""

    t = np.trace(m)

    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2
        return (
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        )

    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return (
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        )

    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return (
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        )

    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return (
        (m[0, 2] + m[2, 0]) / s,
        (m[1, 2] + m[2, 1]) / s,
        0.25 * s,
        (m[1, 0] - m[0, 1]) / s,
    )


class LemonApproach(Node):
    """
    lemon_target_node が固定したLemonの手前 (pregrasp) へ手先を移動する。

    pregrasp は ee_link と Lemon を結んだ直線上の、Lemonから standoff [m]
    手前の点 (指先 = ee_link から approach_axis 方向に tcp_offset の点を置く)。
    TF (pregrasp_frame) は指先の目標姿勢、MoveItの目標は ee_link の姿勢。ee_link の approach_axis がLemonを向き、down_axis が
    できるだけ下 (target frameの-z) を向く姿勢とする。approach_axis まわりの
    回転は roll_tolerance の範囲で許容する (pi/2 未満なら down_axis は上を向かない)。

    ~/approach (Trigger) で計画・実行し、結果を返す。解除や次の対象への
    切り替えは上位 (lemon_harvest_manager_node など) に任せる。
    """

    def __init__(self):
        super().__init__("lemon_approach")

        self.cb_group = ReentrantCallbackGroup()
        self.moveit = MoveItHelper(self, self.cb_group)

        self.declare_parameter("ee_link", "gripper_base")
        # Lemonへ向ける軸と、下へ向ける軸 ("x" / "y" / "z")
        self.declare_parameter("approach_axis", "z")
        self.declare_parameter("down_axis", "x")
        # Lemon中心から指先までの距離 [m]
        self.declare_parameter("standoff", 0.10)
        # ee_link から指先までの approach_axis 方向の距離 [m]
        # (piper: joint7/8 が gripper_base の z=0.1358、指先もほぼそこ)
        self.declare_parameter("tcp_offset", 0.136)
        self.declare_parameter("position_tolerance", 0.01)
        # approach_axis (Lemonへの向き) のずれの許容 [rad]
        self.declare_parameter("axis_tolerance", 0.1)
        # approach_axis まわりの回転の許容 [rad]
        # pi/2 未満にすると down_axis が上を向く姿勢は許さない
        self.declare_parameter("roll_tolerance", 1.4)
        self.declare_parameter("pregrasp_frame", "lemon_pregrasp")
        self.declare_parameter("tf_rate", 20.0)

        self.ee_link = self.get_parameter("ee_link").value
        self.standoff = self.get_parameter("standoff").value
        self.tcp_offset = self.get_parameter("tcp_offset").value
        self.position_tolerance = self.get_parameter("position_tolerance").value
        self.axis_tolerance = self.get_parameter("axis_tolerance").value
        self.roll_tolerance = self.get_parameter("roll_tolerance").value

        axes = {"x": 0, "y": 1, "z": 2}
        self.approach_axis = axes[self.get_parameter("approach_axis").value]
        self.down_axis = axes[self.get_parameter("down_axis").value]

        if self.approach_axis == self.down_axis:
            raise ValueError("approach_axis and down_axis must differ")
        self.pregrasp_frame = self.get_parameter("pregrasp_frame").value

        self.target = None      # PointStamped
        self.pregrasp = None    # TransformStamped
        self.busy = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.status_pub = self.create_publisher(String, "~/status", 10)

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(
            PointStamped,
            "/lemon_target/target",
            self.target_callback,
            latched,
            callback_group=self.cb_group
        )

        self.create_service(
            Trigger, "~/approach", self.approach_callback,
            callback_group=self.cb_group
        )

        self.create_timer(
            1.0 / self.get_parameter("tf_rate").value,
            self.broadcast_tf,
            callback_group=self.cb_group
        )

    def target_callback(self, msg):

        # frame_idが空なら lemon_target 側で対象が解除されている
        if not msg.header.frame_id:
            self.target = None
            self.pregrasp = None
            return

        self.target = msg

    # --------------------------------------------------
    # pregrasp の計算
    # --------------------------------------------------

    def compute_pregrasp(self):

        frame_id = self.target.header.frame_id

        try:
            ee = self.tf_buffer.lookup_transform(
                frame_id, self.ee_link, Time()
            ).transform

        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

        ee_pos = np.array([
            ee.translation.x, ee.translation.y, ee.translation.z
        ])
        lemon = np.array([
            self.target.point.x, self.target.point.y, self.target.point.z
        ])

        direction = lemon - ee_pos
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            self.get_logger().error("ee_link is at the lemon position")
            return None

        # approach_axis: ee_link → Lemon
        approach = direction / dist

        # down_axis: 下向き (-z) を approach に直交化して、できるだけ下を向ける
        down = np.array([0.0, 0.0, -1.0])
        down = down - np.dot(down, approach) * approach

        if np.linalg.norm(down) < 1e-3:
            # Lemonが真下にある場合は現在の down_axis の向きを使う
            r = ee.rotation
            down = quat_to_matrix(r.x, r.y, r.z, r.w)[:, self.down_axis]
            down = down - np.dot(down, approach) * approach

        down /= np.linalg.norm(down)

        # 残りの軸は右手系になるように決める
        other_axis = 3 - self.approach_axis - self.down_axis
        if (self.down_axis - self.approach_axis) % 3 == 1:
            other = np.cross(approach, down)
        else:
            other = np.cross(down, approach)

        m = np.zeros((3, 3))
        m[:, self.approach_axis] = approach
        m[:, self.down_axis] = down
        m[:, other_axis] = other

        q = matrix_to_quat(m)
        p = lemon - self.standoff * approach

        t = TransformStamped()
        t.header.frame_id = frame_id
        t.child_frame_id = self.pregrasp_frame
        t.transform.translation.x = float(p[0])
        t.transform.translation.y = float(p[1])
        t.transform.translation.z = float(p[2])
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = (float(v) for v in q)

        return t

    def make_constraints(self, pregrasp):

        frame_id = pregrasp.header.frame_id

        # 指先の目標から tcp_offset だけ戻した位置が ee_link の目標
        r = pregrasp.transform.rotation
        approach = quat_to_matrix(r.x, r.y, r.z, r.w)[:, self.approach_axis]
        t = pregrasp.transform.translation
        ee_goal = np.array([t.x, t.y, t.z]) - self.tcp_offset * approach

        goal = Pose()
        goal.position.x = float(ee_goal[0])
        goal.position.y = float(ee_goal[1])
        goal.position.z = float(ee_goal[2])
        goal.orientation.w = 1.0

        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [self.position_tolerance]

        position = PositionConstraint()
        position.header.frame_id = frame_id
        position.link_name = self.ee_link
        position.constraint_region.primitives.append(region)
        position.constraint_region.primitive_poses.append(goal)
        position.weight = 1.0

        orientation = OrientationConstraint()
        orientation.header.frame_id = frame_id
        orientation.link_name = self.ee_link
        orientation.orientation = pregrasp.transform.rotation
        # approach_axis まわりだけ roll_tolerance、他は axis_tolerance
        tol = [self.axis_tolerance] * 3
        tol[self.approach_axis] = self.roll_tolerance
        (
            orientation.absolute_x_axis_tolerance,
            orientation.absolute_y_axis_tolerance,
            orientation.absolute_z_axis_tolerance,
        ) = tol
        orientation.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(position)
        constraints.orientation_constraints.append(orientation)

        return constraints

    # --------------------------------------------------
    # Service
    # --------------------------------------------------

    def approach_callback(self, request, response):

        if self.busy:
            response.success = False
            response.message = "busy"
            return response

        if self.target is None:
            response.success = False
            response.message = "no target"
            return response

        self.busy = True

        try:
            pregrasp = self.compute_pregrasp()

            if pregrasp is None:
                response.success = False
                response.message = "failed to compute pregrasp"
                return response

            self.pregrasp = pregrasp

            p = pregrasp.transform.translation
            self.get_logger().info(
                f"Approach {pregrasp.header.frame_id}: "
                f"pregrasp=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )
            self.publish_status("approaching")

            success, message = self.moveit.plan_and_execute(
                self.make_constraints(pregrasp), "approach"
            )

            response.success = success
            response.message = message

        finally:
            self.busy = False

        self.publish_status(response.message)

        if response.success:
            self.get_logger().info("Reached pregrasp")
        else:
            self.get_logger().error(f"Approach failed: {response.message}")

        return response

    # --------------------------------------------------

    def broadcast_tf(self):

        if self.pregrasp is None:
            return

        self.pregrasp.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.pregrasp)

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))


def main():

    rclpy.init()

    node = LemonApproach()
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
