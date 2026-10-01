#!/usr/bin/env python3

import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger


class LemonHarvestManager(Node):
    """
    各ノードのServiceを順に呼ぶだけの簡易シーケンサ (将来Behavior Treeに置換)。

        対象が固定されるのを待つ
        → /lemon_approach/approach  (Lemon手前へ移動。現時点ではここで把持完了とみなす)
        → /arm_home/go_home         (起動時の姿勢へ戻る)
        → /lemon_target/release     (対象を解除。home姿勢の視野で次の対象を選び直す)
        → 繰り返し
    """

    def __init__(self):
        super().__init__("lemon_harvest_manager")

        self.declare_parameter("call_timeout", 120.0)
        self.call_timeout = self.get_parameter("call_timeout").value

        self.cb_group = ReentrantCallbackGroup()

        self.target = None
        self.target_event = threading.Event()

        self.approach_client = self.create_client(
            Trigger, "/lemon_approach/approach", callback_group=self.cb_group
        )
        self.home_client = self.create_client(
            Trigger, "/arm_home/go_home", callback_group=self.cb_group
        )
        self.release_client = self.create_client(
            Trigger, "/lemon_target/release", callback_group=self.cb_group
        )

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

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def target_callback(self, msg):

        # frame_idが空なら対象なし
        if not msg.header.frame_id:
            self.target = None
            self.target_event.clear()
            return

        self.target = msg
        self.target_event.set()

    # --------------------------------------------------

    def run(self):

        while rclpy.ok():

            self.publish_status("waiting_target")

            if not self.target_event.wait(timeout=1.0):
                continue

            p = self.target.point
            self.get_logger().info(
                f"Target ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )

            self.publish_status("approaching")
            ok, message = self.call(self.approach_client)

            if ok:
                # 現時点ではLemon手前への移動で把持完了とみなす
                self.get_logger().info("Grasp done (approach reached)")
            else:
                self.get_logger().warn(f"Approach failed: {message}")

            self.publish_status("going_home")
            ok, message = self.call(self.home_client)

            if not ok:
                # homeに戻れないと次の対象を正しく選べないので停止する
                self.get_logger().error(
                    f"Go home failed: {message}. Stopping sequence"
                )
                self.publish_status("error")
                return

            self.publish_status("releasing")
            self.target_event.clear()
            ok, message = self.call(self.release_client)

            if not ok:
                self.get_logger().warn(f"Release failed: {message}")

    def call(self, client):

        if not client.wait_for_service(timeout_sec=5.0):
            return False, f"{client.srv_name} not available"

        future = client.call_async(Trigger.Request())

        event = threading.Event()
        future.add_done_callback(lambda _: event.set())

        if not event.wait(self.call_timeout):
            return False, "timeout"

        result = future.result()
        return result.success, result.message

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))


def main():

    rclpy.init()

    node = LemonHarvestManager()
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
