#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import tf2_ros
from geometry_msgs.msg import PointStamped, TransformStamped
from std_srvs.srv import Trigger
from yolo_msgs.msg import DetectionArray


class LemonTarget(Node):
    """
    視野内で安定して認識され続けたLemonのうち、base_linkから最も近いものを
    把持対象として選択し、解除されるまでTFを固定して配信する。

    状態:
        COLLECTING: window_sec 秒間の検出を集めて安定したLemonを探す
        LOCKED:     選択した位置にTFを置き続ける (~/release で解除)
    """

    def __init__(self):
        super().__init__("lemon_target")

        self.declare_parameter("class_name", "Lemon")
        self.declare_parameter("score_threshold", 0.8)
        # 安定判定: window_sec 秒間のフレームのうち min_detection_ratio 以上で
        # 同じ位置 (cluster_radius 以内) に認識されていること
        self.declare_parameter("window_sec", 3.0)
        self.declare_parameter("min_detection_ratio", 0.95)
        self.declare_parameter("min_frames", 10)
        self.declare_parameter("cluster_radius", 0.02)
        # 把持済み / 到達不可として解除した位置の近傍は再選択しない
        self.declare_parameter("exclusion_radius", 0.03)
        # base_linkからの距離がこれより遠いものは候補にしない (0以下で無効)
        self.declare_parameter("max_distance", 0.0)
        self.declare_parameter("lemon_frame", "lemon_target")
        self.declare_parameter("tf_rate", 20.0)

        self.class_name = self.get_parameter("class_name").value
        self.score_threshold = self.get_parameter("score_threshold").value
        self.window_sec = self.get_parameter("window_sec").value
        self.min_detection_ratio = self.get_parameter("min_detection_ratio").value
        self.min_frames = self.get_parameter("min_frames").value
        self.cluster_radius = self.get_parameter("cluster_radius").value
        self.exclusion_radius = self.get_parameter("exclusion_radius").value
        self.max_distance = self.get_parameter("max_distance").value
        self.lemon_frame = self.get_parameter("lemon_frame").value

        # 収集中の状態
        self.frames = []        # 受信したフレームの時刻
        self.candidates = []    # [{"samples": [(t, np.array xyz)]}]
        self.start_time = None
        self.frame_id = None

        # 固定中の状態
        self.locked = None      # PointStamped
        self.exclusions = []    # [np.array xyz]

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.target_pub = self.create_publisher(
            PointStamped, "~/target", latched
        )

        self.create_service(Trigger, "~/release", self.release_callback)
        self.create_service(
            Trigger, "~/clear_exclusions", self.clear_exclusions_callback
        )

        self.create_subscription(
            DetectionArray,
            "/yolo/detections_3d",
            self.detection_callback,
            10
        )

        self.create_timer(
            1.0 / self.get_parameter("tf_rate").value,
            self.broadcast_tf
        )

    # --------------------------------------------------
    # 収集・安定判定
    # --------------------------------------------------

    def detection_callback(self, msg):

        if self.locked is not None:
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = t

        self.frames.append(t)
        self.frames = [f for f in self.frames if f > t - self.window_sec]

        # 1フレームで同じ候補を2回数えない
        updated = set()

        for d in msg.detections:

            if d.class_name != self.class_name:
                continue

            if d.score < self.score_threshold:
                continue

            c = d.bbox3d.center.position
            p = np.array([c.x, c.y, c.z])

            if not np.all(np.isfinite(p)):
                continue

            if self.max_distance > 0.0 and np.linalg.norm(p) > self.max_distance:
                continue

            if any(
                np.linalg.norm(p - e) < self.exclusion_radius
                for e in self.exclusions
            ):
                continue

            self.frame_id = d.bbox3d.frame_id

            i = self.find_candidate(p)

            if i is None:
                self.candidates.append({"samples": [(t, p)]})
                updated.add(len(self.candidates) - 1)

            elif i not in updated:
                self.candidates[i]["samples"].append((t, p))
                updated.add(i)

        # window外のサンプルを捨てる
        for cand in self.candidates:
            cand["samples"] = [
                s for s in cand["samples"] if s[0] > t - self.window_sec
            ]
        self.candidates = [c for c in self.candidates if c["samples"]]

        # window_sec 秒以上集めてから判定
        if t - self.start_time < self.window_sec:
            return

        if len(self.frames) < self.min_frames:
            return

        stable = [
            c for c in self.candidates
            if len(c["samples"]) / len(self.frames) >= self.min_detection_ratio
        ]

        if not stable:
            return

        # base_linkから一番近いものを選択
        best = min(stable, key=lambda c: np.linalg.norm(self.center(c)))

        self.lock(best, t)

    def find_candidate(self, p):

        best_i = None
        best_dist = self.cluster_radius

        for i, cand in enumerate(self.candidates):
            dist = np.linalg.norm(p - self.center(cand))
            if dist <= best_dist:
                best_i = i
                best_dist = dist

        return best_i

    @staticmethod
    def center(cand):
        return np.median([s[1] for s in cand["samples"]], axis=0)

    def lock(self, cand, t):

        p = self.center(cand)
        ratio = len(cand["samples"]) / len(self.frames)

        self.locked = PointStamped()
        self.locked.header.stamp = self.get_clock().now().to_msg()
        self.locked.header.frame_id = self.frame_id
        self.locked.point.x = float(p[0])
        self.locked.point.y = float(p[1])
        self.locked.point.z = float(p[2])

        self.target_pub.publish(self.locked)

        self.get_logger().info(
            f"Locked lemon {self.frame_id}: "
            f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
            f"dist={np.linalg.norm(p):.3f} ratio={ratio:.2f} "
            f"candidates={len(self.candidates)}"
        )

    def reset_collection(self):
        self.frames = []
        self.candidates = []
        self.start_time = None

    # --------------------------------------------------
    # TF配信
    # --------------------------------------------------

    def broadcast_tf(self):

        if self.locked is None:
            return

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.locked.header.frame_id
        t.child_frame_id = self.lemon_frame

        t.transform.translation.x = self.locked.point.x
        t.transform.translation.y = self.locked.point.y
        t.transform.translation.z = self.locked.point.z

        # 姿勢は不明なので単位クォータニオン
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    # --------------------------------------------------
    # Services
    # --------------------------------------------------

    def release_callback(self, request, response):

        if self.locked is None:
            response.success = False
            response.message = "No target is locked"
            return response

        p = self.locked.point
        self.exclusions.append(np.array([p.x, p.y, p.z]))

        self.locked = None
        self.reset_collection()

        # frame_idが空のメッセージで「対象なし」を通知する
        # (latchedなので、後から起動したノードに解除済みの対象が届かないように)
        cleared = PointStamped()
        cleared.header.stamp = self.get_clock().now().to_msg()
        self.target_pub.publish(cleared)

        response.success = True
        response.message = (
            f"Released ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), "
            f"{len(self.exclusions)} excluded"
        )
        self.get_logger().info(response.message)

        return response

    def clear_exclusions_callback(self, request, response):

        n = len(self.exclusions)
        self.exclusions = []

        response.success = True
        response.message = f"Cleared {n} exclusions"
        self.get_logger().info(response.message)

        return response


def main():

    rclpy.init()

    node = LemonTarget()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
