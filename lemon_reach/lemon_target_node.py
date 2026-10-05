#!/usr/bin/env python3

from collections import deque

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.time import Time

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import PointStamped, TransformStamped, Vector3Stamped
from lemon_msgs.msg import LemonEllipseArray
from std_srvs.srv import Trigger


class LemonTarget(Node):
    """
    視野内で安定して認識され続けたLemonのうち、base_frameから最も近いものを
    把持対象として選択し、解除されるまでTFを固定して配信する。

    検出は画像の撮影時刻のTFで base_frame へ変換する。最新のTFで変換すると、
    手首カメラが動いている間はYOLOの処理遅延の分だけカメラの移動がLemonの
    位置のずれになり、追従するアームとの間で揺れが大きくなっていく。
    撮影時刻のTFがまだ届いていない検出は、届くまで (最大 tf_timeout 秒) 保留する。

    入力は lemon_ellipse_node の /lemon_ellipse/lemons (マスク付きの検出ごとの
    3D中心と、楕円の短辺の3Dの向き。カメラの光学座標系)。3D中心は detect_3d_node が
    マスク内の深度から求めたもの。

    固定中も固定位置の近傍の検出で位置を更新し続け、~/tracked として配信する
    (~/target は固定位置のまま。approach は固定位置へ、grasp は tracked へ動く)。

    楕円の短辺 (グリッパで挟む向き) を base_frame の3Dの向きに直し、
    LOCK までのサンプルを平均して ~/short_axis に配信する (LOCK 中も更新)。
    向きは 180 度で同じなので、単位ベクトル u の u u^T を平均して最大固有ベクトルを取る。
    向きが定まらない (ほぼ円、サンプル不足、ばらつきが大きい) ときはゼロベクトル。

    状態:
        COLLECTING: window_sec 秒間の検出を集めて安定したLemonを探す
        LOCKED:     選択した位置にTFを置き続ける (~/release で解除)
    """

    def __init__(self):
        super().__init__("lemon_target")

        # 対象クラス (大文字小文字は区別しない)
        self.declare_parameter("class_name", "lemon")
        self.declare_parameter("lemons_topic", "/lemon_ellipse/lemons")
        self.declare_parameter("base_frame", "base_link")
        # 撮影時刻のTFが届くまで検出を保留する最大時間 [s]
        self.declare_parameter("tf_timeout", 0.5)
        # セグメンテーションモデルは sim のLemonに 0.5〜0.6 程度のスコアを出す
        # (誤検出は安定判定で除く)
        self.declare_parameter("score_threshold", 0.5)
        # 安定判定: window_sec 秒間のフレームのうち min_detection_ratio 以上で
        # 同じ位置 (cluster_radius 以内) に認識されていること
        self.declare_parameter("window_sec", 3.0)
        self.declare_parameter("min_detection_ratio", 0.95)
        self.declare_parameter("min_frames", 10)
        self.declare_parameter("cluster_radius", 0.02)
        # 把持済み / 到達不可として解除した位置の近傍は再選択しない
        self.declare_parameter("exclusion_radius", 0.03)
        # base_frameからの距離がこれより遠いものは候補にしない (0以下で無効)
        self.declare_parameter("max_distance", 0.0)
        # 固定中の追従: 現在の推定から track_radius 以内の検出を track_alpha で
        # 平滑化して取り込む。固定位置から max_track_offset より離れたら取り込まない
        self.declare_parameter("track_radius", 0.04)
        self.declare_parameter("track_alpha", 0.3)
        self.declare_parameter("max_track_offset", 0.10)
        # カメラからこれより近い [m] 検出は使わない。近づくとLemonがグリッパの影や
        # 画像の外にかかって検出が不安定になり、追従先が隣のLemonへずれていく
        # (実機の D435 も 0.1〜0.2 m より近い深度は取れない)
        self.declare_parameter("min_depth", 0.12)
        # 向きを決めるのに必要なサンプル数と、そろい具合 (u u^T の平均の最大固有値、
        # 全部同じ向きなら 1、0.85 で角度のばらつき約 ±20 度) の下限
        self.declare_parameter("min_axis_samples", 5)
        self.declare_parameter("min_axis_consistency", 0.85)
        self.declare_parameter("lemon_frame", "lemon_target")
        self.declare_parameter("tracked_frame", "lemon_tracked")
        self.declare_parameter("tf_rate", 20.0)

        self.class_name = self.get_parameter("class_name").value.lower()
        self.base_frame = self.get_parameter("base_frame").value
        self.tf_timeout = self.get_parameter("tf_timeout").value
        self.score_threshold = self.get_parameter("score_threshold").value
        self.window_sec = self.get_parameter("window_sec").value
        self.min_detection_ratio = self.get_parameter("min_detection_ratio").value
        self.min_frames = self.get_parameter("min_frames").value
        self.cluster_radius = self.get_parameter("cluster_radius").value
        self.exclusion_radius = self.get_parameter("exclusion_radius").value
        self.max_distance = self.get_parameter("max_distance").value
        self.track_radius = self.get_parameter("track_radius").value
        self.track_alpha = self.get_parameter("track_alpha").value
        self.max_track_offset = self.get_parameter("max_track_offset").value
        self.min_depth = self.get_parameter("min_depth").value
        self.min_axis_samples = self.get_parameter("min_axis_samples").value
        self.min_axis_consistency = self.get_parameter("min_axis_consistency").value
        self.lemon_frame = self.get_parameter("lemon_frame").value
        self.tracked_frame = self.get_parameter("tracked_frame").value

        # 収集中の状態
        self.frames = []        # 受信したフレームの時刻
        self.candidates = []    # [{"samples": [(t, xyz, 短辺 or None)]}]
        self.start_time = None

        # 固定中の状態
        self.locked = None      # PointStamped
        self.tracked = None     # np.array xyz (base_frame)
        self.axis_samples = []  # 短辺の u u^T (向きが定まるまで)
        self.axis_matrix = None  # 短辺の u u^T の平均 (3x3)
        self.exclusions = []    # [np.array xyz]

        # 撮影時刻のTFを待っている検出 (古い順)
        self.pending = deque()

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.target_pub = self.create_publisher(
            PointStamped, "~/target", latched
        )
        self.tracked_pub = self.create_publisher(
            PointStamped, "~/tracked", latched
        )
        self.short_axis_pub = self.create_publisher(
            Vector3Stamped, "~/short_axis", latched
        )

        self.create_service(Trigger, "~/release", self.release_callback)
        self.create_service(
            Trigger, "~/clear_exclusions", self.clear_exclusions_callback
        )

        self.create_subscription(
            LemonEllipseArray,
            self.get_parameter("lemons_topic").value,
            self.detection_callback,
            10
        )

        self.create_timer(
            1.0 / self.get_parameter("tf_rate").value,
            self.broadcast_tf
        )
        self.create_timer(0.05, self.process_pending)

    # --------------------------------------------------
    # 撮影時刻のTFを待つ
    # --------------------------------------------------

    def detection_callback(self, msg):

        self.pending.append(msg)
        self.process_pending()

    def process_pending(self):

        now = self.get_clock().now()

        while self.pending:

            msg = self.pending[0]
            stamp = Time.from_msg(msg.header.stamp)

            if self.transforms_ready(msg, stamp):
                self.pending.popleft()
                self.handle_detections(msg)

            elif (now - stamp).nanoseconds * 1e-9 > self.tf_timeout:
                self.pending.popleft()
                self.get_logger().warn(
                    "Dropped detections: TF at the image time is not available",
                    throttle_duration_sec=2.0
                )

            else:
                # TFは時刻順に届くので、後の検出もまだ変換できない
                break

    def transforms_ready(self, msg, stamp):

        if not any(self.is_target(lemon) for lemon in msg.lemons):
            return True

        return self.tf_buffer.can_transform(
            self.base_frame, msg.header.frame_id, stamp
        )

    def is_target(self, lemon):
        return lemon.class_name.lower() == self.class_name

    # --------------------------------------------------
    # 収集・安定判定
    # --------------------------------------------------

    def handle_detections(self, msg):

        if self.locked is not None:
            self.track(msg)
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = t

        self.frames.append(t)
        self.frames = [f for f in self.frames if f > t - self.window_sec]

        # 1フレームで同じ候補を2回数えない
        updated = set()

        for p, axis in self.detection_points(msg):

            if self.max_distance > 0.0 and np.linalg.norm(p) > self.max_distance:
                continue

            if any(
                np.linalg.norm(p - e) < self.exclusion_radius
                for e in self.exclusions
            ):
                continue

            i = self.find_candidate(p)

            if i is None:
                self.candidates.append({"samples": [(t, p, axis)]})
                updated.add(len(self.candidates) - 1)

            elif i not in updated:
                self.candidates[i]["samples"].append((t, p, axis))
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

        # base_frameから一番近いものを選択
        best = min(stable, key=lambda c: np.linalg.norm(self.center(c)))

        self.lock(best, t)

    # --------------------------------------------------
    # 固定中の追従
    # --------------------------------------------------

    def track(self, msg):

        locked = np.array([
            self.locked.point.x, self.locked.point.y, self.locked.point.z
        ])

        best = None
        best_axis = None
        best_dist = self.track_radius

        for p, axis in self.detection_points(msg):

            if np.linalg.norm(p - locked) > self.max_track_offset:
                continue

            dist = np.linalg.norm(p - self.tracked)
            if dist <= best_dist:
                best = p
                best_axis = axis
                best_dist = dist

        if best is None:
            return

        self.tracked = self.tracked + self.track_alpha * (best - self.tracked)
        self.publish_tracked(msg.header.stamp)

        if best_axis is not None:
            self.add_axis_sample(best_axis)
            self.publish_short_axis(msg.header.stamp)

    # --------------------------------------------------
    # 検出の変換
    # --------------------------------------------------

    def detection_points(self, msg):
        """
        対象クラス・score・奥行きで絞った検出の中心と短辺の向きを、撮影時刻のTFで
        base_frame へ変換して [(xyz, 短辺の単位ベクトル or None)] で返す。
        短辺の向きが使えない検出 (short_axis_valid = false) は None
        """

        # center はカメラの光学座標系なので z が奥行き
        lemons = [
            lemon for lemon in msg.lemons
            if self.is_target(lemon)
            and lemon.score >= self.score_threshold
            and lemon.center.z >= self.min_depth
        ]

        if not lemons:
            return []

        m = self.lookup(msg.header.frame_id, Time.from_msg(msg.header.stamp))

        if m is None:
            return []

        rotation, translation = m[:3, :3], m[:3, 3]
        points = []

        for lemon in lemons:

            c = lemon.center
            p = rotation @ np.array([c.x, c.y, c.z]) + translation

            if not np.all(np.isfinite(p)):
                continue

            axis = None

            if lemon.short_axis_valid:
                a = lemon.short_axis
                axis = rotation @ np.array([a.x, a.y, a.z])
                axis /= np.linalg.norm(axis)

            points.append((p, axis))

        return points

    # --------------------------------------------------
    # 短辺の向きの平均
    # --------------------------------------------------

    def add_axis_sample(self, axis):

        outer = np.outer(axis, axis)

        if self.axis_matrix is None:
            # 向きが定まるまではサンプルを貯めて平均する
            self.axis_samples.append(outer)

            if len(self.axis_samples) >= self.min_axis_samples:
                self.axis_matrix = np.mean(self.axis_samples, axis=0)
        else:
            self.axis_matrix += self.track_alpha * (outer - self.axis_matrix)

    def current_short_axis(self):
        """平均した短辺の向き。定まらなければ None"""

        if self.axis_matrix is None:
            return None

        values, vectors = np.linalg.eigh(self.axis_matrix)

        if values[-1] < self.min_axis_consistency:
            return None

        return vectors[:, -1]

    def publish_short_axis(self, stamp):

        axis = self.current_short_axis()

        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame

        if axis is not None:
            msg.vector.x, msg.vector.y, msg.vector.z = (float(v) for v in axis)

        self.short_axis_pub.publish(msg)

    def lookup(self, frame_id, stamp):
        """frame_id → base_frame の撮影時刻の同次変換行列"""

        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, frame_id, stamp
            ).transform

        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup failed: {e}", throttle_duration_sec=2.0
            )
            return None

        x, y, z, w = (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)

        m = np.eye(4)
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]

        return m

    # --------------------------------------------------

    def publish_tracked(self, stamp):

        tracked = PointStamped()
        tracked.header.stamp = stamp
        tracked.header.frame_id = self.base_frame
        tracked.point.x = float(self.tracked[0])
        tracked.point.y = float(self.tracked[1])
        tracked.point.z = float(self.tracked[2])

        self.tracked_pub.publish(tracked)

    # --------------------------------------------------

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
        self.locked.header.frame_id = self.base_frame
        self.locked.point.x = float(p[0])
        self.locked.point.y = float(p[1])
        self.locked.point.z = float(p[2])

        self.target_pub.publish(self.locked)

        self.tracked = p.copy()
        self.publish_tracked(self.locked.header.stamp)

        self.axis_samples = []
        self.axis_matrix = None

        for s in cand["samples"]:
            if s[2] is not None:
                self.add_axis_sample(s[2])

        self.publish_short_axis(self.locked.header.stamp)
        axis = self.current_short_axis()

        self.get_logger().info(
            f"Locked lemon {self.base_frame}: "
            f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
            f"dist={np.linalg.norm(p):.3f} ratio={ratio:.2f} "
            f"candidates={len(self.candidates)} "
            f"short_axis={np.round(axis, 2) if axis is not None else None}"
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

        stamp = self.get_clock().now().to_msg()
        p = self.locked.point

        self.tf_broadcaster.sendTransform([
            self.make_tf(stamp, self.lemon_frame, (p.x, p.y, p.z)),
            self.make_tf(stamp, self.tracked_frame, self.tracked),
        ])

    def make_tf(self, stamp, child_frame_id, xyz):

        t = TransformStamped()

        t.header.stamp = stamp
        t.header.frame_id = self.locked.header.frame_id
        t.child_frame_id = child_frame_id

        t.transform.translation.x = float(xyz[0])
        t.transform.translation.y = float(xyz[1])
        t.transform.translation.z = float(xyz[2])

        # 姿勢は不明なので単位クォータニオン
        t.transform.rotation.w = 1.0

        return t

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
        self.tracked = None
        self.axis_samples = []
        self.axis_matrix = None
        self.reset_collection()

        # frame_idが空のメッセージで「対象なし」を通知する
        # (latchedなので、後から起動したノードに解除済みの対象が届かないように)
        cleared = PointStamped()
        cleared.header.stamp = self.get_clock().now().to_msg()
        self.target_pub.publish(cleared)
        self.tracked_pub.publish(cleared)
        cleared_axis = Vector3Stamped()
        cleared_axis.header.stamp = cleared.header.stamp
        self.short_axis_pub.publish(cleared_axis)

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
