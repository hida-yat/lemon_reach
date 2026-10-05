#!/usr/bin/env python3

import math
from collections import OrderedDict

import cv2
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from lemon_msgs.msg import LemonEllipse, LemonEllipseArray
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from yolo_msgs.msg import DetectionArray

from lemon_reach.lemon_ellipse import fit_ellipse


# BGR (画像)
ELLIPSE_COLOR = (0, 255, 255)
MAJOR_COLOR = (255, 160, 0)
MINOR_COLOR = (255, 0, 255)
UNRELIABLE_COLOR = (160, 160, 160)

# RGBA (マーカー)
MARKER_MAJOR = ColorRGBA(r=0.0, g=0.63, b=1.0, a=1.0)
MARKER_MINOR = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
MARKER_UNRELIABLE = ColorRGBA(r=0.63, g=0.63, b=0.63, a=1.0)
MARKER_CENTER = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)


class LemonEllipseNode(Node):
    """
    Lemonの検出 (/yolo/detections_3d) ごとにマスクへ楕円を当てはめ、
    短辺・長辺を求めて配信・表示する。

    配信:
        ~/lemons     LemonEllipseArray (lemon_target が使う)
                     3D中心、短辺・長辺の3Dの向き (カメラの光学座標系)、画像上の楕円
        ~/markers    MarkerArray (RViz で短辺・長辺を3Dで表示)
        ~/dbg_image  /yolo/dbg_image に楕円・長辺・短辺を重ねた画像

    短辺の3Dの向きは、同じ奥行きで画素を短辺方向にずらしたときの向き
    (fx, fy で画素を光学座標系の向きに直す)。視線に垂直な面の中の向きになる。

    debug 画像の表示:
        黄       当てはめた楕円
        水色     長辺
        マゼンタ 短辺 (グリッパで挟む向き)
        灰色     短辺の向きが使えない (ほぼ円、または欠けたマスク)
        文字     長辺/短辺, 短辺の画像横軸からの角度 (時計回り正), 塗りつぶし率
    """

    def __init__(self):
        super().__init__("lemon_ellipse")

        self.declare_parameter("detections_topic", "/yolo/detections_3d")
        self.declare_parameter("image_topic", "/yolo/dbg_image")
        self.declare_parameter("camera_info_topic", "/realsense/color/camera_info")
        # 対象クラス (大文字小文字は区別しない)
        self.declare_parameter("class_name", "lemon")
        # 長辺 / 短辺がこれ未満 (ほぼ円) なら短辺の向きは使えない
        self.declare_parameter("min_aspect", 1.15)
        # マスクの面積 / 楕円の面積がこれ未満 (欠けている) なら短辺の向きは使えない
        self.declare_parameter("min_fill", 0.8)
        self.declare_parameter("publish_debug_image", True)
        # debug 画像と楕円を撮影時刻で組み合わせるために持っておく数
        self.declare_parameter("cache_size", 30)

        self.class_name = self.get_parameter("class_name").value.lower()
        self.min_aspect = self.get_parameter("min_aspect").value
        self.min_fill = self.get_parameter("min_fill").value
        self.cache_size = self.get_parameter("cache_size").value

        self.focal = None   # (fx, fy)
        self.bridge = CvBridge()

        # 撮影時刻 → 楕円 / debug 画像 (どちらが先に届いても組み合わせる)
        self.ellipse_cache = OrderedDict()
        self.image_cache = OrderedDict()

        self.lemons_pub = self.create_publisher(LemonEllipseArray, "~/lemons", 10)
        self.markers_pub = self.create_publisher(MarkerArray, "~/markers", 10)
        self.image_pub = self.create_publisher(Image, "~/dbg_image", 10)

        self.create_subscription(
            DetectionArray,
            self.get_parameter("detections_topic").value,
            self.detections_callback,
            10
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        if self.get_parameter("publish_debug_image").value:
            self.create_subscription(
                Image,
                self.get_parameter("image_topic").value,
                self.image_callback,
                10
            )

    # --------------------------------------------------
    # 楕円
    # --------------------------------------------------

    def camera_info_callback(self, msg):

        if msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self.focal = (msg.k[0], msg.k[4])

    def detections_callback(self, msg):

        out = LemonEllipseArray()
        out.header = msg.header

        for d in msg.detections:

            if d.class_name.lower() != self.class_name or not d.mask.data:
                continue

            # 短辺・長辺は画像の座標系で求めるので、3D中心も同じ座標系であること
            if d.bbox3d.frame_id != msg.header.frame_id:
                self.get_logger().warn(
                    f"bbox3d frame {d.bbox3d.frame_id} != image frame "
                    f"{msg.header.frame_id} (set yolo target_frame to the camera frame)",
                    throttle_duration_sec=10.0
                )
                continue

            lemon = self.make_lemon(d)

            if lemon is not None:
                out.lemons.append(lemon)

        self.lemons_pub.publish(out)
        self.markers_pub.publish(self.make_markers(out))

        self.ellipse_cache[self.key(msg.header.stamp)] = out
        self.trim(self.ellipse_cache)
        self.match()

    def make_lemon(self, d):

        e = fit_ellipse([(p.x, p.y) for p in d.mask.data])

        if e is None:
            return None

        fx, fy = self.focal if self.focal else (1.0, 1.0)

        lemon = LemonEllipse()
        lemon.class_name = d.class_name
        lemon.score = d.score
        lemon.id = d.id
        lemon.center = d.bbox3d.center.position

        # 同じ奥行きで画素を短辺 / 長辺方向にずらしたときの3Dの向き
        for axis, direction in (
            (lemon.short_axis, e.minor_dir), (lemon.long_axis, e.major_dir)
        ):
            v = np.array([direction[0] / fx, direction[1] / fy, 0.0])
            v /= np.linalg.norm(v)
            axis.x, axis.y, axis.z = (float(c) for c in v)

        lemon.short_axis_valid = bool(
            e.aspect >= self.min_aspect and e.fill >= self.min_fill
        )

        lemon.center_u, lemon.center_v = (float(c) for c in e.center)
        lemon.major_radius = float(e.major)
        lemon.minor_radius = float(e.minor)
        lemon.minor_angle = float(e.minor_angle)
        lemon.aspect = float(e.aspect)
        lemon.fill = float(e.fill)

        return lemon

    # --------------------------------------------------
    # RViz マーカー
    # --------------------------------------------------

    def make_markers(self, msg):

        markers = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for i, lemon in enumerate(msg.lemons):

            c = np.array([lemon.center.x, lemon.center.y, lemon.center.z])

            # 画像上の半径 [px] を中心の奥行きで長さ [m] にする
            fx = self.focal[0] if self.focal else None
            scale = c[2] / fx if fx and c[2] > 0.0 else 0.0

            if scale <= 0.0:
                continue

            for ns, axis, radius, color in (
                ("long_axis", lemon.long_axis, lemon.major_radius, MARKER_MAJOR),
                (
                    "short_axis", lemon.short_axis, lemon.minor_radius,
                    MARKER_MINOR if lemon.short_axis_valid else MARKER_UNRELIABLE
                ),
            ):
                half = np.array([axis.x, axis.y, axis.z]) * radius * scale

                m = self.marker(msg.header, ns, i, Marker.LINE_LIST, color)
                m.scale.x = 0.004
                m.points = [self.point(c - half), self.point(c + half)]
                markers.markers.append(m)

            m = self.marker(msg.header, "center", i, Marker.SPHERE, MARKER_CENTER)
            m.pose.position = lemon.center
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.01
            markers.markers.append(m)

        return markers

    @staticmethod
    def marker(header, ns, i, marker_type, color):

        m = Marker()
        m.header = header
        m.ns = ns
        m.id = i
        m.type = marker_type
        m.action = Marker.ADD
        m.color = color
        m.lifetime = Duration(seconds=0.5).to_msg()

        return m

    @staticmethod
    def point(v):
        return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))

    # --------------------------------------------------
    # debug 画像
    # --------------------------------------------------

    def image_callback(self, msg):

        self.image_cache[self.key(msg.header.stamp)] = msg
        self.trim(self.image_cache)
        self.match()

    def match(self):
        """同じ撮影時刻の debug 画像と楕円がそろったら描いて配信する"""

        for key in [k for k in self.image_cache if k in self.ellipse_cache]:

            image_msg = self.image_cache.pop(key)
            ellipses = self.ellipse_cache.pop(key)

            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

            for lemon in ellipses.lemons:
                self.draw(image, lemon)

            self.image_pub.publish(
                self.bridge.cv2_to_imgmsg(
                    image, encoding="bgr8", header=image_msg.header
                )
            )

    def draw(self, image, lemon):

        center = np.array([lemon.center_u, lemon.center_v])
        minor_dir = np.array([math.cos(lemon.minor_angle), math.sin(lemon.minor_angle)])
        major_dir = np.array([-minor_dir[1], minor_dir[0]])
        valid = lemon.short_axis_valid

        cv2.ellipse(
            image, tuple(int(round(v)) for v in center),
            (int(round(lemon.major_radius)), int(round(lemon.minor_radius))),
            math.degrees(math.atan2(major_dir[1], major_dir[0])),
            0, 360, ELLIPSE_COLOR, 2
        )

        self.draw_axis(image, center, major_dir * lemon.major_radius, MAJOR_COLOR, 2)
        self.draw_axis(
            image, center, minor_dir * lemon.minor_radius,
            MINOR_COLOR if valid else UNRELIABLE_COLOR, 3
        )

        lines = [
            f"asp {lemon.aspect:.2f}"
            + ("" if lemon.aspect >= self.min_aspect else " round"),
            f"min {math.degrees(lemon.minor_angle):+.0f}deg",
            f"fill {lemon.fill:.2f}"
            + ("" if lemon.fill >= self.min_fill else " occ?"),
        ]

        # 楕円の下に中央揃えで置き、画像からはみ出さないようにする
        font, scale, step = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 14
        height, width = image.shape[:2]
        top = int(center[1] + lemon.major_radius) + 14
        top = min(top, height - step * (len(lines) - 1) - 4)

        for i, text in enumerate(lines):
            (w, _), _ = cv2.getTextSize(text, font, scale, 1)
            x = int(np.clip(center[0] - w / 2, 2, width - w - 2))
            org = (x, top + step * i)
            cv2.putText(image, text, org, font, scale, (0, 0, 0), 3)
            cv2.putText(image, text, org, font, scale, ELLIPSE_COLOR, 1)

    @staticmethod
    def draw_axis(image, center, half, color, thickness):

        p0 = tuple(int(round(v)) for v in np.asarray(center) - half)
        p1 = tuple(int(round(v)) for v in np.asarray(center) + half)
        cv2.line(image, p0, p1, color, thickness)

    # --------------------------------------------------

    @staticmethod
    def key(stamp):
        return (stamp.sec, stamp.nanosec)

    def trim(self, cache):
        while len(cache) > self.cache_size:
            cache.popitem(last=False)


def main():

    rclpy.init()

    node = LemonEllipseNode()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
