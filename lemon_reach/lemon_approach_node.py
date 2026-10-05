#!/usr/bin/env python3

import math
import xml.etree.ElementTree as ET

import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PointStamped, TransformStamped, Vector3Stamped
from moveit_msgs.msg import Constraints, JointConstraint
from std_msgs.msg import String
from std_srvs.srv import Trigger

from lemon_reach.kinematics import Chain
from lemon_reach.moveit_helper import MoveItHelper


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axis_rotation(axis, angle):
    """axis (0: x, 1: y, 2: z) まわりの回転行列"""

    c, s = math.cos(angle), math.sin(angle)
    i, j = [k for k in range(3) if k != axis]

    m = np.eye(3)
    m[i, i] = c
    m[i, j] = -s
    m[j, i] = s
    m[j, j] = c

    return m


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

    その後の把持 (lemon_grasp_node) は姿勢を保ったまま指先をまっすぐLemonへ
    入れるので、pregrasp から Lemon の insert_margin 奥までの直線上を IK で
    確かめ、入れられる回転と関節角を選んで、その関節角へ移動する。Lemonが
    届く範囲の端にあると、pregrasp には行けても関節限界で入れられない。
    直線上で特異点に近づく (ヤコビアンの条件数が max_condition を超える) 経路も
    捨てる。Servo がそこで減速・停止して挿入がタイムアウトするため。
    ee_link → Lemon の向きで入れられなければ、base から見たLemonの方向に
    approach_pitches の角度で上から入る向きを順に試す。

    lemon_target が楕円の短辺の向き (/lemon_target/short_axis) を出していれば、
    approach_axis まわりの回転は、grasp_axis (指が閉じる向き) を短辺に合わせる角度
    にする。指の向きは 180 度で同じなので、down_axis が下に近い方を先に試す
    (カメラが手先の上側に残る)。どの向きでも入れられなければ短辺を無視して探し直す。
    短辺に合わせた姿勢を grasp_frame の TF としてLemonの位置に出す。

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
        # 指が閉じる軸 (piper: joint7/8 は gripper_base の y 方向に動く)
        self.declare_parameter("grasp_axis", "y")
        # 短辺を approach に垂直な面へ投影した長さがこれ未満 (短辺が approach と
        # ほぼ平行で、指の向きが決まらない) なら短辺を使わない
        self.declare_parameter("min_axis_projection", 0.5)
        # Lemon中心から指先までの距離 [m]
        self.declare_parameter("standoff", 0.10)
        # ee_link から指先までの approach_axis 方向の距離 [m]
        # (piper: joint7/8 が gripper_base の z=0.1358、指先もほぼそこ)
        self.declare_parameter("tcp_offset", 0.136)
        # approach_axis まわりの回転を roll_step 刻みで roll_tolerance まで探す [rad]
        # pi/2 未満にすると down_axis が上を向く姿勢は許さない
        self.declare_parameter("roll_tolerance", 1.4)
        self.declare_parameter("roll_step", 0.35)
        # ee_link → Lemon の向きで入れられないときに試す、水平から下向きの角度 [deg]
        # (根元に近い低いLemonは上からでないと入れられない)
        self.declare_parameter(
            "approach_pitches", [60.0, 75.0, 90.0, 45.0, 30.0, 15.0, 0.0]
        )
        # 指先をLemon中心からさらに奥へ入れられることを確かめる距離 [m]
        # (追従でLemonの推定が動く分と、Servo の関節限界の余裕)
        self.declare_parameter("insert_margin", 0.03)
        # 直線上で IK を確かめる点の数と、隣り合う解の関節角の差の上限 [rad]
        # 特異点を通る区間は短いので細かく確かめる
        self.declare_parameter("ik_check_points", 10)
        self.declare_parameter("max_joint_step", 0.5)
        self.declare_parameter(
            "joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        )
        self.declare_parameter("joint_tolerance", 0.01)
        # 直線上の解は関節限界からこれ以上離れていること [rad]
        # (Servo の joint_limit_margin 0.1 より内側で止まらないように)
        self.declare_parameter("joint_limit_margin", 0.15)
        # 直線上のヤコビアン (kinematics_tip の原点) の条件数の上限。
        # Servo は lower_singularity_threshold (120) から減速し始める
        self.declare_parameter("max_condition", 100.0)
        self.declare_parameter("kinematics_root", "base_link")
        self.declare_parameter("kinematics_tip", "link6")
        self.declare_parameter("pregrasp_frame", "lemon_pregrasp")
        # 指先をLemonに入れたときの姿勢 (短辺に合わせた向き) のTF
        self.declare_parameter("grasp_frame", "lemon_grasp_frame")
        self.declare_parameter("tf_rate", 20.0)

        self.ee_link = self.get_parameter("ee_link").value
        self.standoff = self.get_parameter("standoff").value
        self.tcp_offset = self.get_parameter("tcp_offset").value
        self.roll_tolerance = self.get_parameter("roll_tolerance").value
        self.roll_step = self.get_parameter("roll_step").value
        self.approach_pitches = self.get_parameter("approach_pitches").value
        self.insert_margin = self.get_parameter("insert_margin").value
        self.ik_check_points = self.get_parameter("ik_check_points").value
        self.max_joint_step = self.get_parameter("max_joint_step").value
        self.joint_names = self.get_parameter("joint_names").value
        self.joint_tolerance = self.get_parameter("joint_tolerance").value
        self.joint_limit_margin = self.get_parameter("joint_limit_margin").value
        self.max_condition = self.get_parameter("max_condition").value
        self.joint_limits = None    # {name: (lower, upper)}
        self.chain = None           # 条件数の計算 (robot_description から)
        self.rejects = {}           # 直線挿入を捨てた理由ごとの回数

        axes = {"x": 0, "y": 1, "z": 2}
        self.approach_axis = axes[self.get_parameter("approach_axis").value]
        self.down_axis = axes[self.get_parameter("down_axis").value]

        if self.approach_axis == self.down_axis:
            raise ValueError("approach_axis and down_axis must differ")
        self.grasp_axis = axes[self.get_parameter("grasp_axis").value]

        if len({self.approach_axis, self.down_axis, self.grasp_axis}) != 3:
            raise ValueError("approach_axis, down_axis and grasp_axis must differ")

        self.min_axis_projection = self.get_parameter("min_axis_projection").value
        self.pregrasp_frame = self.get_parameter("pregrasp_frame").value
        self.grasp_frame_id = self.get_parameter("grasp_frame").value

        self.target = None      # PointStamped
        self.short_axis = None  # np.array (target frame) or None
        self.pregrasp = None    # TransformStamped
        self.grasp_frame = None  # TransformStamped
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

        self.create_subscription(
            Vector3Stamped,
            "/lemon_target/short_axis",
            self.short_axis_callback,
            latched,
            callback_group=self.cb_group
        )

        self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
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
            self.grasp_frame = None
            return

        self.target = msg

    def short_axis_callback(self, msg):

        v = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

        # frame_idが空、またはゼロベクトルなら向きは定まっていない
        if not msg.header.frame_id or np.linalg.norm(v) < 1e-6:
            self.short_axis = None
            return

        self.short_axis = (msg.header.frame_id, v / np.linalg.norm(v))

    def robot_description_callback(self, msg):

        limits = {}

        for joint in ET.fromstring(msg.data).iter("joint"):

            limit = joint.find("limit")

            if joint.get("name") in self.joint_names and limit is not None:
                limits[joint.get("name")] = (
                    float(limit.get("lower", -math.inf)),
                    float(limit.get("upper", math.inf)),
                )

        self.joint_limits = limits

        try:
            self.chain = Chain(
                msg.data,
                self.get_parameter("kinematics_root").value,
                self.get_parameter("kinematics_tip").value,
            )
        except ValueError as e:
            self.get_logger().error(f"Cannot build kinematic chain: {e}")

    def max_condition_between(self, q0, q1, steps=4):
        """q0 から q1 まで関節空間で補間した姿勢の条件数の最大"""

        return max(
            self.chain.condition({
                name: q0[name] + t * (q1[name] - q0[name])
                for name in self.chain.joint_names
            })
            for t in np.linspace(0.0, 1.0, steps + 1)
        )

    def reject(self, reason):
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        return None

    def within_limits(self, joints):

        if self.joint_limits is None:
            self.get_logger().warn(
                "Joint limits are not received from /robot_description",
                throttle_duration_sec=5.0
            )
            return True

        return all(
            lower + self.joint_limit_margin
            <= joints[name]
            <= upper - self.joint_limit_margin
            for name, (lower, upper) in self.joint_limits.items()
        )

    # --------------------------------------------------
    # pregrasp の計算
    # --------------------------------------------------

    def approach_directions(self, frame_id):
        """
        試す approach の向き (frame_id での単位ベクトル)。最初は ee_link → Lemon、
        その後は frame_id の原点から見たLemonの方向に approach_pitches の角度で
        下向きにしたもの
        """

        lemon = np.array([
            self.target.point.x, self.target.point.y, self.target.point.z
        ])
        yaw = math.atan2(lemon[1], lemon[0])

        directions = [None]

        for pitch in np.radians(self.approach_pitches):
            directions.append(np.array([
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                -math.sin(pitch),
            ]))

        return directions

    def compute_pregrasp(self, approach=None):
        """approach (None なら ee_link → Lemon) の向きで入る pregrasp"""

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

        if approach is None:

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

    # --------------------------------------------------
    # 直線で入れられるかの確認
    # --------------------------------------------------

    def default_rolls(self):
        """0, +step, -step, +2step, ... (|roll| <= roll_tolerance)"""

        n = int(self.roll_tolerance / self.roll_step)

        return [0.0] + [
            sign * k * self.roll_step
            for k in range(1, n + 1) for sign in (1.0, -1.0)
        ]

    def aligned_rolls(self, pregrasp):
        """
        grasp_axis を短辺に合わせる approach_axis まわりの回転 [roll, roll ± pi]。
        短辺が approach とほぼ平行なら None
        """

        frame_id, short = self.short_axis

        if frame_id != pregrasp.header.frame_id:
            self.get_logger().warn(
                f"short_axis frame {frame_id} != {pregrasp.header.frame_id}",
                throttle_duration_sec=5.0
            )
            return None

        r = pregrasp.transform.rotation
        base = quat_to_matrix(r.x, r.y, r.z, r.w)
        approach = base[:, self.approach_axis]
        grasp = base[:, self.grasp_axis]

        # 短辺を approach に垂直な面へ投影
        target = short - np.dot(short, approach) * approach

        if np.linalg.norm(target) < self.min_axis_projection:
            return None

        target /= np.linalg.norm(target)

        # grasp を approach まわりに回して target に合わせる角度
        roll = math.atan2(
            np.dot(np.cross(grasp, target), approach), np.dot(grasp, target)
        )

        # 180 度回しても同じなので、回す量が小さい方 (down_axis が下に近い方) を先に
        if roll > math.pi / 2:
            roll -= math.pi
        elif roll <= -math.pi / 2:
            roll += math.pi

        return [roll, roll - math.pi if roll > 0.0 else roll + math.pi]

    def find_insertion(self, pregrasp, rolls):
        """
        pregrasp の指先から Lemon の insert_margin 奥まで、姿勢を保ったまま
        まっすぐ入れられる approach_axis まわりの回転を rolls の順に探す。

        @return (その回転の pregrasp, pregrasp での関節角 {name: position})
                見つからなければ None
        """

        seed = self.moveit.get_joint_positions()

        if seed is None:
            return None

        frame_id = pregrasp.header.frame_id
        r = pregrasp.transform.rotation
        base = quat_to_matrix(r.x, r.y, r.z, r.w)
        approach = base[:, self.approach_axis]
        t = pregrasp.transform.translation
        start = np.array([t.x, t.y, t.z])
        length = self.standoff + self.insert_margin

        for roll in rolls:

            q = matrix_to_quat(base @ axis_rotation(self.approach_axis, roll))
            joints = None

            # 一番奥から pregrasp へ、前の解を初期値にして解き、途中で
            # 別の解 (腕の構えの切り替え) に飛ばないことを確かめる
            for d in np.linspace(length, 0.0, self.ik_check_points):

                ee = start + (d - self.tcp_offset) * approach
                solution = self.moveit.compute_ik(
                    frame_id, self.ee_link, ee, q,
                    seed if joints is None else joints
                )

                if solution is None:
                    solution = self.reject("ik")
                    break

                if not self.within_limits(solution):
                    solution = self.reject("joint_limit")
                    break

                if joints is not None and max(
                    abs(solution[j] - joints[j]) for j in self.joint_names
                ) > self.max_joint_step:
                    solution = self.reject("jump")
                    break

                # 前の点からこの点までの間も含めて特異点に近づかないこと
                if self.chain is not None and self.max_condition_between(
                    solution if joints is None else joints, solution
                ) > self.max_condition:
                    solution = self.reject("singularity")
                    break

                joints = solution

            if solution is None:
                continue

            a = np.round(approach, 2)
            self.get_logger().info(
                f"Straight insertion found (approach={a}, roll={roll:.2f})"
            )

            goal = TransformStamped()
            goal.header.frame_id = frame_id
            goal.child_frame_id = self.pregrasp_frame
            goal.transform.translation = pregrasp.transform.translation
            (
                goal.transform.rotation.x,
                goal.transform.rotation.y,
                goal.transform.rotation.z,
                goal.transform.rotation.w,
            ) = (float(v) for v in q)

            return goal, {j: joints[j] for j in self.joint_names}

        return None

    def make_constraints(self, joints):

        constraints = Constraints()

        for name, position in joints.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = position
            jc.tolerance_above = self.joint_tolerance
            jc.tolerance_below = self.joint_tolerance
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        return constraints

    def search_insertion(self):
        """
        接近方向と回転を探す。短辺の向きがあれば、まず短辺に合わせた回転だけで探し、
        見つからなければ短辺を無視して探し直す。

        @return (pregrasp, 関節角)、見つからなければ None、pregrasp が作れなければ False
        """

        modes = ["aligned", "default"] if self.short_axis else ["default"]
        self.rejects = {}

        if self.chain is None:
            self.get_logger().warn(
                "Kinematic chain is not ready; singularities are not checked"
            )

        for mode in modes:

            for approach in self.approach_directions(self.target.header.frame_id):

                pregrasp = self.compute_pregrasp(approach)

                if pregrasp is None:
                    return False

                if mode == "aligned":
                    rolls = self.aligned_rolls(pregrasp)

                    # この方向からは短辺の向きに指を合わせられない
                    if rolls is None:
                        continue
                else:
                    rolls = self.default_rolls()

                insertion = self.find_insertion(pregrasp, rolls)

                if insertion is not None:
                    self.get_logger().info(
                        "Grasp aligned with the short axis" if mode == "aligned"
                        else "Grasp not aligned with the short axis"
                    )
                    return insertion

            if mode == "aligned":
                self.get_logger().warn(
                    "No straight insertion aligned with the short axis; ignoring it "
                    f"(rejected: {self.rejects})"
                )

        self.get_logger().warn(f"No straight insertion (rejected: {self.rejects})")

        return None

    def make_grasp_frame(self, pregrasp):
        """pregrasp と同じ姿勢でLemonの位置に置いた TF"""

        t = TransformStamped()
        t.header.frame_id = pregrasp.header.frame_id
        t.child_frame_id = self.grasp_frame_id
        t.transform.translation.x = self.target.point.x
        t.transform.translation.y = self.target.point.y
        t.transform.translation.z = self.target.point.z
        t.transform.rotation = pregrasp.transform.rotation

        return t

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
            insertion = self.search_insertion()

            if insertion is False:
                response.success = False
                response.message = "failed to compute pregrasp"
                return response

            if insertion is None:
                response.success = False
                response.message = "cannot insert straight into the lemon"
                return response

            pregrasp, joints = insertion
            self.pregrasp = pregrasp
            self.grasp_frame = self.make_grasp_frame(pregrasp)

            p = pregrasp.transform.translation
            self.get_logger().info(
                f"Approach {pregrasp.header.frame_id}: "
                f"pregrasp=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )
            self.publish_status("approaching")

            success, message = self.moveit.plan_and_execute(
                self.make_constraints(joints), "approach"
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

        stamp = self.get_clock().now().to_msg()
        self.pregrasp.header.stamp = stamp
        transforms = [self.pregrasp]

        if self.grasp_frame is not None:
            self.grasp_frame.header.stamp = stamp
            transforms.append(self.grasp_frame)

        self.tf_broadcaster.sendTransform(transforms)

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
