#!/usr/bin/env python3
"""
URDF の直列リンク (root → tip) の順運動学とヤコビアン。

MoveIt Servo と同じく、tip リンクの原点での 6 x n ヤコビアン (並進 [m] と回転 [rad])
の条件数で特異点の近さを測る。
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_matrix(axis, angle):
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])

    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


class Chain:
    """root から tip までの関節の並び (固定関節を含む)"""

    def __init__(self, urdf, root, tip):

        joints = {}

        # <transmission> などの中の <joint> は除き、ロボット直下の関節だけを見る
        for joint in ET.fromstring(urdf).findall("joint"):

            origin = joint.find("origin")
            axis = joint.find("axis")

            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]

            if origin is not None:
                xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]

            transform = np.eye(4)
            transform[:3, :3] = _rpy_matrix(*rpy)
            transform[:3, 3] = xyz

            a = [1.0, 0.0, 0.0]
            if axis is not None:
                a = [float(v) for v in axis.get("xyz", "1 0 0").split()]

            joints[joint.find("child").get("link")] = {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "origin": transform,
                "axis": np.array(a) / np.linalg.norm(a),
            }

        # tip から root へたどる
        self.joints = []
        link = tip

        while link != root:
            if link not in joints:
                raise ValueError(f"no chain from {root} to {tip}")
            self.joints.append(joints[link])
            link = joints[link]["parent"]

        self.joints.reverse()
        self.joint_names = [
            j["name"] for j in self.joints if j["type"] in ("revolute", "continuous")
        ]

    def jacobian(self, positions):
        """positions {name: 角度} での tip 原点のヤコビアン (root 座標)"""

        T = np.eye(4)
        origins = []
        axes = []

        for j in self.joints:

            T = T @ j["origin"]

            if j["type"] in ("revolute", "continuous"):
                origins.append(T[:3, 3].copy())
                axes.append(T[:3, :3] @ j["axis"])

                R = np.eye(4)
                R[:3, :3] = _axis_angle_matrix(j["axis"], positions[j["name"]])
                T = T @ R

        tip = T[:3, 3]

        return np.vstack([
            np.column_stack([np.cross(a, tip - o) for o, a in zip(origins, axes)]),
            np.column_stack(axes),
        ])

    def condition(self, positions):
        """ヤコビアンの条件数 (最大特異値 / 最小特異値)"""

        s = np.linalg.svd(self.jacobian(positions), compute_uv=False)

        return s[0] / max(s[-1], 1e-12)
