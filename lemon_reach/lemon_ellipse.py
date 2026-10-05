#!/usr/bin/env python3
"""
Lemonのマスクに楕円を当てはめて、長辺・短辺を求める。

輪郭点に直接当てはめる (cv2.fitEllipse) と、葉で一部が欠けたマスクで極端な楕円に
なりやすいので、マスクを塗りつぶした画素の分布の 2 次モーメント (共分散) から
同じモーメントを持つ楕円を求める。一様に塗った楕円では、軸方向の分散が
(半径)^2 / 4 になるので、半径 = 2 * sqrt(固有値)。
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Ellipse:
    """画像座標 (x 右, y 下) の楕円"""

    center: np.ndarray      # (x, y) [px]
    major_dir: np.ndarray   # 長辺の向き (単位ベクトル)
    minor_dir: np.ndarray   # 短辺の向き (単位ベクトル)
    major: float            # 長辺の半径 [px]
    minor: float            # 短辺の半径 [px]
    fill: float             # マスクの面積 / 楕円の面積 (欠けたマスクほど小さい)

    @property
    def aspect(self):
        """長辺 / 短辺"""
        return self.major / max(self.minor, 1e-6)

    @property
    def minor_angle(self):
        """
        短辺が画像の横軸となす角 [rad] (-pi/2, pi/2]。y が下向きなので、
        正は画像上で時計回り。楕円の向きは 180 度で同じなので、この範囲に畳む
        """
        angle = math.atan2(self.minor_dir[1], self.minor_dir[0])

        if angle <= -math.pi / 2:
            angle += math.pi
        elif angle > math.pi / 2:
            angle -= math.pi

        return angle


def fit_ellipse(points, min_pixels=20):
    """
    マスクの輪郭 (画像座標の点列 [(x, y), ...]) から楕円を求める。
    点が少ない、または塗りつぶした画素が min_pixels 未満なら None
    """

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if len(pts) < 3:
        return None

    # 輪郭を囲む範囲だけで塗りつぶす
    origin = np.floor(pts.min(axis=0))
    size = np.ceil(pts.max(axis=0) - origin).astype(int) + 1

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts - origin).astype(np.int32)], 1)

    ys, xs = np.nonzero(mask)

    if len(xs) < min_pixels:
        return None

    coords = np.column_stack([xs, ys]).astype(np.float64)
    center = coords.mean(axis=0)
    cov = np.cov(coords - center, rowvar=False, bias=True)

    # 固有値は昇順: 0 番目が短辺、1 番目が長辺
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, 0.0)

    major = 2.0 * math.sqrt(values[1])
    minor = 2.0 * math.sqrt(values[0])

    if minor <= 0.0:
        return None

    return Ellipse(
        center=center + origin,
        major_dir=vectors[:, 1],
        minor_dir=vectors[:, 0],
        major=major,
        minor=minor,
        fill=len(xs) / (math.pi * major * minor),
    )
