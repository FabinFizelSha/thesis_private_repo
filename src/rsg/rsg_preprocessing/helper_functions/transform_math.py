"""Rigid-body transform utilities used by the RSG preprocessor."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from nav_msgs.msg import Odometry


class TransformMath:
    """Small static helpers for quaternions, matrices, and odometry transforms."""

    @staticmethod
    def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Convert a quaternion to a 3x3 rotation matrix."""
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            return np.eye(3, dtype=np.float64)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        return np.array([
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ], dtype=np.float64)

    @staticmethod
    def matrix_to_quaternion(rotation: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert a 3x3 rotation matrix to quaternion in x, y, z, w order."""
        m = rotation
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            return 0.0, 0.0, 0.0, 1.0
        return qx / norm, qy / norm, qz / norm, qw / norm

    @staticmethod
    def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert roll, pitch, yaw to matrix using R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        return rz @ ry @ rx

    @staticmethod
    def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Create a 4x4 homogeneous transform from rotation and translation."""
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation.reshape(3)
        return transform

    @staticmethod
    def odom_to_transform(odom_msg: Odometry) -> np.ndarray:
        """Convert a ROS Odometry message into a homogeneous transform."""
        pose = odom_msg.pose.pose
        translation = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
        rotation = TransformMath.quaternion_to_matrix(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        return TransformMath.make_transform(rotation, translation)

    @staticmethod
    def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        """Spherically interpolate two quaternions in x, y, z, w order."""
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            result = q0 + alpha * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = math.acos(dot)
        theta = theta_0 * alpha
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q0) + (s1 * q1)
