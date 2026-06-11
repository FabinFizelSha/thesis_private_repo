"""Odometry buffering and timestamp association."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from nav_msgs.msg import Odometry

from rsg_preprocessing.helper_functions.time_utils import stamp_to_float
from rsg_preprocessing.helper_functions.transform_math import TransformMath


class OdomBuffer:
    """Time-ordered odometry buffer with nearest and interpolated lookup."""

    def __init__(self, max_size: int, tolerance_sec: float, use_interpolation: bool, assume_ordered: bool = True) -> None:
        self.max_size = max_size
        self.tolerance_sec = tolerance_sec
        self.use_interpolation = use_interpolation
        self.assume_ordered = assume_ordered
        self._messages: List[Odometry] = []

    def add(self, msg: Odometry) -> None:
        """Add an odometry message to the buffer.

        In normal live/rosbag operation, odometry arrives in timestamp order.
        ``assume_ordered=True`` avoids sorting on every callback for speed.
        """
        self._messages.append(msg)
        if not self.assume_ordered:
            self._messages.sort(key=lambda m: stamp_to_float(m.header.stamp))
        if len(self._messages) > self.max_size:
            self._messages = self._messages[-self.max_size:]

    def lookup(self, target_time: float) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        """Find or interpolate odometry transform for a target RGB time."""
        if not self._messages:
            return None, None, "odom_buffer_empty"
        if self.use_interpolation:
            interpolated = self._lookup_interpolated(target_time)
            if interpolated[0] is not None:
                return interpolated
        return self._lookup_nearest(target_time)

    def _lookup_nearest(self, target_time: float) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        nearest = min(self._messages, key=lambda m: abs(stamp_to_float(m.header.stamp) - target_time))
        delta = abs(stamp_to_float(nearest.header.stamp) - target_time)
        if delta > self.tolerance_sec:
            return None, delta, "nearest_odom_outside_tolerance"
        return TransformMath.odom_to_transform(nearest), delta, "nearest_odom"

    def _lookup_interpolated(self, target_time: float) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        before: Optional[Odometry] = None
        after: Optional[Odometry] = None
        for msg in self._messages:
            msg_time = stamp_to_float(msg.header.stamp)
            if msg_time <= target_time:
                before = msg
            elif msg_time > target_time:
                after = msg
                break
        if before is None or after is None:
            return None, None, "interpolation_bounds_missing"

        before_time = stamp_to_float(before.header.stamp)
        after_time = stamp_to_float(after.header.stamp)
        nearest_delta = min(abs(target_time - before_time), abs(after_time - target_time))
        if nearest_delta > self.tolerance_sec:
            return None, nearest_delta, "interpolated_odom_outside_tolerance"
        if after_time <= before_time:
            return TransformMath.odom_to_transform(before), 0.0, "duplicate_odom_time"

        alpha = (target_time - before_time) / (after_time - before_time)
        p0 = before.pose.pose.position
        p1 = after.pose.pose.position
        translation = np.array([
            p0.x + alpha * (p1.x - p0.x),
            p0.y + alpha * (p1.y - p0.y),
            p0.z + alpha * (p1.z - p0.z),
        ], dtype=np.float64)

        o0 = before.pose.pose.orientation
        o1 = after.pose.pose.orientation
        q0 = np.array([o0.x, o0.y, o0.z, o0.w], dtype=np.float64)
        q1 = np.array([o1.x, o1.y, o1.z, o1.w], dtype=np.float64)
        q = TransformMath.slerp(q0, q1, alpha)
        rotation = TransformMath.quaternion_to_matrix(q[0], q[1], q[2], q[3])
        return TransformMath.make_transform(rotation, translation), nearest_delta, "interpolated_odom"
