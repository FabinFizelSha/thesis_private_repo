"""Camera IMU buffering and timestamp association."""

from __future__ import annotations

from typing import List, Optional, Tuple

from sensor_msgs.msg import Imu

from rsg_preprocessing.helper_functions.time_utils import stamp_to_float


class ImuBuffer:
    """Time-ordered IMU buffer with nearest-sample lookup."""

    def __init__(self, max_size: int, tolerance_sec: float, assume_ordered: bool = True) -> None:
        self.max_size = max_size
        self.tolerance_sec = tolerance_sec
        self.assume_ordered = assume_ordered
        self._messages: List[Imu] = []

    def add(self, msg: Imu) -> None:
        """Add an IMU message to the buffer."""
        self._messages.append(msg)
        if not self.assume_ordered:
            self._messages.sort(key=lambda m: stamp_to_float(m.header.stamp))
        if len(self._messages) > self.max_size:
            self._messages = self._messages[-self.max_size:]

    def lookup(self, target_time: float) -> Tuple[Optional[Imu], Optional[float], str]:
        """Find the nearest IMU sample for a target RGB timestamp."""
        if not self._messages:
            return None, None, "imu_buffer_empty"
        nearest = min(self._messages, key=lambda m: abs(stamp_to_float(m.header.stamp) - target_time))
        delta = abs(stamp_to_float(nearest.header.stamp) - target_time)
        if delta > self.tolerance_sec:
            return None, delta, "nearest_imu_outside_tolerance"
        return nearest, delta, "nearest_imu"
