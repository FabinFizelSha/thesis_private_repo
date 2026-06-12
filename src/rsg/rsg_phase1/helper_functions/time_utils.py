"""Time conversion helpers for Phase 1 nodes."""

from __future__ import annotations


def stamp_to_float(stamp) -> float:
    """Convert a ROS builtin_interfaces/Time stamp to seconds as float."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9
