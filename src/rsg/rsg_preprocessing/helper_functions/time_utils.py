"""Small time utilities for ROS timestamp handling."""

from __future__ import annotations

from typing import Any


def stamp_to_float(stamp: Any) -> float:
    """Convert a ROS stamp with sec/nanosec fields to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9
