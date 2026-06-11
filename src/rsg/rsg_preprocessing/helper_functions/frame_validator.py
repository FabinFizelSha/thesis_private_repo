"""Frame validation helpers for the RSG preprocessor."""

from __future__ import annotations

from typing import Optional

from sensor_msgs.msg import Image

from rsg_preprocessing.helper_functions.config_loader import PreprocessorConfig


class FrameValidator:
    """Validation rules for synchronized RGB-D-pose frames."""

    def __init__(self, config: PreprocessorConfig) -> None:
        self.config = config

    def validate_resolution(self, rgb_msg: Image, depth_msg: Image) -> Optional[str]:
        """Validate that RGB and aligned depth resolutions match, if enabled."""
        if not self.config.check_resolution:
            return None
        if rgb_msg.width != depth_msg.width or rgb_msg.height != depth_msg.height:
            return "rgb_depth_resolution_mismatch"
        return None

    def validate_depth_ratio(self, invalid_depth_ratio: float) -> Optional[str]:
        """Validate invalid-depth ratio, if this check is enabled."""
        if not self.config.check_invalid_depth_ratio:
            return None
        if invalid_depth_ratio < 0.0:
            return None
        if invalid_depth_ratio > self.config.max_invalid_depth_ratio:
            return "too_many_invalid_depth_pixels"
        return None
