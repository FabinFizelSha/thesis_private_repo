"""RGB and depth image conversion helpers.

The converter keeps the hot path cheap. If an input image already has the
required encoding, the output message reuses the original image data buffer and
only replaces the header frame id. Full cv_bridge conversion is used only when
an actual format/unit conversion is needed.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from rsg_preprocessing.helper_functions.config_loader import PreprocessorConfig


class ImageConverter:
    """Convert ROS image messages into normalized ROS image messages and arrays."""

    def __init__(self, bridge: CvBridge, config: PreprocessorConfig) -> None:
        self.bridge = bridge
        self.config = config

    def _shallow_image_reference(self, msg: Image, frame_id: str) -> Image:
        """Return a new Image message that reuses the original data buffer.

        ``copy.deepcopy(msg)`` copies the full image payload and can be costly
        for every frame. This method copies only metadata fields and reuses
        ``msg.data``. The original input message header is not modified.
        """
        out = Image()
        out.header = copy.copy(msg.header)
        out.header.frame_id = frame_id
        out.height = msg.height
        out.width = msg.width
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = msg.step
        out.data = msg.data
        return out

    def convert_rgb(self, rgb_msg: Image) -> Tuple[Optional[np.ndarray], Image]:
        """Convert input RGB image to the configured output encoding.

        If the input image already has the required encoding, the image data is
        forwarded without a full image copy. If conversion is disabled, the
        original encoding is forwarded as configured by the user.
        """
        target_encoding = self.config.output_rgb_encoding

        if rgb_msg.encoding == target_encoding or not self.config.convert_rgb:
            return None, self._shallow_image_reference(rgb_msg, self.config.camera_frame)

        rgb_array = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding=target_encoding)
        if rgb_array.dtype != np.uint8:
            rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

        out_msg = self.bridge.cv2_to_imgmsg(rgb_array, encoding=target_encoding)
        out_msg.header = copy.copy(rgb_msg.header)
        out_msg.header.frame_id = self.config.camera_frame
        return rgb_array, out_msg

    def convert_depth(self, depth_msg: Image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Image, float]:
        """Convert aligned depth to metres and optionally compute invalid-depth ratio.

        Fast path: if depth is already ``32FC1`` and the required output is also
        ``32FC1``, the original image payload is forwarded without constructing
        a new ROS image. A NumPy view is created only if invalid-depth checking
        is enabled.
        """
        target_encoding = self.config.output_depth_encoding

        # User explicitly requested no depth conversion. This is useful for
        # debugging, but should only be used when downstream accepts the input
        # encoding. Invalid-depth metadata can still be estimated if enabled.
        if not self.config.convert_depth:
            depth_out = self._shallow_image_reference(depth_msg, self.config.camera_frame)
            invalid_ratio = self._compute_invalid_ratio_if_enabled(depth_msg)
            return None, None, depth_out, invalid_ratio

        # Fast path for already-normalized metric float depth.
        if depth_msg.encoding == "32FC1" and target_encoding == "32FC1":
            depth_out = self._shallow_image_reference(depth_msg, self.config.camera_frame)
            depth_m = None
            invalid_ratio = -1.0
            if self.config.compute_invalid_depth_ratio or self.config.check_invalid_depth_ratio:
                depth_m = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_m = depth_m.astype(np.float32, copy=False)
                invalid_ratio = self.compute_invalid_depth_ratio(depth_m)
            return depth_m, None, depth_out, invalid_ratio

        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_msg.encoding == "16UC1":
            depth_m = depth_raw.astype(np.float32) * self.config.depth_scale_to_meter
        elif depth_msg.encoding == "32FC1":
            depth_m = depth_raw.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported depth encoding: {depth_msg.encoding}")

        invalid_ratio = self.compute_invalid_depth_ratio(depth_m)
        valid_mask = None
        if self.config.compute_invalid_depth_ratio:
            valid_mask = self.make_valid_depth_mask(depth_m)

        out_msg = self.bridge.cv2_to_imgmsg(depth_m, encoding=target_encoding)
        out_msg.header = copy.copy(depth_msg.header)
        out_msg.header.frame_id = self.config.camera_frame
        return depth_m, valid_mask, out_msg, invalid_ratio

    def _compute_invalid_ratio_if_enabled(self, depth_msg: Image) -> float:
        """Compute depth validity only when configured to do so."""
        if not self.config.compute_invalid_depth_ratio and not self.config.check_invalid_depth_ratio:
            return -1.0

        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_msg.encoding == "16UC1":
            depth_m = depth_raw.astype(np.float32) * self.config.depth_scale_to_meter
        elif depth_msg.encoding == "32FC1":
            depth_m = depth_raw.astype(np.float32, copy=False)
        else:
            return -1.0
        return self.compute_invalid_depth_ratio(depth_m)

    def make_valid_depth_mask(self, depth_m: np.ndarray) -> np.ndarray:
        """Return a validity mask for metric depth values."""
        return (
            np.isfinite(depth_m)
            & (depth_m >= self.config.min_depth_m)
            & (depth_m <= self.config.max_depth_m)
        )

    def compute_invalid_depth_ratio(self, depth_m: np.ndarray) -> float:
        """Compute invalid-depth ratio, optionally on a sampled depth grid."""
        if not self.config.compute_invalid_depth_ratio and not self.config.check_invalid_depth_ratio:
            return -1.0

        stride = max(1, int(self.config.depth_check_stride))
        sample = depth_m[::stride, ::stride] if stride > 1 else depth_m
        valid_mask = (
            np.isfinite(sample)
            & (sample >= self.config.min_depth_m)
            & (sample <= self.config.max_depth_m)
        )
        return 1.0 - float(np.count_nonzero(valid_mask)) / float(valid_mask.size)
