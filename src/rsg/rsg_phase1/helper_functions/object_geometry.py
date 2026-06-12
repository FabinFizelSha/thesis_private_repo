"""Object geometry estimation from masks, depth, camera intrinsics, and pose."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class ObjectGeometryEstimator:
    """Estimate 2D/3D metadata for SAM mask candidates.

    The estimator is deliberately lightweight. It samples mask pixels using a
    configurable stride, rejects invalid depth values, back-projects image
    points to camera coordinates, and transforms them using the existing
    framework-compatible ``tx`` and ``rotM`` fields.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def estimate(self, mask: np.ndarray, depth_m: np.ndarray, camera_info: Any, tx: np.ndarray, rot_m: np.ndarray) -> Dict[str, Any]:
        """Return geometry metadata for one binary mask."""
        geometry: Dict[str, Any] = {}
        ys, xs = np.where(mask)
        if xs.size == 0:
            geometry["valid_geometry"] = False
            geometry["reason"] = "empty_mask"
            return geometry

        bbox_2d = self._bbox_2d(xs, ys)
        centroid_2d = [float(np.median(xs)), float(np.median(ys))]
        geometry["bbox_2d"] = bbox_2d
        geometry["centroid_2d"] = centroid_2d
        geometry["mask_area_px"] = int(xs.size)

        if not self.config.estimate_object_geometry:
            geometry["valid_geometry"] = False
            geometry["reason"] = "geometry_disabled"
            return geometry

        stride = max(1, int(self.config.projection_stride))
        xs_sample = xs[::stride]
        ys_sample = ys[::stride]
        z = depth_m[ys_sample, xs_sample].astype(np.float64, copy=False)
        valid = np.isfinite(z) & (z >= self.config.min_depth_m) & (z <= self.config.max_depth_m)
        xs_valid = xs_sample[valid].astype(np.float64)
        ys_valid = ys_sample[valid].astype(np.float64)
        z_valid = z[valid]

        total = int(z.size)
        valid_count = int(z_valid.size)
        geometry["depth_valid_points"] = valid_count
        geometry["depth_sample_points"] = total
        geometry["depth_valid_ratio"] = 0.0 if total == 0 else float(valid_count / total)

        if valid_count < self.config.min_valid_depth_points:
            geometry["valid_geometry"] = False
            geometry["reason"] = "too_few_valid_depth_points"
            return geometry

        fx, fy, cx, cy = self._intrinsics(camera_info)
        x_cam = (xs_valid - cx) * z_valid / fx
        y_cam = (ys_valid - cy) * z_valid / fy
        points_cam = np.stack([x_cam, y_cam, z_valid], axis=1)
        points_world = (rot_m @ points_cam.T).T + tx.reshape(1, 3)

        if self.config.centroid_method.lower() == "mean":
            centroid = np.mean(points_world, axis=0)
        else:
            centroid = np.median(points_world, axis=0)

        min_xyz = np.min(points_world, axis=0)
        max_xyz = np.max(points_world, axis=0)
        size_xyz = np.maximum(max_xyz - min_xyz, 0.0)
        volume = float(size_xyz[0] * size_xyz[1] * size_xyz[2])

        geometry["valid_geometry"] = True
        geometry["centroid_3d"] = [float(v) for v in centroid]
        geometry["bbox_3d_min"] = [float(v) for v in min_xyz]
        geometry["bbox_3d_max"] = [float(v) for v in max_xyz]
        geometry["bbox_3d_size"] = [float(v) for v in size_xyz]
        geometry["bbox_volume_m3"] = volume
        geometry["depth_min_m"] = float(np.min(z_valid))
        geometry["depth_max_m"] = float(np.max(z_valid))
        geometry["depth_median_m"] = float(np.median(z_valid))
        return geometry

    @staticmethod
    def _bbox_2d(xs: np.ndarray, ys: np.ndarray) -> list[int]:
        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        return [x_min, y_min, int(x_max - x_min + 1), int(y_max - y_min + 1)]

    @staticmethod
    def _intrinsics(camera_info: Any) -> Tuple[float, float, float, float]:
        k = list(camera_info.k)
        fx = float(k[0]) if k[0] else 1.0
        fy = float(k[4]) if k[4] else 1.0
        cx = float(k[2])
        cy = float(k[5])
        return fx, fy, cx, cy


def filter_metadata(metadata: Dict[str, Any], config: Any) -> Dict[str, Any]:
    """Remove optional metadata fields according to config switches."""
    if not config.include_bbox_2d:
        metadata.pop("bbox_2d", None)
    if not config.include_centroid_2d:
        metadata.pop("centroid_2d", None)
    if not config.include_centroid_3d:
        metadata.pop("centroid_3d", None)
    if not config.include_bbox_3d:
        for key in ["bbox_3d_min", "bbox_3d_max", "bbox_3d_size"]:
            metadata.pop(key, None)
    if not config.include_bbox_volume:
        metadata.pop("bbox_volume_m3", None)
    if not config.include_depth_stats:
        for key in ["depth_valid_points", "depth_sample_points", "depth_valid_ratio", "depth_min_m", "depth_max_m", "depth_median_m"]:
            metadata.pop(key, None)
    if not config.include_mask_area:
        metadata.pop("mask_area_px", None)
    return metadata
