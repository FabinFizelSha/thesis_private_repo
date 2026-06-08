"""Segmentation backends for the RSG semantic adapter.

The module exposes a small object-oriented interface:

``DummyGridSegmenter``
    Lightweight test segmenter that returns a rectangular mask.

``SamAutomaticSegmenter``
    SAM automatic mask generator with optional depth gating, ROI cropping, and
    input resizing. These optimizations are optional and exist to reduce
    computation before RAP/VLM processing.

All segmenters return a list of ``MaskProposal`` objects in full-resolution
image coordinates. The ROS node can then rasterize these masks into a dense
Hydra-compatible semantic image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MaskProposal:
    """A single 2D object/region mask proposal.

    Attributes
    ----------
    mask:
        Full-resolution boolean mask with shape ``H x W``.
    score:
        Segmenter confidence / quality score.
    source:
        Short string describing how the mask was produced.
    bbox:
        Full-resolution bounding box as ``(x, y, width, height)``.
    """

    mask: np.ndarray
    score: float
    source: str
    bbox: Tuple[int, int, int, int]


class DummyGridSegmenter:
    """Simple deterministic segmenter for ROS/Hydra plumbing tests."""

    def __init__(self, box_fraction: float = 0.35):
        self.box_fraction = float(box_fraction)

    def segment(
        self, rgb: np.ndarray, depth_m: Optional[np.ndarray] = None
    ) -> List[MaskProposal]:
        h, w = rgb.shape[:2]
        bw = int(w * self.box_fraction)
        bh = int(h * self.box_fraction)
        x0 = int((w - bw) / 2)
        y0 = int((h - bh) / 2)

        mask = np.zeros((h, w), dtype=bool)
        mask[y0 : y0 + bh, x0 : x0 + bw] = True

        return [
            MaskProposal(mask=mask, score=1.0, source="dummy_grid", bbox=(x0, y0, bw, bh))
        ]


class SamAutomaticSegmenter:
    """SAM automatic-mask backend with robotics-oriented filtering.

    The expensive part is SAM itself. The optional depth gate and ROI mode reduce
    the amount of image content that SAM sees:

    1. Construct a valid-depth mask from ``depth_m``.
    2. Extract connected depth-supported regions of interest.
    3. Run SAM on those crops, not necessarily on the whole image.
    4. Resize/copy masks back into full-resolution image coordinates.
    5. Reject masks that are too small, too large, or weakly supported by depth.
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda",
        min_area: int = 800,
        max_masks: int = 25,
        *,
        use_depth_gate: bool = True,
        min_depth_m: float = 0.25,
        max_depth_m: float = 3.0,
        min_near_pixels: int = 1500,
        min_mask_depth_overlap: float = 0.50,
        depth_dilate_px: int = 9,
        use_depth_roi: bool = True,
        roi_padding_px: int = 24,
        min_roi_area_px: int = 2500,
        max_rois: int = 4,
        points_per_side: int = 8,
        pred_iou_thresh: float = 0.92,
        stability_score_thresh: float = 0.88,
        resize_input: bool = True,
        resize_max_side: int = 512,
        resize_min_scale: float = 0.25,
    ):
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "segment_anything is not installed. Use segmenter_mode:=dummy_grid "
                "or install Meta Segment Anything."
            ) from exc

        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint does not exist: {checkpoint_path}. "
                "Make sure sam_checkpoint and sam_model_type match."
            )

        self.min_area = int(min_area)
        self.max_masks = int(max_masks)

        self.use_depth_gate = bool(use_depth_gate)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_near_pixels = int(min_near_pixels)
        self.min_mask_depth_overlap = float(min_mask_depth_overlap)
        self.depth_dilate_px = int(depth_dilate_px)

        self.use_depth_roi = bool(use_depth_roi)
        self.roi_padding_px = int(roi_padding_px)
        self.min_roi_area_px = int(min_roi_area_px)
        self.max_rois = int(max_rois)

        self.resize_input = bool(resize_input)
        self.resize_max_side = int(resize_max_side)
        self.resize_min_scale = float(resize_min_scale)

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=int(points_per_side),
            pred_iou_thresh=float(pred_iou_thresh),
            stability_score_thresh=float(stability_score_thresh),
        )

    def segment(
        self, rgb: np.ndarray, depth_m: Optional[np.ndarray] = None
    ) -> List[MaskProposal]:
        """Return full-resolution mask proposals for one RGB-D frame."""
        h, w = rgb.shape[:2]
        valid_depth_mask: Optional[np.ndarray] = None

        if self.use_depth_gate and depth_m is not None:
            valid_depth_mask = self._make_valid_depth_mask(depth_m)
            if int(valid_depth_mask.sum()) < self.min_near_pixels:
                return []

        if self.use_depth_roi and valid_depth_mask is not None:
            rois = self._extract_depth_rois(valid_depth_mask, w, h)
            proposals: List[MaskProposal] = []
            for roi in rois:
                proposals.extend(self._run_sam_on_roi(rgb, valid_depth_mask, roi))
        else:
            proposals = self._run_sam_on_roi(rgb, valid_depth_mask, (0, 0, w, h))

        proposals.sort(key=lambda p: int(p.mask.sum()), reverse=True)
        return proposals[: self.max_masks]

    def _make_valid_depth_mask(self, depth_m: np.ndarray) -> np.ndarray:
        """Return pixels whose depth is inside the useful metric range."""
        valid = np.isfinite(depth_m)
        valid &= depth_m >= self.min_depth_m
        valid &= depth_m <= self.max_depth_m
        valid = valid.astype(np.uint8)

        # Dilate slightly to tolerate RGB-depth boundary misalignment.
        if self.depth_dilate_px > 0:
            k = max(1, int(self.depth_dilate_px))
            kernel = np.ones((k, k), dtype=np.uint8)
            valid = cv2.dilate(valid, kernel, iterations=1)

        return valid.astype(bool)

    def _extract_depth_rois(
        self, valid_depth_mask: np.ndarray, image_w: int, image_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """Extract connected valid-depth ROIs as ``(x, y, w, h)`` boxes."""
        mask_u8 = valid_depth_mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)

        rois: List[Tuple[int, int, int, int, int]] = []
        for label_idx in range(1, num_labels):  # label 0 is background
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < self.min_roi_area_px:
                continue

            pad = self.roi_padding_px
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(image_w, x + w + pad)
            y1 = min(image_h, y + h + pad)
            rois.append((x0, y0, x1 - x0, y1 - y0, area))

        rois.sort(key=lambda item: item[4], reverse=True)
        return [(x, y, w, h) for x, y, w, h, _ in rois[: self.max_rois]]

    def _run_sam_on_roi(
        self,
        rgb: np.ndarray,
        valid_depth_mask: Optional[np.ndarray],
        roi: Tuple[int, int, int, int],
    ) -> List[MaskProposal]:
        """Run SAM on one ROI and convert masks back to full-image coordinates."""
        x0, y0, rw, rh = roi
        x1 = x0 + rw
        y1 = y0 + rh

        rgb_roi = rgb[y0:y1, x0:x1]
        depth_roi_mask = None
        if valid_depth_mask is not None:
            depth_roi_mask = valid_depth_mask[y0:y1, x0:x1]

        rgb_for_sam, depth_for_sam, scale = self._resize_for_sam(rgb_roi, depth_roi_mask)
        results = self.generator.generate(rgb_for_sam)

        full_h, full_w = rgb.shape[:2]
        max_area = 0.60 * full_h * full_w
        proposals: List[MaskProposal] = []

        for item in results:
            small_mask = item.get("segmentation")
            if small_mask is None:
                continue

            mask_roi = small_mask.astype(bool)
            if scale < 0.999:
                mask_roi = self._resize_mask(mask_roi, rh, rw)

            if depth_roi_mask is not None:
                overlap = np.logical_and(mask_roi, depth_roi_mask).sum()
                overlap_ratio = float(overlap) / max(float(mask_roi.sum()), 1.0)
                if overlap_ratio < self.min_mask_depth_overlap:
                    continue
                mask_roi = np.logical_and(mask_roi, depth_roi_mask)

            full_mask = np.zeros((full_h, full_w), dtype=bool)
            full_mask[y0:y1, x0:x1] = mask_roi

            area = int(full_mask.sum())
            if area < self.min_area or area > max_area:
                continue

            bbox = self._bbox_from_mask(full_mask)
            if bbox is None:
                continue

            score = float(item.get("predicted_iou", item.get("stability_score", 0.5)))
            source = "sam_depth_roi" if self.use_depth_roi else "sam"
            proposals.append(MaskProposal(mask=full_mask, score=score, source=source, bbox=bbox))

        return proposals

    def _resize_for_sam(
        self, rgb: np.ndarray, mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """Resize a SAM input ROI while preserving aspect ratio."""
        if not self.resize_input:
            return rgb, mask, 1.0

        h, w = rgb.shape[:2]
        max_side = max(h, w)
        if max_side <= self.resize_max_side:
            return rgb, mask, 1.0

        scale = max(float(self.resize_max_side) / float(max_side), self.resize_min_scale)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_small = None
        if mask is not None:
            mask_small = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return rgb_small, mask_small, scale

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize a binary mask with nearest-neighbor interpolation."""
        return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask.astype(bool))
        if len(xs) == 0:
            return None
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max())
        y1 = int(ys.max())
        return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def make_segmenter(mode: str, params: dict):
    """Factory for segmenter backends."""
    mode = (mode or "dummy_grid").lower()
    if mode == "dummy_grid":
        return DummyGridSegmenter(params.get("dummy_box_fraction", 0.35))
    if mode == "sam":
        return SamAutomaticSegmenter(
            checkpoint=params.get("sam_checkpoint", ""),
            model_type=params.get("sam_model_type", "vit_b"),
            device=params.get("sam_device", "cuda"),
            min_area=params.get("sam_min_area", 800),
            max_masks=params.get("sam_max_masks", 25),
            use_depth_gate=params.get("use_depth_gate", True),
            min_depth_m=params.get("sam_min_depth_m", 0.25),
            max_depth_m=params.get("sam_max_depth_m", 3.0),
            min_near_pixels=params.get("sam_min_near_pixels", 1500),
            min_mask_depth_overlap=params.get("sam_min_mask_depth_overlap", 0.50),
            depth_dilate_px=params.get("sam_depth_dilate_px", 9),
            use_depth_roi=params.get("sam_use_depth_roi", True),
            roi_padding_px=params.get("sam_roi_padding_px", 24),
            min_roi_area_px=params.get("sam_min_roi_area_px", 2500),
            max_rois=params.get("sam_max_rois", 4),
            points_per_side=params.get("sam_points_per_side", 8),
            pred_iou_thresh=params.get("sam_pred_iou_thresh", 0.92),
            stability_score_thresh=params.get("sam_stability_score_thresh", 0.88),
            resize_input=params.get("sam_resize_input", True),
            resize_max_side=params.get("sam_resize_max_side", 512),
            resize_min_scale=params.get("sam_resize_min_scale", 0.25),
        )
    raise ValueError(f"Unknown segmenter_mode: {mode}")
