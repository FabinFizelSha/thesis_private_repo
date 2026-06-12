"""Build Hydra-friendly semantic and instance pixel label images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class ClassifiedMask:
    """One SAM mask after RAP classification."""

    mask_id: str
    mask: np.ndarray
    label: str
    label_id: int
    instance_id: int
    confidence: float
    status: str
    candidate_id: str
    metadata: Dict[str, Any]


class LabelMapBuilder:
    """Convert classified masks into semantic and instance label images."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def build(self, image_shape: Tuple[int, int], masks: List[ClassifiedMask]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return semantic map, instance map, label table, objects, and unknown objects."""
        height, width = image_shape
        semantic = np.zeros((height, width), dtype=np.uint16)
        instance = np.zeros((height, width), dtype=np.uint16)

        # Draw larger masks first and smaller masks later so smaller foreground
        # instances are not hidden by large background-like masks.
        masks_sorted = sorted(masks, key=lambda item: int(np.count_nonzero(item.mask)), reverse=True)
        objects: List[Dict[str, Any]] = []
        unknowns: List[Dict[str, Any]] = []
        label_table: Dict[str, Any] = {"0": "background"}

        for item in masks_sorted:
            if item.mask.shape != semantic.shape:
                continue
            semantic[item.mask] = np.uint16(item.label_id)
            instance[item.mask] = np.uint16(item.instance_id)
            label_table[str(item.label_id)] = item.label

            obj = {
                "candidate_id": item.candidate_id,
                "mask_id": item.mask_id,
                "label": item.label,
                "label_id": int(item.label_id),
                "instance_id": int(item.instance_id),
                "confidence": float(item.confidence),
                "status": item.status,
                **item.metadata,
            }
            if item.status.startswith("unknown"):
                if self.config.include_unknown_objects:
                    unknowns.append(obj)
                    objects.append(obj)
            else:
                if self.config.include_known_objects:
                    objects.append(obj)

        return semantic, instance, label_table, objects, unknowns
