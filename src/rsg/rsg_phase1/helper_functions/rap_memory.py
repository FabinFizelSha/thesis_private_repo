"""Lightweight RAP memory update hook for VLM-labelled unknown objects."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict

from rsg_phase1.helper_functions.json_utils import safe_json_dumps


class RapMemoryUpdater:
    """Append VLM-labelled unknown objects to a JSONL memory file.

    The baseline framework adds VLM-labelled unknown objects back into RAP so
    future retrieval can recognize them. In this ROS phase-1 implementation, the
    safe default is a lightweight JSONL update log. A real RAP database writer can
    replace this class without changing the ROS message flow.
    """

    def __init__(self, enabled: bool, output_path: str, min_confidence: float, logger: Any) -> None:
        self.enabled = enabled
        self.output_path = Path(output_path).expanduser().resolve()
        self.min_confidence = float(min_confidence)
        self.logger = logger
        self._lock = Lock()
        if self.enabled:
            self.logger.info(f"RAP memory update log enabled: {self.output_path}")
        else:
            self.logger.info("RAP memory update log disabled.")

    def update_from_vlm(self, vlm_result: Dict[str, Any], object_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Record a VLM result for later RAP memory ingestion."""
        if not self.enabled:
            return {"rap_memory_update": "disabled"}
        confidence = float(vlm_result.get("confidence", 0.0))
        if not bool(vlm_result.get("success", False)) or confidence < self.min_confidence:
            return {"rap_memory_update": "skipped", "reason": "low_confidence_or_failed", "confidence": confidence}

        record = {
            "candidate_id": object_metadata.get("candidate_id"),
            "source_frame_id": object_metadata.get("source_frame_id"),
            "predicted_label": vlm_result.get("label"),
            "confidence": confidence,
            "centroid_3d": object_metadata.get("centroid_3d"),
            "bbox_2d": object_metadata.get("bbox_2d"),
            "bbox_3d_min": object_metadata.get("bbox_3d_min"),
            "bbox_3d_max": object_metadata.get("bbox_3d_max"),
            "backend": vlm_result.get("backend"),
            "model": vlm_result.get("model"),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as stream:
                stream.write(safe_json_dumps(record) + "\n")
        return {"rap_memory_update": "logged", "path": str(self.output_path), "label": vlm_result.get("label"), "confidence": confidence}
