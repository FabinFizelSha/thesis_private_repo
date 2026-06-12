"""Lightweight SAM/RAP/VLM backend adapters for Phase 1.

The dummy backends make the ROS 2 data flow testable on machines that cannot
run SAM, RAP, or a VLM. The baseline/custom hooks are deliberately isolated so
real model integrations can be added without changing the coordinator node.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SamMask:
    """Raw mask candidate produced by a SAM-like backend."""

    mask_id: str
    mask: np.ndarray
    bbox_2d: list[int]
    area_px: int


@dataclass
class RapMatch:
    """RAP retrieval result for one object cutout/mask."""

    label: str
    confidence: float
    is_known: bool
    distance: float
    metadata: Dict[str, Any]


class DummySamBackend:
    """Deterministic segmentation backend for development PCs.

    It creates a small number of rectangular masks in the image. This does not
    represent real segmentation quality, but it lets the complete ROS pipeline,
    label-map generation, Hydra forwarding, timing, and VLM dummy path be tested.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def segment(self, rgb: np.ndarray) -> List[SamMask]:
        height, width = rgb.shape[:2]
        masks: List[SamMask] = []
        count = min(self.config.sam_dummy_num_masks, self.config.sam_max_masks)
        if count <= 0:
            return masks

        for idx in range(count):
            # Distribute deterministic boxes across the image while keeping them
            # inside the frame. The first object is central for simple tests.
            box_w = max(20, width // 5)
            box_h = max(20, height // 5)
            if idx == 0:
                x0 = max(0, width // 2 - box_w // 2)
                y0 = max(0, height // 2 - box_h // 2)
            else:
                x0 = int((idx + 1) * width / (count + 2) - box_w / 2)
                y0 = int((idx + 1) * height / (count + 2) - box_h / 2)
                x0 = min(max(0, x0), max(0, width - box_w))
                y0 = min(max(0, y0), max(0, height - box_h))
            x1 = min(width, x0 + box_w)
            y1 = min(height, y0 + box_h)
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
            area = int(np.count_nonzero(mask))
            if area >= self.config.sam_min_mask_pixels:
                masks.append(SamMask(mask_id=f"mask_{idx:03d}", mask=mask, bbox_2d=[x0, y0, x1 - x0, y1 - y0], area_px=area))
        return masks


class BaselineSamBackend(DummySamBackend):
    """Placeholder adapter for the original baseline SAM code.

    The original framework can be connected here when its Python modules are on
    PYTHONPATH. Until then this class falls back to the dummy implementation so
    the ROS package remains buildable and testable.
    """

    def __init__(self, config: Any, logger: Any) -> None:
        super().__init__(config)
        self.logger = logger
        self.available = False
        try:
            # The exact baseline module name depends on how the original repo is
            # installed. Keep this import optional to avoid breaking local tests.
            import SamSegmenter  # type: ignore  # noqa: F401
            self.available = True
        except Exception as exc:
            self.logger.warn(f"Baseline SAM backend not available; using dummy SAM. Reason: {exc}")


class DummyRapBackend:
    """Deterministic RAP-like retrieval backend.

    It marks every Nth mask as unknown and labels the rest using configured dummy
    class names. This mirrors the baseline idea of known retrieval vs unknown
    learning without requiring the RAP model during development.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.known_labels = config.rap_dummy_known_labels or ["dummy_object"]

    def classify(self, rgb: np.ndarray, mask: SamMask, index: int) -> RapMatch:
        unknown_every = max(1, self.config.rap_dummy_unknown_every_n)
        is_unknown = (index % unknown_every) == 0
        if is_unknown or not self.config.rap_enabled:
            return RapMatch(
                label="unknown_object",
                confidence=0.0,
                is_known=False,
                distance=1.0,
                metadata={"backend": "dummy", "decision": "forced_unknown"},
            )
        label = self.known_labels[index % len(self.known_labels)]
        confidence = max(self.config.rap_confidence_threshold + 0.1, 0.5)
        return RapMatch(
            label=label,
            confidence=float(confidence),
            is_known=True,
            distance=max(0.0, 1.0 - float(confidence)),
            metadata={"backend": "dummy", "decision": "known_label"},
        )


class BaselineRapBackend(DummyRapBackend):
    """Placeholder adapter for the baseline VisualRAP object retrieval path."""

    def __init__(self, config: Any, logger: Any) -> None:
        super().__init__(config)
        self.logger = logger
        self.available = False
        try:
            import LoadDataToRAP  # type: ignore  # noqa: F401
            self.available = True
        except Exception as exc:
            self.logger.warn(f"Baseline RAP backend not available; using dummy RAP. Reason: {exc}")


class DummyVlmBackend:
    """Dummy unknown-object identification backend for non-VLM machines."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def identify(self, rgb_crop: Optional[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.vlm_dummy_delay_sec > 0.0:
            time.sleep(self.config.vlm_dummy_delay_sec)
        candidate_id = str(metadata.get("candidate_id", "unknown"))
        suffix = candidate_id.split("_")[-1] if "_" in candidate_id else candidate_id
        return {
            "success": True,
            "label": f"{self.config.vlm_dummy_label_prefix}_{suffix}",
            "confidence": float(self.config.vlm_confidence),
            "backend": "dummy",
            "model": "dummy",
            "raw_response": "dummy_vlm_path",
        }


class OpenAIVisionVlmBackend:
    """Small OpenAI-compatible HTTP adapter for Qwen-style VLM servers.

    This supports local servers such as vLLM, LMDeploy, or any service exposing
    an OpenAI-compatible chat-completions endpoint. It is intentionally optional
    and uses only the Python standard library.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def identify(self, rgb_crop: Optional[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if rgb_crop is None or rgb_crop.size == 0:
            return {"success": False, "label": "unknown_object", "confidence": 0.0, "backend": "openai_vision", "model": self.config.vlm_model, "raw_response": "empty_crop"}

        try:
            image_url = self._crop_to_data_url(rgb_crop)
            payload = {
                "model": self.config.vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.vlm_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": 32,
                "temperature": 0.0,
            }
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                self.config.vlm_endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.config.vlm_timeout_sec) as response:
                body = response.read().decode("utf-8")
            result = json.loads(body)
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "unknown_object")
            label = self._clean_label(text)
            return {
                "success": True,
                "label": label,
                "confidence": float(self.config.vlm_confidence),
                "backend": "openai_vision",
                "model": self.config.vlm_model,
                "raw_response": text,
            }
        except Exception as exc:
            return {
                "success": False,
                "label": "unknown_object",
                "confidence": 0.0,
                "backend": "openai_vision",
                "model": self.config.vlm_model,
                "raw_response": str(exc),
            }

    @staticmethod
    def _clean_label(text: str) -> str:
        label = str(text).strip().splitlines()[0].strip().strip(".:- ")
        if not label:
            return "unknown_object"
        return label.replace(" ", "_").lower()[:80]

    @staticmethod
    def _crop_to_data_url(rgb_crop: np.ndarray) -> str:
        try:
            import cv2
            bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            # Fallback to PNG via PIL if available.
            from io import BytesIO
            from PIL import Image  # type: ignore
            stream = BytesIO()
            Image.fromarray(rgb_crop).save(stream, format="PNG")
            b64 = base64.b64encode(stream.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"


def make_sam_backend(config: Any, logger: Any) -> Any:
    if config.sam_backend.lower() == "baseline":
        return BaselineSamBackend(config, logger)
    return DummySamBackend(config)


def make_rap_backend(config: Any, logger: Any) -> Any:
    if config.rap_backend.lower() == "baseline":
        return BaselineRapBackend(config, logger)
    return DummyRapBackend(config)


def make_vlm_backend(config: Any) -> Any:
    mode = config.vlm_mode.lower()
    if mode in {"openai", "openai_vision", "qwen", "qwen_http", "real"}:
        return OpenAIVisionVlmBackend(config)
    return DummyVlmBackend(config)
