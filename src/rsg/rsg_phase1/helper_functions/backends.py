"""SAM/RAP/VLM backend adapters for the Phase 1 object-detection node.

Two profiles are supported by configuration:

``dummy``
    Fully deterministic, dependency-light path for development PCs.

``real_orin`` / ``real``
    Deployment path intended for Jetson Orin/Thor.  It mirrors the baseline
    framework: SAM produces masks, VisualRAP performs CLIP + Chroma retrieval,
    and an OpenAI-compatible VLM endpoint identifies persistent unknown tracks.

The real adapters are optional at import time.  Missing heavy dependencies do
not break the ROS package build.  They are loaded only when the real profile is
selected.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SamMask:
    """Raw mask candidate produced by a SAM-like backend."""

    mask_id: str
    mask: np.ndarray
    bbox_2d: list[int]
    area_px: int
    crop: Optional[Any] = None
    score: float = 0.0
    metadata: Dict[str, Any] | None = None


@dataclass
class RapMatch:
    """RAP retrieval result for one object cutout/mask."""

    label: str
    confidence: float
    is_known: bool
    distance: float
    metadata: Dict[str, Any]


class DummySamBackend:
    """Deterministic segmentation backend for development PCs."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def segment(self, rgb: np.ndarray) -> List[SamMask]:
        height, width = rgb.shape[:2]
        masks: List[SamMask] = []
        count = min(int(self.config.sam_dummy_num_masks), int(self.config.sam_max_masks))
        if count <= 0:
            return masks

        for idx in range(count):
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
            if area >= int(self.config.sam_min_mask_pixels):
                masks.append(
                    SamMask(
                        mask_id=f"mask_{idx:03d}",
                        mask=mask,
                        bbox_2d=[x0, y0, x1 - x0, y1 - y0],
                        area_px=area,
                        crop=rgb[y0:y1, x0:x1].copy(),
                        score=1.0,
                        metadata={"backend": "dummy"},
                    )
                )
        return masks


class RealSamBackend:
    """Segment Anything backend compatible with the baseline SamSegmenter.

    The adapter first tries to import the baseline ``scripts.SamSegmenter`` if
    available.  If that module is not on PYTHONPATH, it falls back to a native
    implementation using ``segment_anything`` directly.  This makes deployment
    possible either with the original baseline repository installed or with only
    the required Python dependencies installed on the Jetson.
    """

    CHECKPOINT_URLS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    CHECKPOINT_NAMES = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    def __init__(self, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.backend_name = "real_sam"
        self._baseline_segmenter = None
        self._mask_generator = None
        self._init_backend()

    def _init_backend(self) -> None:
        # Try baseline class first.  It uses SAM vit_h and returns dicts with
        # crop, bbox, mask, and score.
        for module_name in ("scripts.SamSegmenter", "SamSegmenter"):
            try:
                module = __import__(module_name, fromlist=["SamSegmenter"])
                SamSegmenter = getattr(module, "SamSegmenter")
                kwargs = {
                    "checkpoint_path": self.config.sam_checkpoint_path or None,
                    "device": self.config.sam_device or None,
                    "mask_threshold": self.config.sam_mask_threshold,
                    "padding": self.config.sam_padding,
                    "min_segment_pixels": self.config.sam_min_mask_pixels,
                    "points_per_side": self.config.sam_points_per_side,
                    "pred_iou_thresh": self.config.sam_pred_iou_thresh,
                }
                self._baseline_segmenter = SamSegmenter(**kwargs)
                self.backend_name = f"baseline:{module_name}"
                self.logger.info(f"Using baseline SamSegmenter backend from {module_name}.")
                return
            except Exception as exc:
                last_exc = exc

        # Fall back to direct segment-anything implementation.
        try:
            import torch  # type: ignore
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Real SAM backend requested but neither baseline SamSegmenter nor "
                "segment_anything is available. Install the baseline repo on PYTHONPATH "
                "or install segment-anything + torch."
            ) from exc

        model_type = str(self.config.sam_model_type or "vit_h")
        checkpoint_path = self.config.sam_checkpoint_path or self._resolve_checkpoint(model_type)
        device = self.config.sam_device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=int(self.config.sam_points_per_side),
            pred_iou_thresh=float(self.config.sam_pred_iou_thresh),
            stability_score_thresh=float(self.config.sam_mask_threshold),
            min_mask_region_area=int(self.config.sam_min_mask_pixels),
        )
        self.backend_name = f"segment_anything:{model_type}"
        self.logger.info(f"Using native Segment Anything backend: model_type={model_type}, device={device}.")

    def _resolve_checkpoint(self, model_type: str) -> str:
        cache_dir = Path(str(self.config.sam_checkpoint_cache_dir)).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = self.CHECKPOINT_NAMES.get(model_type, self.CHECKPOINT_NAMES["vit_h"])
        checkpoint_path = cache_dir / checkpoint_name
        if checkpoint_path.exists():
            return str(checkpoint_path)
        if not bool(self.config.sam_auto_download):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}. Set phase1.sam.checkpoint_path "
                "or enable phase1.sam.auto_download."
            )
        url = self.CHECKPOINT_URLS.get(model_type, self.CHECKPOINT_URLS["vit_h"])
        self.logger.warn(f"Downloading SAM checkpoint {model_type} to {checkpoint_path}. This can take several minutes.")
        urllib.request.urlretrieve(url, checkpoint_path)
        return str(checkpoint_path)

    def segment(self, rgb: np.ndarray) -> List[SamMask]:
        if self._baseline_segmenter is not None:
            try:
                from PIL import Image  # type: ignore
                detections = self._baseline_segmenter.segment(Image.fromarray(rgb))
            except Exception as exc:
                raise RuntimeError(f"Baseline SamSegmenter failed during segment(): {exc}") from exc
            return self._convert_baseline_detections(rgb, detections)

        if self._mask_generator is None:
            return []
        mask_dicts = self._mask_generator.generate(rgb)
        masks: List[SamMask] = []
        h, w = rgb.shape[:2]
        for idx, item in enumerate(mask_dicts[: int(self.config.sam_max_masks)]):
            seg = np.asarray(item.get("segmentation"), dtype=bool)
            bbox = item.get("bbox", [0, 0, 0, 0])
            x, y, bw, bh = [int(v) for v in bbox]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x0 + max(0, bw)), min(h, y0 + max(0, bh))
            area = int(np.count_nonzero(seg))
            if area < int(self.config.sam_min_mask_pixels):
                continue
            masks.append(
                SamMask(
                    mask_id=f"mask_{idx:03d}",
                    mask=seg,
                    bbox_2d=[x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                    area_px=area,
                    crop=rgb[y0:y1, x0:x1].copy() if x1 > x0 and y1 > y0 else None,
                    score=float(item.get("stability_score", item.get("predicted_iou", 0.0)) or 0.0),
                    metadata={"backend": self.backend_name},
                )
            )
        return masks

    @staticmethod
    def _convert_baseline_detections(rgb: np.ndarray, detections: List[Dict[str, Any]]) -> List[SamMask]:
        h, w = rgb.shape[:2]
        masks: List[SamMask] = []
        for idx, det in enumerate(detections):
            raw_bbox = det.get("bbox", [0, 0, 0, 0])
            if len(raw_bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in raw_bbox]
            # Baseline SamSegmenter returns padded bbox as (x1, y1, x2, y2).
            x0, y0 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask = np.asarray(det.get("mask"), dtype=bool)
            area = int(np.count_nonzero(mask)) if mask.size else int(max(0, x2 - x0) * max(0, y2 - y0))
            masks.append(
                SamMask(
                    mask_id=f"mask_{idx:03d}",
                    mask=mask,
                    bbox_2d=[x0, y0, max(0, x2 - x0), max(0, y2 - y0)],
                    area_px=area,
                    crop=det.get("crop", None),
                    score=float(det.get("score", 0.0) or 0.0),
                    metadata={"backend": "baseline_sam"},
                )
            )
        return masks


class DummyRapBackend:
    """Deterministic RAP-like retrieval backend."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.known_labels = config.rap_dummy_known_labels or ["dummy_object"]

    def classify(self, rgb: np.ndarray, mask: SamMask, index: int) -> RapMatch:
        unknown_every = max(1, int(self.config.rap_dummy_unknown_every_n))
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
        confidence = max(float(self.config.rap_confidence_threshold) + 0.1, 0.5)
        return RapMatch(
            label=label,
            confidence=float(confidence),
            is_known=True,
            distance=max(0.0, 1.0 - float(confidence)),
            metadata={"backend": "dummy", "decision": "known_label"},
        )


class RealRapBackend:
    """Baseline-style VisualRAP adapter using CLIP + ChromaDB."""

    def __init__(self, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.backend_name = "real_rap"
        self._visual_rap = None
        self._init_backend()

    def _init_backend(self) -> None:
        # Prefer the baseline VisualRAP class if the original repo is installed.
        for module_name in ("scripts.VisualRAP", "VisualRAP"):
            try:
                module = __import__(module_name, fromlist=["VisualRAP"])
                VisualRAP = getattr(module, "VisualRAP")
                self._visual_rap = VisualRAP(
                    storage_path=self.config.rap_storage_path,
                    model_name=self.config.rap_model_name,
                    device=self.config.rap_device or None,
                    chroma_host=self.config.rap_chroma_host,
                    chroma_port=int(self.config.rap_chroma_port),
                    auto_start_server=bool(self.config.rap_auto_start_server),
                )
                self.backend_name = f"baseline:{module_name}"
                self.logger.info(f"Using baseline VisualRAP backend from {module_name}.")
                return
            except Exception as exc:
                last_exc = exc

        # Native implementation compatible with the baseline API.
        self._visual_rap = _NativeVisualRAP(
            storage_path=self.config.rap_storage_path,
            model_name=self.config.rap_model_name,
            device=self.config.rap_device or None,
            chroma_host=self.config.rap_chroma_host,
            chroma_port=int(self.config.rap_chroma_port),
            collection_name=self.config.rap_collection_name,
            auto_start_server=bool(self.config.rap_auto_start_server),
            logger=self.logger,
        )
        self.backend_name = "native_visual_rap"
        self.logger.info("Using native VisualRAP-compatible backend.")

    def classify(self, rgb: np.ndarray, mask: SamMask, index: int) -> RapMatch:
        crop = self._get_crop(rgb, mask)
        if crop is None:
            return RapMatch("unknown_object", 0.0, False, 1.0, {"backend": self.backend_name, "reason": "empty_crop"})
        try:
            label, distance = self._visual_rap.query(image=crop, threshold=float(self.config.rap_distance_threshold))
            label = str(label).strip()
            distance = float(distance)
        except Exception as exc:
            return RapMatch("unknown_object", 0.0, False, 1.0, {"backend": self.backend_name, "error": str(exc)})
        is_known = bool(label and label.lower() not in {"unknown", "unknown_object", "unclear", "unsure", "n/a"} and distance <= float(self.config.rap_distance_threshold))
        confidence = max(0.0, min(1.0, 1.0 - distance))
        if not is_known:
            label = "unknown_object"
            confidence = 0.0
        return RapMatch(label, confidence, is_known, distance, {"backend": self.backend_name, "distance": distance})

    def add_image(self, image: Any, label: str) -> None:
        if hasattr(self._visual_rap, "add_image"):
            self._visual_rap.add_image(image, label)

    @staticmethod
    def _get_crop(rgb: np.ndarray, mask: SamMask) -> Any:
        if mask.crop is not None:
            return mask.crop
        x, y, w, h = [int(v) for v in mask.bbox_2d]
        if w <= 0 or h <= 0:
            return None
        crop_np = rgb[y:y + h, x:x + w]
        if crop_np.size == 0:
            return None
        try:
            from PIL import Image  # type: ignore
            return Image.fromarray(crop_np)
        except Exception:
            return crop_np


class _NativeVisualRAP:
    """Small local implementation of baseline VisualRAP.

    It uses CLIP from ``transformers`` and ChromaDB's HTTP client.  Training
    images must be loaded once into the ``visual_rag`` collection.  The helper
    script ``rsg_load_rap_memory`` included in this package does that.
    """

    def __init__(
        self,
        storage_path: str,
        model_name: str,
        device: Optional[str],
        chroma_host: str,
        chroma_port: int,
        collection_name: str,
        auto_start_server: bool,
        logger: Any,
    ) -> None:
        try:
            import torch  # type: ignore
            from chromadb import HttpClient  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Real RAP backend requires torch, transformers, and chromadb. "
                "Install them or put the baseline VisualRAP module on PYTHONPATH."
            ) from exc

        self.storage_path = str(Path(storage_path).expanduser().resolve())
        self.chroma_host = chroma_host
        self.chroma_port = int(chroma_port)
        self.collection_name = collection_name
        self.logger = logger
        self.server_process = None
        self._owns_server = False
        if auto_start_server and not self._is_server_running():
            self._start_chroma_server()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.client = HttpClient(host=chroma_host, port=self.chroma_port)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _is_server_running(self) -> bool:
        try:
            import requests  # type: ignore
            response = requests.get(f"http://{self.chroma_host}:{self.chroma_port}/api/v1/heartbeat", timeout=1)
            return response.status_code == 200
        except Exception:
            return False

    def _start_chroma_server(self) -> None:
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Starting ChromaDB server on port {self.chroma_port}; storage={self.storage_path}")
        self.server_process = subprocess.Popen(
            ["chroma", "run", "--port", str(self.chroma_port), "--path", self.storage_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._owns_server = True
        for _ in range(10):
            if self._is_server_running():
                return
            time.sleep(1.0)
        raise RuntimeError(f"Failed to start ChromaDB server on port {self.chroma_port}")

    def _to_pil(self, image: Any) -> Any:
        if hasattr(image, "mode"):
            return image
        from PIL import Image  # type: ignore
        return Image.fromarray(np.asarray(image, dtype=np.uint8))

    def embed_image(self, image: Any) -> np.ndarray:
        import torch  # type: ignore
        pil_image = self._to_pil(image)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.vision_model(**inputs)
            img_emb = self.model.visual_projection(image_features.pooler_output)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb.cpu().numpy().astype("float32")[0]

    def add_image(self, image: Any, label: str) -> None:
        import uuid
        emb = self.embed_image(image)
        self.collection.add(embeddings=[emb.tolist()], documents=[str(label).strip()], ids=[str(uuid.uuid4())])

    def query(self, image: Any, top_k: int = 3, threshold: float = 0.3) -> Tuple[str, float]:
        emb = self.embed_image(image)
        results = self.collection.query(query_embeddings=[emb.tolist()], n_results=int(top_k))
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "unknown", 1.0
        best_distance = float(results.get("distances", [[1.0]])[0][0])
        best_label = str(docs[0]).strip()
        if best_distance > float(threshold):
            return "unknown", best_distance
        return best_label, best_distance


class DummyVlmBackend:
    """Dummy unknown-object identification backend for non-VLM machines."""

    def __init__(self, config: Any) -> None:
        self.config = config
        # Support the baseline environment convention while keeping YAML as the
        # primary source of truth.  If OPENAI_BASE_URL is set, treat it as the
        # OpenAI-compatible /v1 base URL and append /chat/completions.
        env_base = os.environ.get("OPENAI_BASE_URL", "").strip()
        self.endpoint = (env_base.rstrip("/") + "/chat/completions") if env_base else str(config.vlm_endpoint)
        self.model = os.environ.get("MODEL", str(config.vlm_model))

    def identify(self, rgb_crop: Optional[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if float(self.config.vlm_dummy_delay_sec) > 0.0:
            time.sleep(float(self.config.vlm_dummy_delay_sec))
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


class OpenAICompatibleVlmBackend:
    """OpenAI-compatible HTTP adapter for local Qwen-style VLM servers."""

    def __init__(self, config: Any) -> None:
        self.config = config
        # Support the baseline environment convention while keeping YAML as the
        # primary source of truth.  If OPENAI_BASE_URL is set, treat it as the
        # OpenAI-compatible /v1 base URL and append /chat/completions.
        env_base = os.environ.get("OPENAI_BASE_URL", "").strip()
        self.endpoint = (env_base.rstrip("/") + "/chat/completions") if env_base else str(config.vlm_endpoint)
        self.model = os.environ.get("MODEL", str(config.vlm_model))

    def identify(self, rgb_crop: Optional[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if rgb_crop is None or rgb_crop.size == 0:
            return {"success": False, "label": "unknown_object", "confidence": 0.0, "backend": self.config.vlm_mode, "model": self.config.vlm_model, "raw_response": "empty_crop"}
        try:
            image_url = self._crop_to_data_url(rgb_crop)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.vlm_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": int(self.config.vlm_max_tokens),
                "temperature": float(self.config.vlm_temperature),
            }
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            api_key = str(self.config.vlm_api_key or "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=float(self.config.vlm_timeout_sec)) as response:
                body = response.read().decode("utf-8")
            result = json.loads(body)
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "unknown_object")
            label = self._clean_label(text)
            return {
                "success": label not in {"", "unknown", "unknown_object", "unclear", "none"},
                "label": label or "unknown_object",
                "confidence": float(self.config.vlm_confidence),
                "backend": self.config.vlm_mode,
                "model": self.model,
                "raw_response": text,
            }
        except Exception as exc:
            return {
                "success": False,
                "label": "unknown_object",
                "confidence": 0.0,
                "backend": self.config.vlm_mode,
                "model": self.model,
                "raw_response": str(exc),
            }

    def _crop_to_data_url(self, rgb_crop: np.ndarray) -> str:
        try:
            import cv2  # type: ignore
            bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.vlm_jpeg_quality)])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            from PIL import Image  # type: ignore
            stream = BytesIO()
            Image.fromarray(rgb_crop).save(stream, format="PNG")
            b64 = base64.b64encode(stream.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"

    @staticmethod
    def _clean_label(text: str) -> str:
        label = str(text).strip().splitlines()[0].strip().strip(".:- ").strip('"\'')
        if not label:
            return "unknown_object"
        # Keep labels simple for RAP memory and downstream metadata.
        return label.replace(" ", "_").lower()[:80]


def _allow_fallback(config: Any) -> bool:
    return bool(getattr(config, "allow_dummy_fallback", False))


def make_sam_backend(config: Any, logger: Any) -> Any:
    backend = str(config.sam_backend).lower()
    if backend in {"real", "baseline", "sam", "segment_anything", "orin"}:
        try:
            return RealSamBackend(config, logger)
        except Exception as exc:
            if _allow_fallback(config):
                logger.error(f"Real SAM backend failed; falling back to dummy SAM because allow_dummy_fallback=true. Reason: {exc}")
                return DummySamBackend(config)
            raise
    return DummySamBackend(config)


def make_rap_backend(config: Any, logger: Any) -> Any:
    backend = str(config.rap_backend).lower()
    if backend in {"real", "baseline", "visual_rap", "vrap", "orin"}:
        try:
            return RealRapBackend(config, logger)
        except Exception as exc:
            if _allow_fallback(config):
                logger.error(f"Real RAP backend failed; falling back to dummy RAP because allow_dummy_fallback=true. Reason: {exc}")
                return DummyRapBackend(config)
            raise
    return DummyRapBackend(config)


def make_vlm_backend(config: Any) -> Any:
    mode = str(config.vlm_mode).lower()
    if mode in {"openai", "openai_vision", "qwen", "qwen_http", "real", "orin"}:
        return OpenAICompatibleVlmBackend(config)
    return DummyVlmBackend(config)
