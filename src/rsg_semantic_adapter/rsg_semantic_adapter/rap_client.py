from dataclasses import dataclass
import requests
import numpy as np
from .utils import encode_image_to_base64

@dataclass
class RapResult:
    label: str
    confidence: float
    source: str

class DisabledRapClient:
    def query(self, crop_rgb: np.ndarray) -> RapResult:
        return RapResult(label="object_candidate", confidence=0.0, source="rap_disabled")

class HttpRapClient:
    def __init__(self, endpoint: str, top_k: int = 5, timeout_s: float = 2.0):
        self.endpoint = endpoint
        self.top_k = int(top_k)
        self.timeout_s = float(timeout_s)
    def query(self, crop_rgb: np.ndarray) -> RapResult:
        payload = {"image_base64": encode_image_to_base64(crop_rgb), "top_k": self.top_k}
        try:
            r = requests.post(self.endpoint, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            label = data.get("label", data.get("top_label", "object_candidate"))
            conf = float(data.get("confidence", data.get("score", 0.0)))
            return RapResult(label=str(label).strip().lower(), confidence=conf, source="rap_http")
        except Exception:
            return RapResult(label="object_candidate", confidence=0.0, source="rap_http_failed")

def make_rap_client(mode: str, params: dict):
    mode = (mode or "disabled").lower()
    if mode == "disabled":
        return DisabledRapClient()
    if mode == "http":
        return HttpRapClient(params.get("rap_http_endpoint", "http://127.0.0.1:8010/query"), params.get("rap_top_k", 5), params.get("rap_timeout_s", 2.0))
    raise ValueError(f"Unknown rap_mode: {mode}")
