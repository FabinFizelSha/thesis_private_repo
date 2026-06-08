from dataclasses import dataclass
import json, re, requests
import numpy as np
from .utils import encode_image_to_base64

@dataclass
class VlmResult:
    label: str
    confidence: float
    risk_type: str
    risk_score: float
    reason: str
    source: str

class DisabledVlmClient:
    def query(self, crop_rgb: np.ndarray, context: str = "") -> VlmResult:
        return VlmResult("object_candidate", 0.0, "unknown", 0.0, "VLM disabled", "vlm_disabled")

class OpenAICompatibleQwenClient:
    def __init__(self, endpoint: str, model: str, api_key: str = "EMPTY", timeout_s: float = 20.0):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}
    def query(self, crop_rgb: np.ndarray, context: str = "") -> VlmResult:
        image_b64 = encode_image_to_base64(crop_rgb)
        prompt = (
            "You are labeling object crops for a mobile robot semantic map. Return ONLY valid JSON with keys: "
            "label, confidence, risk_type, risk_score, reason. Use a concise object label. If unsure, use object_candidate. "
            "risk_score must be 0.0 to 1.0. " + f"Context: {context}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]}],
            "temperature": 0.1,
            "max_tokens": 256,
        }
        try:
            r = requests.post(self.endpoint, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            data = self._extract_json(content)
            return VlmResult(
                str(data.get("label", "object_candidate")).strip().lower(),
                float(data.get("confidence", 0.0)),
                str(data.get("risk_type", "unknown")),
                float(data.get("risk_score", 0.0)),
                str(data.get("reason", "")),
                "qwen_vlm",
            )
        except Exception as exc:
            return VlmResult("object_candidate", 0.0, "unknown", 0.0, f"VLM call failed: {exc}", "qwen_vlm_failed")

def make_vlm_client(mode: str, params: dict):
    mode = (mode or "disabled").lower()
    if mode == "disabled":
        return DisabledVlmClient()
    if mode in ("qwen", "openai_compatible"):
        return OpenAICompatibleQwenClient(params.get("vlm_endpoint", "http://127.0.0.1:8005/v1/chat/completions"), params.get("vlm_model", "qwen2.5-vl"), params.get("vlm_api_key", "EMPTY"), params.get("vlm_timeout_s", 20.0))
    raise ValueError(f"Unknown vlm_mode: {mode}")
