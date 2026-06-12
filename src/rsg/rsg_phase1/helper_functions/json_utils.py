"""JSON helpers for ROS string metadata fields."""

from __future__ import annotations

import json
from typing import Any


def safe_json_dumps(data: Any) -> str:
    """Serialize metadata to compact JSON without raising into the ROS hot path."""
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    except Exception as exc:  # pragma: no cover - defensive
        return json.dumps({"serialization_error": str(exc)}, separators=(",", ":"))


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON metadata. Return ``default`` if empty or invalid."""
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _json_default(obj: Any) -> Any:
    """Convert common NumPy values to JSON-compatible Python values."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
