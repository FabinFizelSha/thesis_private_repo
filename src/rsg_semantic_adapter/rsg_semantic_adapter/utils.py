import base64
import io
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image as PILImage


def encode_image_to_base64(rgb: np.ndarray) -> str:
    pil_img = PILImage.fromarray(rgb.astype(np.uint8), mode="RGB")
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def crop_from_bbox(rgb: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 8) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = rgb.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    return rgb[y0:y1, x0:x1].copy()


def create_color_overlay(rgb: np.ndarray, semantic: np.ndarray, label_colors: Dict[int, Tuple[int, int, int]], alpha: float = 0.55) -> np.ndarray:
    overlay = rgb.copy().astype(np.float32)
    color_img = np.zeros_like(overlay)
    for label_id, color in label_colors.items():
        color_img[semantic == int(label_id)] = np.array(color, dtype=np.float32)
    nonzero = semantic != 0
    overlay[nonzero] = (1.0 - alpha) * overlay[nonzero] + alpha * color_img[nonzero]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def depth_to_meters(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    if depth.dtype == np.uint16 or depth.dtype == np.uint32:
        return depth.astype(np.float32) * float(depth_scale)
    return depth.astype(np.float32)


def centroid_from_mask_depth(mask: np.ndarray, depth_m: np.ndarray, k: np.ndarray, max_samples: int = 5000) -> Optional[Tuple[float, float, float]]:
    if k is None:
        return None
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 0.05)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if len(xs) == 0:
        return None
    if len(xs) > max_samples:
        idx = np.random.choice(len(xs), size=max_samples, replace=False)
        xs, ys, z = xs[idx], ys[idx], z[idx]
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return (float(np.median(x)), float(np.median(y)), float(np.median(z)))


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


def transform_point(point, transform_stamped):
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    r = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    p = np.array(point, dtype=np.float64)
    out = r @ p + np.array([t.x, t.y, t.z], dtype=np.float64)
    return (float(out[0]), float(out[1]), float(out[2]))
