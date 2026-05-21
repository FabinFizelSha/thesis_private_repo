import base64
import os
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import HTTPException

from risk_scene_graph_core.config.parameters import tmp_dir


def ensure_dirs() -> None:
    """Create required temporary output directories.

    Args:
        None

    Returns:
        None
    """
    os.makedirs(os.path.join(tmp_dir, "inVLM"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "outVLM"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "outPNG"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "completeGraph"), exist_ok=True)

def b64_to_ndarray(b64_str: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Decode a base64-encoded image into a NumPy array.

    Args:
        b64_str: Base64-encoded image string.
        flags: OpenCV imdecode flags.

    Returns:
        Decoded image as a NumPy array.
    """
    try:
        raw = base64.b64decode(b64_str)
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, flags)
        if img is None:
            raise ValueError("Decoded image is None")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


def fill_invalid_depth_nearest(depth_img: np.ndarray) -> np.ndarray:
    """Fill non-positive/NaN depth values with nearest valid neighbor.

    Args:
        depth_img: Depth image as a 2D NumPy array (meters).

    Returns:
        Depth image with invalid values replaced by nearest valid depth.
    """
    if depth_img.ndim == 3 and depth_img.shape[2] == 1:
        depth_img = depth_img[:, :, 0]
    if depth_img.ndim != 2:
        raise ValueError("Depth image must be single-channel")

    invalid_mask = (depth_img <= 0) | np.isnan(depth_img)
    if not invalid_mask.any():
        return depth_img

    valid_mask = ~invalid_mask
    if not valid_mask.any():
        return np.ones_like(depth_img, dtype=depth_img.dtype)

    src = invalid_mask.astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        src,
        distanceType=cv2.DIST_L2,
        maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    zero_coords = np.column_stack(np.where(src == 0))
    zero_labels = labels[src == 0]
    max_label = int(labels.max())
    label_to_y = np.zeros(max_label + 1, dtype=np.int32)
    label_to_x = np.zeros(max_label + 1, dtype=np.int32)
    label_to_y[zero_labels] = zero_coords[:, 0]
    label_to_x[zero_labels] = zero_coords[:, 1]

    nearest_y = label_to_y[labels]
    nearest_x = label_to_x[labels]

    filled = depth_img.copy()
    filled[invalid_mask] = depth_img[nearest_y[invalid_mask], nearest_x[invalid_mask]]
    return filled


def convert_numpy(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python types.

    Args:
        obj: Arbitrary object potentially containing NumPy types.

    Returns:
        Object with NumPy scalars/arrays converted to native types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def graph_to_json(G) -> Dict[str, Any]:
    """Convert a NetworkX graph into a JSON-serializable dict.

    Args:
        G: NetworkX graph instance.

    Returns:
        Dictionary with "nodes" and "edges" lists.
    """
    nodes = []
    for nid, attrs in G.nodes(data=True):
        node = {"id": int(nid)}
        for k, v in attrs.items():
            node[k] = convert_numpy(v)
        layer = str(node.get("layer", "")).upper()
        if layer != "ROBOT":
            if "semantic_description" not in node:
                node["semantic_description"] = ""
            if "semantic_pose" not in node:
                node["semantic_pose"] = ""
        nodes.append(node)

    edges = []
    for u, v, attrs in G.edges(data=True):
        edge = {"from": int(u), "to": int(v)}
        if "rel" in attrs:
            edge["rel"] = attrs["rel"]
        if "distance" in attrs:
            edge["distance"] = convert_numpy(attrs["distance"])
        if "appearance" in attrs:
            edge["appearance"] = convert_numpy(attrs["appearance"])
        edges.append(edge)
    return {"nodes": nodes, "edges": edges}


def save_masks_image(rgb_img: np.ndarray, detections: list, detected_objects: list, timestamp: str, masks_all_path: str, masks_detected_path: str) -> None:
    """Save visualization images for all masks and detected-object masks.

    Args:
        rgb_img: RGB image array.
        detections: List of all SAM detections.
        detected_objects: List of recognized objects with masks.
        timestamp: Frame timestamp string.
        masks_all_path: Output folder for all masks visualization.
        masks_detected_path: Output folder for detected-object masks visualization.

    Returns:
        None
    """
    mask_image = rgb_img.copy().astype(np.float32)
    colors = np.random.randint(50, 255, (len(detections), 3), dtype=np.uint8)
    
    for idx, det in enumerate(detections):
        mask = det["mask"] 
        color = colors[idx].astype(np.float32)
        mask_image[mask] = mask_image[mask] * 0.5 + color * 0.5
    
    mask_image = np.clip(mask_image, 0, 255).astype(np.uint8)
    os.makedirs(masks_all_path, exist_ok=True)
    mask_output_file = os.path.join(masks_all_path, f"{timestamp}.png")
    cv2.imwrite(mask_output_file, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
    

    if len(detected_objects) > 0:
        mask_image = rgb_img.copy().astype(np.float32)
        colors = np.random.randint(50, 255, (len(detected_objects), 3), dtype=np.uint8)
        
        for idx, obj in enumerate(detected_objects):
            mask = obj["mask"]
            color = colors[idx].astype(np.float32)
            mask_image[mask] = mask_image[mask] * 0.5 + color * 0.5
        
        mask_image = np.clip(mask_image, 0, 255).astype(np.uint8)
        os.makedirs(masks_detected_path, exist_ok=True)
        mask_output_file = os.path.join(masks_detected_path, f"{timestamp}.png")
        cv2.imwrite(mask_output_file, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))


def filter_duplicate_nodes(graph_dict: Dict[str, Any], position_tol: float = 0.05) -> Dict[str, Any]:
    """Remove duplicate nodes and fix edges pointing to duplicates.

    Args:
        graph_dict: Graph dict with "nodes" and "edges".
        position_tol: Tolerance for considering positions identical.

    Returns:
        Deduplicated graph dict.
    """
    if not isinstance(graph_dict, dict):
        return graph_dict

    nodes = list(graph_dict.get("nodes", []))
    edges = list(graph_dict.get("edges", []))

    def normalize_name(name):
        return str(name or "").strip().lower()

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def pos_bucket(value):
        if position_tol <= 0:
            return value
        return round(value / position_tol) * position_tol

    if not nodes:
        graph_dict["nodes"] = []
        graph_dict["edges"] = []
        return graph_dict

    def find(parent, idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(parent, a, b):
        ra = find(parent, a)
        rb = find(parent, b)
        if ra != rb:
            parent[rb] = ra

    parent = list(range(len(nodes)))
    bucket_x = {}
    bucket_y = {}

    for idx, node in enumerate(nodes):
        name = normalize_name(node.get("name"))
        layer = normalize_name(node.get("layer"))
        xb = pos_bucket(safe_float(node.get("x")))
        yb = pos_bucket(safe_float(node.get("y")))
        key_x = (name, layer, xb)
        key_y = (name, layer, yb)

        if key_x in bucket_x:
            union(parent, idx, bucket_x[key_x])
        else:
            bucket_x[key_x] = idx

        if key_y in bucket_y:
            union(parent, idx, bucket_y[key_y])
        else:
            bucket_y[key_y] = idx

    root_to_keep = {}
    id_map = {}
    deduped_nodes = []

    for idx, node in enumerate(nodes):
        root = find(parent, idx)
        if root not in root_to_keep:
            root_to_keep[root] = node
            deduped_nodes.append(node)
            continue

        kept_node = root_to_keep[root]
        kept_id = kept_node.get("id")
        node_id = node.get("id")
        if kept_id is not None and node_id is not None:
            id_map[node_id] = kept_id

    room_nodes = [node for node in deduped_nodes if normalize_name(node.get("layer")) == "room"]
    if len(room_nodes) > 1:
        room_nodes_sorted = sorted(
            room_nodes,
            key=lambda node: safe_float(node.get("id"), float("inf"))
        )
        keep_room = room_nodes_sorted[0]
        keep_room_id = keep_room.get("id")
        for node in room_nodes_sorted[1:]:
            node_id = node.get("id")
            if keep_room_id is not None and node_id is not None:
                id_map[node_id] = keep_room_id
        deduped_nodes = [node for node in deduped_nodes if node.get("id") == keep_room_id or normalize_name(node.get("layer")) != "room"]

    deduped_edges = []
    edge_seen = set()

    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        rel = edge.get("rel")

        if src in id_map:
            src = id_map[src]
        if dst in id_map:
            dst = id_map[dst]

        if src == dst:
            continue

        edge_key = (src, dst, normalize_name(rel))
        if edge_key in edge_seen:
            continue

        edge_seen.add(edge_key)
        deduped_edges.append({"from": src, "to": dst, "rel": rel})

    graph_dict["nodes"] = deduped_nodes
    graph_dict["edges"] = deduped_edges
    return graph_dict

'''
{
    "_counter": 2,  # Next frame ID would be 2
    
    "frame_0": {
        "frame_id": 0,
        "timestamp": "1305031455",
        "status": "completed",
        "queued_at": 1768741457.0666,
        "started_at": 1768741457.0773,
        "completed_at": 1768741457.3992,
        "source": "dataset"
    },
    
    "frame_1": {
        "frame_id": 1,
        "timestamp": "1305031458",
        "status": "completed",
        "queued_at": 1768741758.1234,
        "started_at": 1768741758.2456,
        "completed_at": 1768741758.6789,
        "source": "dataset"
    },
    
    "latest": {
        "nodes": [
            {"id": 0, "name": "object1", "x": 0.5, "y": 0.5},
            {"id": 1, "name": "object2", "x": 0.3, "y": 0.7}
        ],
        "edges": [
            {"from": 0, "to": 1, "rel": "next_to"}
        ]
    }
}
'''