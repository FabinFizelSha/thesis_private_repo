import json
import os
import time
from typing import Any, Dict, Optional

import cv2
from PIL import Image as PILImage
import networkx as nx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from risk_scene_graph_core.config.parameters import img_size, tmp_dir, masks_output_path, masks_detected_output_path, vrap_distance_threshold, sam_stability_threshold, sam_pred_iou_thresh, sam_padding, sam_min_segment_pixels, K_rgb, dist_coeffs_rgb, K_depth, dist_coeffs_depth
from risk_scene_graph_core.scripts.SceneGraph import SceneGraph
from risk_scene_graph_core.scripts.VlmHelper import VLMHelper, DummyVLMHelper
from risk_scene_graph_core.scripts.WorkerUtils import (
    b64_to_ndarray,
    ensure_dirs,
    fill_invalid_depth_nearest,
    filter_duplicate_nodes,
    graph_to_json,
    save_masks_image,
)




class FramePayload(BaseModel):
    timestamp: str = Field(..., example="1305031455")
    rgb: str = Field(..., description="Base64-encoded RGB image (JPEG/PNG)", example="iVBORw0KGgoAAAANSUhEUg...")
    depth: str = Field(..., description="Base64-encoded depth image (PNG, same resolution as RGB)", example="iVBORw0KGgoAAAANSUhEUg...")
    tx: list[float] = Field(..., description="Translation vector [x,y,z]", example=[1.234, -0.567, 2.890])
    rotM: list[list[float]] = Field(..., description="3x3 rotation matrix as nested lists", example=[
        [0.9999, -0.0087, 0.0000],
        [0.0087, 0.9999, 0.0000],
        [0.0000, 0.0000, 1.0000]
    ])


class RobotInfoPayload(BaseModel):
    """Flexible payload for robot information - accepts any JSON structure"""
    info: dict = Field(..., description="Any JSON-serializable robot data (e.g., battery, speed, status, etc.)", example={"battery": 85, "speed": 1.5, "mode": "autonomous"})


class QueueWorker:
    def __init__(self, queue, shared_state, use_offline: bool = False, use_sam_rap: bool = True, save_output: bool = True, learning_queue=None, small_vlm: bool = False):
        """Worker that consumes frames from a queue and processes them sequentially.

        Args:
            queue: Multiprocessing queue with frame items.
            shared_state: Shared state dictionary for API.
            use_offline: Whether to use DummyVLMHelper.
            use_sam_rap: Whether to enable SAM and Visual RAP.
            save_output: Whether to persist output artifacts.
            learning_queue: Optional queue for unknown objects.

        Returns:
            None
        """
        self.queue = queue
        self.shared_state = shared_state
        self.use_offline = use_offline
        self.use_sam_rap = use_sam_rap
        self.save_output = save_output
        self.learning_queue = learning_queue
        self.small_vlm = small_vlm

        self.graph = None
        self.vlm_helper = None
        self.sam_segmenter = None
        self.visual_rap = None

    def _init_vlm(self) -> None:
        """Initialize the VLM helper.

        Args:
            None

        Returns:
            None
        """
        if self.use_offline:
            self.vlm_helper = DummyVLMHelper(save_output=self.save_output, small_vlm=self.small_vlm)
            print("[Consumer] Using DummyVLMHelper (offline mode)")
        else:
            try:
                self.vlm_helper = VLMHelper(save_output=self.save_output, small_vlm=self.small_vlm)
                print("[Consumer] Using VLMHelper (online mode)")
            except Exception as e:
                print(f"[Consumer] Failed to init VLMHelper: {e}. Falling back to DummyVLMHelper.")
                self.vlm_helper = DummyVLMHelper(save_output=self.save_output, small_vlm=self.small_vlm)

    def _init_sam_rap(self) -> None:
        """Initialize SAM and Visual RAP helpers if enabled.

        Args:
            None

        Returns:
            None
        """
        if not self.use_sam_rap:
            print("[Consumer] SAM/RAP disabled")
            return

        from risk_scene_graph_core.scripts.SamSegmenter import SamSegmenter
        from risk_scene_graph_core.scripts.isualRAP import VisualRAP
        try:
            self.sam_segmenter = SamSegmenter(
                mask_threshold=sam_stability_threshold,
                pred_iou_thresh=sam_pred_iou_thresh,
                padding=sam_padding,
                min_segment_pixels=sam_min_segment_pixels
            )
            print("[Consumer] SamSegmenter initialized")
        except Exception as e:
            print(f"[Consumer] Warning: Failed to init SamSegmenter: {e}")
            raise

        try:
            self.visual_rap = VisualRAP()
            print("[Consumer] VisualRAP initialized")
        except Exception as e:
            print(f"[Consumer] Warning: Failed to init VisualRAP: {e}")
            raise

    def _mark_frame_status(self, item: Dict[str, Any], status: str, error: Optional[str] = None) -> None:
        """Update shared state for a frame status.

        Args:
            item: Frame payload dict with metadata.
            status: New status string.
            error: Optional error message.

        Returns:
            None
        """
        if "frame_id" not in item:
            return
        frame_id = item["frame_id"]
        frame_key = f"frame_{frame_id}"
        if frame_key not in self.shared_state:
            return
        frame_data = dict(self.shared_state[frame_key])
        frame_data["status"] = status
        if status == "processing":
            frame_data["started_at"] = time.time()
        if status == "failed":
            frame_data["error"] = error or "unknown error"
        self.shared_state[frame_key] = frame_data

    def _process_item(self, item: Dict[str, Any]) -> None:
        """Process a single frame through VLM, SAM, and graph updates.

        Args:
            item: Frame payload dict with images and pose.

        Returns:
            None
        """
        rgb_img = item["rgb_img"]
        depth_img = item.get("depth_img")
        tx = np.array(item["tx"], dtype=float)
        rotM = np.array(item["rotM"], dtype=float)
        timestamp = str(item["timestamp"]) if "timestamp" in item else str(int(time.time()))

        print(f"[ProcessItem] Processing {timestamp}")

        if "robot_info_pending" in self.shared_state:
            robot_info = self.shared_state.pop("robot_info_pending")
            self.graph.add_robot_info(robot_info)

        expected_h, expected_w = int(img_size[0]), int(img_size[1])
        rh, rw = rgb_img.shape[:2]
        dh, dw = depth_img.shape[:2]
        if (rw != expected_w or rh != expected_h) or (dw != expected_w or dh != expected_h):
            raise ValueError(
                f"Image size mismatch. Expected {expected_w}x{expected_h}, got RGB {rw}x{rh}, DEPTH {dw}x{dh}."
            )

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        rgb_img = cv2.undistort(
            rgb_img,
            K_rgb.astype(np.float32),
            dist_coeffs_rgb.astype(np.float32),
        )
        
        depth_img = depth_img.astype(np.float32)
        depth_img = cv2.undistort(
            depth_img,
            K_depth.astype(np.float32),
            dist_coeffs_depth.astype(np.float32),
        )
        depth_img = fill_invalid_depth_nearest(depth_img=depth_img)

        detected_objects = []
        if self.use_sam_rap:
            pil_rgb = PILImage.fromarray(rgb_img)
            detections = self.sam_segmenter.segment(image=pil_rgb)

            for det in detections:
                label, distance = self.visual_rap.query(image=det["crop"], threshold=vrap_distance_threshold)
                label = str(label).strip()

                if label != "unknown":
                    detected_objects.append({
                        "label": label,
                        "bbox": det["bbox"],
                        "distance": float(distance),
                        "score": float(det["score"]),
                        "mask": det["mask"]
                    })
                else:
                    if self.learning_queue is not None:
                        self.learning_queue.put({
                            "crop": det["crop"],
                            "bbox": det["bbox"],
                            "mask": det["mask"],
                            "score": float(det["score"]),
                            "timestamp": timestamp
                        })

            if self.save_output:
                save_masks_image(
                    rgb_img=rgb_img,
                    detections=detections,
                    detected_objects=detected_objects,
                    timestamp=timestamp,
                    masks_all_path=masks_output_path,
                    masks_detected_path=masks_detected_output_path,
                )

        cutted_graph = self.graph.get_cutouts(tx=tx, rotM=rotM, depth_img=depth_img)
        cutted_graph_in_px = self.graph.convert_graph_pos_to_pixel(graph=cutted_graph, tx=tx, rotM=rotM)

        max_id = -1
        for node in cutted_graph_in_px.get("nodes", []):
            node_id = node.get("id")
            if isinstance(node_id, (int, np.integer)):
                max_id = max(max_id, int(node_id))
        graph_next_id = max_id + 1 if max_id >= 0 else 0
        current_next_id = self.shared_state.get("next_node_id", 0)
        next_node_id = max(current_next_id, graph_next_id)

        new_vlm_graph, valid = self.vlm_helper.vlm_inference(
            cutout_dict=cutted_graph_in_px,
            timestamp=timestamp,
            image=rgb_img,
            detected_objects=detected_objects,
            next_node_id=next_node_id,
        )
        if valid:
            if self.small_vlm:
                new_vlm_graph = filter_duplicate_nodes(graph_dict=new_vlm_graph)
            converted_graph_in_coord = self.graph.convert_graph_pixel_to_pos(
                graph=new_vlm_graph,
                depth_img=depth_img,
                tx=tx,
                rotM=rotM,
            )
            self.graph.process_vlm_update(cutout_before=cutted_graph_in_px, vlm_output=converted_graph_in_coord)

        self.graph.update_robot_position(tx=tx, rotM=rotM)

        G = self.graph.get_networkx_graph()
        max_graph_id = max((int(nid) for nid in G.nodes), default=-1)
        graph_next_id = max_graph_id + 1 if max_graph_id >= 0 else 0
        current_next_id = self.shared_state.get("next_node_id", 0)
        self.shared_state["next_node_id"] = max(current_next_id, graph_next_id)

        if self.save_output:
            out_json = graph_to_json(G=G)

            ensure_dirs()

            with open(os.path.join(tmp_dir, "completeGraph", f"{timestamp}.json"), "w") as f:
                json.dump(out_json, f, indent=4)

            self.graph.visualize_graph(save_path=os.path.join(tmp_dir, "outPNG", f"{timestamp}.png"), save=True, show=False)

            self.shared_state["latest"] = out_json
        else:
            G = self.graph.get_networkx_graph()
            self.shared_state["latest"] = graph_to_json(G=G)

        if "frame_id" in item:
            frame_id = item["frame_id"]
            frame_key = f"frame_{frame_id}"
            if frame_key in self.shared_state:
                frame_data = dict(self.shared_state[frame_key])
                frame_data["status"] = "completed" if valid else "failed"
                if not valid:
                    frame_data["error"] = "inference failed"
                frame_data["completed_at"] = time.time()
                self.shared_state[frame_key] = frame_data

    def run(self) -> None:
        """Start consuming queue items until a sentinel is received.

        Args:
            None

        Returns:
            None
        """
        if self.save_output:
            ensure_dirs()
        self.graph = SceneGraph()
        self._init_vlm()
        self._init_sam_rap()

        print("[Consumer] Starting queue processing loop")
        while True:
            item = self.queue.get()
            if item is None:
                break
            try:
                self._mark_frame_status(item, "processing")
                self._process_item(item)
            except Exception as e:
                self._mark_frame_status(item, "failed", str(e))
                print(f"[Consumer] Error processing item: {e}")


class LearningWorker:
    def __init__(self, learning_queue, use_offline: bool = False):
        """Worker that classifies unknown objects and adds them to Visual RAP.

        Args:
            learning_queue: Queue of unknown object crops.
            use_offline: Whether to disable learning.

        Returns:
            None
        """
        self.learning_queue = learning_queue
        self.use_offline = use_offline
        self.vlm_helper = None
        self.visual_rap = None

    def _init_helpers(self) -> bool:
        """Initialize VLM and Visual RAP helpers.

        Args:
            None

        Returns:
            True when initialized successfully, otherwise False.
        """
        if self.use_offline:
            print("[LearningWorker] Offline mode - learning disabled")
            return False

        try:
            self.vlm_helper = VLMHelper(save_output=False)
        except Exception as e:
            print(f"[LearningWorker] Failed to init VLMHelper: {e}. Learning disabled.")
            return False

        try:
            from risk_scene_graph_core.scripts.VisualRAP import VisualRAP
            self.visual_rap = VisualRAP(auto_start_server=False)
            print("[LearningWorker] Ready for unknown object classification")
        except Exception as e:
            print(f"[LearningWorker] Failed to connect to VisualRAP: {e}")
            return False

        return True

    def run(self) -> None:
        """Start learning loop until a sentinel is received.

        Args:
            None

        Returns:
            None
        """
        if not self._init_helpers():
            return

        while True:
            item = self.learning_queue.get()
            if item is None:
                break

            try:
                crop = item["crop"]
                bbox = item["bbox"]
                timestamp = item.get("timestamp", "unknown")

                label = self.vlm_helper.classify_object(crop_image=crop)

                if label is None:
                    print(f"[LearningWorker] Failed to classify object at {timestamp}")
                    continue

                if label in ["unknown", "unclear", "unsure", "n/a"]:
                    print(f"[LearningWorker] Skipped unclear object at {timestamp}")
                    continue

                self.visual_rap.add_image(image=crop, label=label)
                print(f"[LearningWorker] Learned new object: '{label}' (timestamp: {timestamp}, bbox: {bbox})")
            except Exception as e:
                print(f"[LearningWorker] Error processing unknown object: {e}")

def dataset_feeder(queue, dataset: list[dict], shared_state: Dict[str, Any] = None):
    """Feed dataset samples into the processing queue.

    Args:
        queue: Multiprocessing queue for frames.
        dataset: List of dataset samples.
        shared_state: Optional shared state dictionary for tracking.

    Returns:
        None
    """
    for idx, data in enumerate(dataset):
        frame_id = None
        
        # Track frame in shared state 
        if shared_state is not None:
            if "_counter" not in shared_state:
                shared_state["_counter"] = 0
            
            # Increment counter
            frame_id = shared_state["_counter"]
            shared_state["_counter"] = frame_id + 1
            
            # Store as key
            shared_state[f"frame_{frame_id}"] = {
                "frame_id": frame_id,
                "timestamp": data["timestamp"],
                "status": "queued",
                "queued_at": time.time(),
                "source": "dataset"
            }
        
        queue.put(
            {
                "frame_id": frame_id,
                "timestamp": data["timestamp"],
                "rgb_img": data["rgb_img"],
                "depth_img": data.get("depth_img"),
                "tx": data["tx"],
                "rotM": data["rotM"],
                "source": "dataset",
            }
        )


class APIWorker:
    def __init__(self, queue, shared_state, host: str = "0.0.0.0", port: int = 8000):
        """Initialize the API wrapper for the worker queue.

        Args:
            queue: Multiprocessing queue for frames.
            shared_state: Shared state dictionary.
            host: Host address.
            port: Port number.

        Returns:
            None
        """
        self.queue = queue
        self.shared_state = shared_state
        self.host = host
        self.port = int(port)
        # Initialize frame counter
        if "_counter" not in self.shared_state:
            self.shared_state["_counter"] = 0
    
    def _get_next_frame_id(self) -> int:
        """counter increment and return next frame ID.

        Args:
            None

        Returns:
            Next frame ID.
        """
        current = self.shared_state["_counter"]
        self.shared_state["_counter"] = current + 1
        return current

    def _build_app(self) -> FastAPI:
        """Build the FastAPI application with routes.

        Args:
            None

        Returns:
            Configured FastAPI app.
        """
        app = FastAPI(title="RiskSceneGraph Worker API")

        @app.get("/health")
        def health():
            """Health check endpoint.

            Args:
                None

            Returns:
                Status dictionary.
            """
            return {"status": "ok"}

        @app.post("/frame")
        def frame(payload: FramePayload):
            """Enqueue a frame for processing.

            Args:
                payload: Frame payload with images and pose.

            Returns:
                Queue status and frame ID.
            """
            rgb_img = b64_to_ndarray(b64_str=payload.rgb, flags=cv2.IMREAD_COLOR)
            depth_img = b64_to_ndarray(b64_str=payload.depth, flags=cv2.IMREAD_UNCHANGED)
            
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0
            else:
                # Fallback, try to interpret as meters
                depth_img = depth_img.astype(np.float32)
            depth_img = fill_invalid_depth_nearest(depth_img=depth_img)

            tx = np.array(payload.tx, dtype=float)
            rotM = np.array(payload.rotM, dtype=float)
            if rotM.shape != (3, 3):
                raise HTTPException(status_code=400, detail="rotM must be 3x3")

            frame_id = self._get_next_frame_id()
            
            # Track frame status
            self.shared_state[f"frame_{frame_id}"] = {
                "frame_id": frame_id,
                "timestamp": payload.timestamp,
                "status": "queued",
                "queued_at": time.time(),
                "source": "api"
            }

            # Enqueue for processing
            item = {
                "frame_id": frame_id,
                "timestamp": payload.timestamp,
                "rgb_img": rgb_img,
                "depth_img": depth_img,
                "tx": tx,
                "rotM": rotM,
                "source": "api",
            }
            self.queue.put(item)
            return {
                "status": "queued",
                "frame_id": frame_id,
                "timestamp": payload.timestamp,
                "queue_size": self.queue.qsize()
            }

        @app.get("/scenegraph")
        def get_scenegraph():
            """Return the latest scene graph.

            Args:
                None

            Returns:
                Latest scene graph dictionary.
            """
            latest = self.shared_state.get("latest")
            if latest is None:
                return {"nodes": [], "edges": []}
            return latest
        
        @app.post("/robot/info")
        def add_robot_info(payload: RobotInfoPayload):
            """Add or update robot information in the scene graph. 
            
            Note: The robot info will be applied on the next frame processed through the queue.
            It is not applied immediately upon this API call."""
            try:
                self.shared_state["robot_info_pending"] = payload.info
                return {
                    "status": "accepted",
                    "message": "Robot information queued for update on next frame processing",
                    "info": payload.info
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error updating robot info: {e}")
        
        @app.get("/robot")
        def get_robot_info():
            """Get current robot node information from scene graph.

            Args:
                None

            Returns:
                Robot node dictionary or None.
            """
            latest = self.shared_state.get("latest")
            if latest is None:
                return {"robot": None}
            
            # Find and return robot node
            for node in latest.get("nodes", []):
                if node.get("layer") == "ROBOT":
                    return {"robot": node}
            
            return {"robot": None}
        
        @app.get("/queue/status")
        def queue_status():
            """Get current queue status and statistics.

            Args:
                None

            Returns:
                Queue statistics dictionary.
            """
            queued = 0
            processing = 0
            completed = 0
            failed = 0
            
            for key in self.shared_state.keys():
                if key.startswith("frame_"):
                    frame_data = self.shared_state.get(key, {})
                    status = frame_data.get("status")
                    if status == "queued":
                        queued += 1
                    elif status == "processing":
                        processing += 1
                    elif status == "completed":
                        completed += 1
                    elif status == "failed":
                        failed += 1
            
            return {
                "queue_size": self.queue.qsize(),
                "total_frames": self.shared_state.get("_counter", 0),
                "queued": queued,
                "processing": processing,
                "completed": completed,
                "failed": failed
            }
        
        @app.get("/frame/{frame_id}")
        def get_frame_status(frame_id: int):
            """Get status of a specific frame by ID.

            Args:
                frame_id: Frame ID integer.

            Returns:
                Frame status dictionary.
            """
            frame_key = f"frame_{frame_id}"
            
            if frame_key not in self.shared_state:
                raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found")
            
            return self.shared_state.get(frame_key)


        return app

    def run(self):
        """Start the FastAPI server.

        Args:
            None

        Returns:
            None
        """
        app = self._build_app()
        uvicorn.run(app, host=self.host, port=self.port)