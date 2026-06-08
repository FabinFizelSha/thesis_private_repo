#!/usr/bin/env python3
"""ROS2 semantic adapter node for Hydra-compatible risk scene mapping.

This node implements the real-time side of the proposed architecture:

1. Receive RGB-D and CameraInfo from a rosbag or robot.
2. Segment object candidates with SAM/dummy segmenter.
3. Ask RAP for a fast known-label match.
4. If RAP is low-confidence, publish the object immediately as
   ``unknown_object`` and enqueue its crop for asynchronous VLM inference.
5. Publish a dense semantic label image for Hydra.
6. Optionally republish RGB, depth, and CameraInfo after semantic processing so
   Hydra receives a synchronized delayed input bundle.
7. Publish delayed VLM/risk updates on ``/rsg/semantic/vlm_updates``.

The key timing rule is: **Hydra never waits for Qwen/VLM**. Unknown objects are
sent to Hydra immediately with a stable unknown-object class ID. The VLM result
is published later with the same ``adapter_object_id`` so a fusion node can match
it to the corresponding Hydra node.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import message_filters
import tf2_ros

from .segmenters import make_segmenter
from .rap_client import make_rap_client
from .qwen_vlm_client import make_vlm_client
from .unknown_object_queue import UnknownObjectJob, UnknownObjectVlmQueue
from .utils import (
    crop_from_bbox,
    create_color_overlay,
    depth_to_meters,
    centroid_from_mask_depth,
    transform_point,
)


class SemanticAdapterNode(Node):
    """ROS2 node that produces Hydra-compatible semantic images.

    The class owns ROS publishers/subscribers and orchestrates the perception
    pipeline. Heavy VLM work is delegated to ``UnknownObjectVlmQueue`` so that
    the callback can publish a semantic image without waiting for Qwen.
    """

    def __init__(self):
        super().__init__("rsg_semantic_adapter")

        self._declare_parameters()
        self.bridge = CvBridge()

        self.camera_k: Optional[np.ndarray] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.last_process_time = 0.0
        self.object_counter = 0

        self.map_frame = self.get_parameter("map_frame").value
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.background_label_id = int(self.get_parameter("background_label_id").value)
        self.object_candidate_label_id = int(self.get_parameter("object_candidate_label_id").value)
        self.unknown_object_label_id = int(self.get_parameter("unknown_object_label_id").value)
        self.unknown_label_name = str(self.get_parameter("unknown_label_name").value)

        self.label_map: Dict[str, int] = {
            str(k).lower(): int(v)
            for k, v in json.loads(self.get_parameter("label_map_json").value).items()
        }

        params = self._algorithm_params()
        self.segmenter = make_segmenter(self.get_parameter("segmenter_mode").value, params)
        self.rap = make_rap_client(self.get_parameter("rap_mode").value, params)
        self.vlm = make_vlm_client(self.get_parameter("vlm_mode").value, params)

        self._create_publishers()
        self._create_tf_and_subscribers()
        self._create_vlm_queue()

        self.get_logger().info(
            "RSG Semantic Adapter started: "
            f"segmenter={self.get_parameter('segmenter_mode').value}, "
            f"RAP={self.get_parameter('rap_mode').value}, "
            f"VLM={self.get_parameter('vlm_mode').value}, "
            f"vlm_async_mode={self.get_parameter('vlm_async_mode').value}, "
            f"republish_hydra_inputs={self.get_parameter('republish_hydra_inputs').value}"
        )

    def _declare_parameters(self) -> None:
        """Declare all ROS parameters used by the node."""
        # Input topics.
        self.declare_parameter("rgb_topic", "/tesse/left_cam/rgb/image_raw")
        self.declare_parameter("depth_topic", "/tesse/depth_cam/mono/image_raw")
        self.declare_parameter("camera_info_topic", "/tesse/left_cam/camera_info")

        # Semantic adapter output topics.
        self.declare_parameter("semantic_topic", "/rsg/semantic/image_raw")
        self.declare_parameter("overlay_topic", "/rsg/semantic/overlay")
        self.declare_parameter("object_updates_topic", "/rsg/semantic/object_updates")
        self.declare_parameter("vlm_updates_topic", "/rsg/semantic/vlm_updates")

        # Delayed synchronized Hydra input topics.
        self.declare_parameter("republish_hydra_inputs", True)
        self.declare_parameter("hydra_rgb_topic", "/rsg/hydra/left_cam/rgb/image_raw")
        self.declare_parameter(
            "hydra_depth_topic", "/rsg/hydra/left_cam/depth_registered/image_rect"
        )
        self.declare_parameter(
            "hydra_camera_info_topic", "/rsg/hydra/left_cam/rgb/camera_info"
        )
        self.declare_parameter("use_rgb_header_for_republished_depth", True)

        # Frames and depth scaling.
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("depth_scale", 0.001)

        # Pipeline modes.
        self.declare_parameter("segmenter_mode", "dummy_grid")
        self.declare_parameter("rap_mode", "disabled")
        self.declare_parameter("vlm_mode", "disabled")
        self.declare_parameter("use_vlm_for_low_confidence", True)
        self.declare_parameter("vlm_async_mode", True)

        # Runtime control.
        self.declare_parameter("min_process_interval_s", 0.5)
        self.declare_parameter("sync_slop_s", 0.08)
        self.declare_parameter("rap_confidence_threshold", 0.65)
        self.declare_parameter("vlm_confidence_threshold", 0.40)
        self.declare_parameter("max_masks_per_frame", 12)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("semantic_encoding", "mono8")

        # Labels. 41 remains the ADE20K-compatible temporary object proxy.
        self.declare_parameter("background_label_id", 0)
        self.declare_parameter("object_candidate_label_id", 41)
        self.declare_parameter("unknown_object_label_id", 41)
        self.declare_parameter("unknown_label_name", "unknown_object")
        self.declare_parameter(
            "label_map_json",
            json.dumps(
                {
                    "wall": 0,
                    "floor": 3,
                    "cabinet": 10,
                    "door": 14,
                    "table": 15,
                    "chair": 19,
                    "sofa": 23,
                    "shelf": 24,
                    "desk": 33,
                    "box": 41,
                    "bottle": 98,
                    "monitor": 143,
                    "object_candidate": 41,
                    "unknown_object": 41,
                    "extension cable": 41,
                    "cable": 41,
                    "trip hazard": 41,
                }
            ),
        )

        # SAM parameters.
        self.declare_parameter("sam_checkpoint", "")
        self.declare_parameter("sam_model_type", "vit_b")
        self.declare_parameter("sam_device", "cuda")
        self.declare_parameter("sam_min_area", 800)
        self.declare_parameter("sam_max_masks", 25)
        self.declare_parameter("dummy_box_fraction", 0.35)

        # SAM depth-gate parameters.
        self.declare_parameter("use_depth_gate", True)
        self.declare_parameter("sam_min_depth_m", 0.25)
        self.declare_parameter("sam_max_depth_m", 3.0)
        self.declare_parameter("sam_min_near_pixels", 1500)
        self.declare_parameter("sam_min_mask_depth_overlap", 0.50)
        self.declare_parameter("sam_depth_dilate_px", 9)

        # SAM ROI and resize parameters.
        self.declare_parameter("sam_use_depth_roi", True)
        self.declare_parameter("sam_roi_padding_px", 24)
        self.declare_parameter("sam_min_roi_area_px", 2500)
        self.declare_parameter("sam_max_rois", 4)
        self.declare_parameter("sam_points_per_side", 8)
        self.declare_parameter("sam_pred_iou_thresh", 0.92)
        self.declare_parameter("sam_stability_score_thresh", 0.88)
        self.declare_parameter("sam_resize_input", True)
        self.declare_parameter("sam_resize_max_side", 512)
        self.declare_parameter("sam_resize_min_scale", 0.25)

        # RAP HTTP parameters.
        self.declare_parameter("rap_http_endpoint", "http://127.0.0.1:8010/query")
        self.declare_parameter("rap_top_k", 5)
        self.declare_parameter("rap_timeout_s", 2.0)

        # Local OpenAI-compatible VLM endpoint parameters.
        self.declare_parameter("vlm_endpoint", "http://127.0.0.1:8005/v1/chat/completions")
        self.declare_parameter("vlm_model", "qwen2.5-vl")
        self.declare_parameter("vlm_api_key", "EMPTY")
        self.declare_parameter("vlm_timeout_s", 20.0)

        # FIFO queue controls for unknown-object VLM processing.
        self.declare_parameter("enqueue_unknown_for_vlm", True)
        self.declare_parameter("vlm_queue_max_size", 32)
        self.declare_parameter("vlm_queue_drop_policy", "drop_oldest")

    def _algorithm_params(self) -> Dict[str, object]:
        """Collect non-ROS algorithm parameters for helper classes."""
        names = [
            "sam_checkpoint",
            "sam_model_type",
            "sam_device",
            "sam_min_area",
            "sam_max_masks",
            "dummy_box_fraction",
            "use_depth_gate",
            "sam_min_depth_m",
            "sam_max_depth_m",
            "sam_min_near_pixels",
            "sam_min_mask_depth_overlap",
            "sam_depth_dilate_px",
            "sam_use_depth_roi",
            "sam_roi_padding_px",
            "sam_min_roi_area_px",
            "sam_max_rois",
            "sam_points_per_side",
            "sam_pred_iou_thresh",
            "sam_stability_score_thresh",
            "sam_resize_input",
            "sam_resize_max_side",
            "sam_resize_min_scale",
            "rap_http_endpoint",
            "rap_top_k",
            "rap_timeout_s",
            "vlm_endpoint",
            "vlm_model",
            "vlm_api_key",
            "vlm_timeout_s",
        ]
        return {name: self.get_parameter(name).value for name in names}

    def _create_publishers(self) -> None:
        """Create all ROS publishers."""
        self.semantic_pub = self.create_publisher(
            Image, self.get_parameter("semantic_topic").value, 10
        )
        self.overlay_pub = self.create_publisher(
            Image, self.get_parameter("overlay_topic").value, 10
        )
        self.updates_pub = self.create_publisher(
            String, self.get_parameter("object_updates_topic").value, 10
        )
        self.vlm_updates_pub = self.create_publisher(
            String, self.get_parameter("vlm_updates_topic").value, 10
        )

        self.republish_hydra_inputs = bool(
            self.get_parameter("republish_hydra_inputs").value
        )
        if self.republish_hydra_inputs:
            self.hydra_rgb_pub = self.create_publisher(
                Image, self.get_parameter("hydra_rgb_topic").value, 10
            )
            self.hydra_depth_pub = self.create_publisher(
                Image, self.get_parameter("hydra_depth_topic").value, 10
            )
            self.hydra_info_pub = self.create_publisher(
                CameraInfo, self.get_parameter("hydra_camera_info_topic").value, 10
            )
        else:
            self.hydra_rgb_pub = None
            self.hydra_depth_pub = None
            self.hydra_info_pub = None

    def _create_tf_and_subscribers(self) -> None:
        """Create TF buffer and input subscriptions."""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("rgb_topic").value, qos_profile=qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("depth_topic").value, qos_profile=qos
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=float(self.get_parameter("sync_slop_s").value),
        )
        self.sync.registerCallback(self.synced_callback)

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.camera_info_callback,
            10,
        )

    def _create_vlm_queue(self) -> None:
        """Create and start the optional asynchronous VLM FIFO worker."""
        enabled = (
            bool(self.get_parameter("enqueue_unknown_for_vlm").value)
            and bool(self.get_parameter("vlm_async_mode").value)
            and self.get_parameter("vlm_mode").value.lower() != "disabled"
        )
        self.vlm_queue = UnknownObjectVlmQueue(
            self.vlm,
            self.publish_vlm_update,
            enabled=enabled,
            max_size=int(self.get_parameter("vlm_queue_max_size").value),
            drop_policy=str(self.get_parameter("vlm_queue_drop_policy").value),
            logger=self.get_logger(),
        )
        self.vlm_queue.start()

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Store camera intrinsics and latest CameraInfo message."""
        self.latest_camera_info = msg
        self.camera_k = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def label_to_id(self, label: str) -> int:
        """Map a string label to a stable Hydra integer class ID."""
        return int(self.label_map.get(str(label).lower(), self.object_candidate_label_id))

    def lookup_transform_optional(self, target_frame: str, source_frame: str):
        """Try a TF lookup without blocking semantic image publication."""
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception:
            return None

    def make_semantic_msg(self, semantic: np.ndarray, header) -> Image:
        """Convert a 2D label array into a ROS Image for Hydra."""
        encoding = self.get_parameter("semantic_encoding").value
        msg = Image()
        msg.header = header
        msg.height = int(semantic.shape[0])
        msg.width = int(semantic.shape[1])
        msg.is_bigendian = False

        if encoding == "16UC1":
            arr = semantic.astype(np.uint16)
            msg.encoding = "16UC1"
            msg.step = int(msg.width * 2)
        else:
            arr = semantic.astype(np.uint8)
            msg.encoding = "mono8"
            msg.step = int(msg.width)

        msg.data = arr.tobytes()
        return msg

    def republish_inputs_for_hydra(self, rgb_msg: Image, depth_msg: Image) -> None:
        """Republish RGB/depth/CameraInfo after semantic processing.

        This makes Hydra consume a slower but synchronized input bundle instead
        of raw RGB/depth plus delayed semantic images.
        """
        if not self.republish_hydra_inputs:
            return
        if self.latest_camera_info is None:
            return

        rgb_out = copy.deepcopy(rgb_msg)
        depth_out = copy.deepcopy(depth_msg)
        info_out = copy.deepcopy(self.latest_camera_info)

        if bool(self.get_parameter("use_rgb_header_for_republished_depth").value):
            depth_out.header = copy.deepcopy(rgb_msg.header)
        info_out.header = copy.deepcopy(rgb_msg.header)

        self.hydra_info_pub.publish(info_out)
        self.hydra_rgb_pub.publish(rgb_out)
        self.hydra_depth_pub.publish(depth_out)

    def publish_vlm_update(self, update: Dict[str, object]) -> None:
        """Publish one delayed VLM/risk update as JSON."""
        self.vlm_updates_pub.publish(String(data=json.dumps(update)))

    def _make_adapter_object_id(self, stamp, local_index: int) -> str:
        """Create a stable ID for one SAM/RAP object observation."""
        return f"{int(stamp.sec)}_{int(stamp.nanosec)}_{int(local_index):04d}"

    def _enqueue_unknown_if_needed(
        self,
        *,
        crop: np.ndarray,
        adapter_object_id: str,
        rgb_msg: Image,
        bbox_xywh: List[int],
        centroid_camera: Optional[List[float]],
        centroid_map: Optional[List[float]],
        rap_label: str,
        rap_confidence: float,
        mask_area_px: int,
    ) -> bool:
        """Push a low-confidence object crop to the background VLM queue."""
        if self.get_parameter("vlm_mode").value.lower() == "disabled":
            return False
        if not bool(self.get_parameter("enqueue_unknown_for_vlm").value):
            return False
        if not bool(self.get_parameter("vlm_async_mode").value):
            return False

        job = UnknownObjectJob(
            adapter_object_id=adapter_object_id,
            stamp={
                "sec": int(rgb_msg.header.stamp.sec),
                "nanosec": int(rgb_msg.header.stamp.nanosec),
            },
            frame_id=rgb_msg.header.frame_id,
            crop_rgb=crop,
            bbox_xywh=bbox_xywh,
            centroid_camera=centroid_camera,
            centroid_map=centroid_map,
            semantic_label=self.unknown_label_name,
            semantic_label_id=self.unknown_object_label_id,
            rap_label=rap_label,
            rap_confidence=float(rap_confidence),
            mask_area_px=int(mask_area_px),
        )
        return self.vlm_queue.enqueue(job)

    def synced_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """Process one synchronized RGB-D pair."""
        t_start = time.perf_counter()
        now = time.time()
        min_dt = float(self.get_parameter("min_process_interval_s").value)
        if now - self.last_process_time < min_dt:
            return
        self.last_process_time = now

        if self.camera_k is None:
            self.get_logger().warn("Waiting for CameraInfo before semantic generation.")
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Failed image conversion: {exc}")
            return

        depth_m = depth_to_meters(depth, self.depth_scale)
        h, w = rgb.shape[:2]
        semantic = np.full((h, w), self.background_label_id, dtype=np.uint16)

        try:
            proposals = self.segmenter.segment(rgb, depth_m)
        except Exception as exc:
            self.get_logger().error(f"Segmentation failed: {exc}")
            return

        proposals = proposals[: int(self.get_parameter("max_masks_per_frame").value)]
        tf_map_from_camera = self.lookup_transform_optional(self.map_frame, rgb_msg.header.frame_id)
        objects: List[Dict[str, object]] = []

        rap_threshold = float(self.get_parameter("rap_confidence_threshold").value)
        vlm_sync_allowed = (
            bool(self.get_parameter("use_vlm_for_low_confidence").value)
            and not bool(self.get_parameter("vlm_async_mode").value)
            and self.get_parameter("vlm_mode").value.lower() != "disabled"
        )

        for proposal in proposals:
            self.object_counter += 1
            adapter_object_id = self._make_adapter_object_id(
                rgb_msg.header.stamp, self.object_counter
            )

            x, y, bw, bh = proposal.bbox
            bbox_xywh = [int(x), int(y), int(bw), int(bh)]
            crop = crop_from_bbox(rgb, proposal.bbox)
            rap = self.rap.query(crop)

            centroid_cam_tuple = centroid_from_mask_depth(proposal.mask, depth_m, self.camera_k)
            centroid_map_tuple = (
                transform_point(centroid_cam_tuple, tf_map_from_camera)
                if centroid_cam_tuple and tf_map_from_camera
                else None
            )
            centroid_camera = list(centroid_cam_tuple) if centroid_cam_tuple else None
            centroid_map = list(centroid_map_tuple) if centroid_map_tuple else None

            is_known_by_rap = float(rap.confidence) >= rap_threshold
            queued_for_vlm = False
            risk_type = "unknown"
            risk_score = 0.0
            reason = ""

            if is_known_by_rap:
                final_label = rap.label
                final_conf = float(rap.confidence)
                final_source = rap.source
                label_id = self.label_to_id(final_label)
            else:
                # Fast-path behavior: publish an unknown-object class to Hydra
                # immediately and optionally send the crop to Qwen in parallel.
                final_label = self.unknown_label_name
                final_conf = float(rap.confidence)
                final_source = f"{rap.source}_low_confidence"
                label_id = self.unknown_object_label_id

                if vlm_sync_allowed:
                    # Backward-compatible mode: Qwen blocks the callback.
                    vlm = self.vlm.query(
                        crop,
                        context=(
                            f"adapter_object_id={adapter_object_id}; "
                            f"frame={rgb_msg.header.frame_id}; "
                            f"RAP={rap.label}:{rap.confidence:.2f}"
                        ),
                    )
                    if vlm.confidence >= float(
                        self.get_parameter("vlm_confidence_threshold").value
                    ):
                        final_label = vlm.label
                        final_conf = float(vlm.confidence)
                        final_source = vlm.source
                        label_id = self.label_to_id(final_label)
                    risk_type = vlm.risk_type
                    risk_score = float(vlm.risk_score)
                    reason = vlm.reason
                else:
                    queued_for_vlm = self._enqueue_unknown_if_needed(
                        crop=crop,
                        adapter_object_id=adapter_object_id,
                        rgb_msg=rgb_msg,
                        bbox_xywh=bbox_xywh,
                        centroid_camera=centroid_camera,
                        centroid_map=centroid_map,
                        rap_label=rap.label,
                        rap_confidence=float(rap.confidence),
                        mask_area_px=int(proposal.mask.sum()),
                    )

            semantic[proposal.mask.astype(bool)] = int(label_id)

            objects.append(
                {
                    "adapter_object_id": adapter_object_id,
                    "local_object_id": self.object_counter,
                    "label": final_label,
                    "label_id": int(label_id),
                    "confidence": float(final_conf),
                    "source": final_source,
                    "rap_label": rap.label,
                    "rap_confidence": float(rap.confidence),
                    "rap_threshold": rap_threshold,
                    "known_by_rap": bool(is_known_by_rap),
                    "queued_for_vlm": bool(queued_for_vlm),
                    "vlm_queue_size": self.vlm_queue.size(),
                    "bbox_xywh": bbox_xywh,
                    "mask_area_px": int(proposal.mask.sum()),
                    "centroid_camera": centroid_camera,
                    "centroid_map": centroid_map,
                    "risk_type": risk_type,
                    "risk_score": float(risk_score),
                    "reason": reason,
                }
            )

        semantic_msg = self.make_semantic_msg(semantic, rgb_msg.header)

        # Publish delayed Hydra inputs first, then semantic image with the same
        # original timestamp. Hydra sees a slower but internally synchronized set.
        self.republish_inputs_for_hydra(rgb_msg, depth_msg)
        self.semantic_pub.publish(semantic_msg)

        if bool(self.get_parameter("publish_overlay").value):
            colors = {
                self.background_label_id: (0, 0, 0),
                self.object_candidate_label_id: (255, 128, 0),
                self.unknown_object_label_id: (255, 0, 255),
            }
            for _, label_id in self.label_map.items():
                colors[int(label_id)] = (
                    int((int(label_id) * 53) % 255),
                    int((int(label_id) * 97) % 255),
                    int((int(label_id) * 193) % 255),
                )
            overlay = create_color_overlay(rgb, semantic, colors)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
            overlay_msg.header = rgb_msg.header
            self.overlay_pub.publish(overlay_msg)

        elapsed = time.perf_counter() - t_start
        update_payload = {
            "stamp": {
                "sec": int(rgb_msg.header.stamp.sec),
                "nanosec": int(rgb_msg.header.stamp.nanosec),
            },
            "frame_id": rgb_msg.header.frame_id,
            "map_frame": self.map_frame,
            "processing_latency_s": float(elapsed),
            "vlm_queue_size": self.vlm_queue.size(),
            "objects": objects,
        }
        self.updates_pub.publish(String(data=json.dumps(update_payload)))

        self.get_logger().info(
            f"Published semantic image with {len(objects)} objects; "
            f"queued={sum(1 for o in objects if o['queued_for_vlm'])}; "
            f"queue_size={self.vlm_queue.size()}; "
            f"latency={elapsed:.3f}s"
        )

    def shutdown_workers(self) -> None:
        """Stop background workers before node shutdown."""
        if hasattr(self, "vlm_queue"):
            self.vlm_queue.stop()


def main(args=None):
    rclpy.init(args=args)
    node = SemanticAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_workers()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
