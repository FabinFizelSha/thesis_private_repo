"""Single-process Phase 1 object detection node.

This node implements Option A from the design discussion: the coordinator
and object-classifier worker run inside one ROS 2 Python process.  This avoids
the expensive ROS round trip of sending full RGB-D frames from coordinator to
classifier and then sending label maps back to the coordinator.

The node still keeps the same logical separation:

- a FIFO frame queue before SAM/RAP, used as a cushion for occasional slow
  SAM/RAP frames;
- SAM/RAP + unknown-track association;
- a separate FIFO VLM queue after RAP for slow unknown-object identification;
- direct Hydra-ready output for every processed frame.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray, String

from rsg.msg import Phase1ClassificationResult, Phase1VlmResult, RsgFrame, RsgHydraFrame

from rsg_phase1.helper_functions.backends import SamMask, make_rap_backend, make_sam_backend, make_vlm_backend
from rsg_phase1.helper_functions.frame_cache import BoundedFrameCache, CachedFrame, EvidenceBuffer
from rsg_phase1.helper_functions.json_utils import safe_json_dumps, safe_json_loads
from rsg_phase1.helper_functions.label_map_builder import ClassifiedMask, LabelMapBuilder
from rsg_phase1.helper_functions.object_geometry import ObjectGeometryEstimator, filter_metadata
from rsg_phase1.helper_functions.phase1_config import Phase1Config
from rsg_phase1.helper_functions.phase1_csv_recorder import Phase1CsvDebugRecorder
from rsg_phase1.helper_functions.phase1_timing_recorder import Phase1TimingRecorder
from rsg_phase1.helper_functions.rap_memory import RapMemoryUpdater
from rsg_phase1.helper_functions.time_utils import stamp_to_float
from rsg_phase1.helper_functions.unknown_tracker import UnknownObjectTracker


class RSGObjectDetection(Node):
    """Combined coordinator + object classifier for lower latency.

    The ROS boundary is only at the input and output:

    ``/rsg/preprocessed/frame`` -> internal FIFO -> SAM/RAP -> Hydra output.

    There is no ROS message round trip between coordinator and classifier.
    """

    def __init__(self) -> None:
        super().__init__("rsg_object_detection")

        self.declare_parameter("config_file", "")
        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        if not config_file:
            raise ValueError("Parameter 'config_file' must point to rsg_pipeline.yaml")

        self.config = Phase1Config.from_yaml(config_file, node_key="rsg_object_detection")
        self.set_parameters([
            rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, self.config.use_sim_time)
        ])

        self.bridge = CvBridge()
        self.sam_backend = make_sam_backend(self.config, self.get_logger())
        self.rap_backend = make_rap_backend(self.config, self.get_logger())
        self.vlm_backend = make_vlm_backend(self.config)
        self.rap_memory_updater = RapMemoryUpdater(
            enabled=self.config.rap_update_enabled,
            output_path=self.config.rap_memory_path,
            min_confidence=self.config.rap_update_min_confidence,
            logger=self.get_logger(),
        )
        self.geometry_estimator = ObjectGeometryEstimator(self.config)
        self.label_map_builder = LabelMapBuilder(self.config)
        self.unknown_tracker = UnknownObjectTracker(self.config, self.get_logger())

        # One application-level frame FIFO before SAM/RAP. This is the only
        # frame-side processing queue in the combined design.
        self.frame_fifo: "queue.Queue[RsgFrame]" = queue.Queue(maxsize=self.config.request_queue_size)
        self.frame_cache = BoundedFrameCache(self.config.frame_cache_size)
        self.evidence_buffer = EvidenceBuffer(self.config.evidence_buffer_size)

        # Separate FIFO after RAP/unknown tracking. VLM can be slow without
        # blocking the frame-to-Hydra path.
        self.vlm_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.config.vlm_queue_size)
        self._vlm_queue_event_index = 0
        self._frame_fifo_event_index = 0
        self.vlm_queue_dropped_count = 0

        self._stop_event = threading.Event()
        self._classification_thread = threading.Thread(target=self._classification_loop, daemon=True)
        self._vlm_thread = threading.Thread(target=self._vlm_loop, daemon=True)

        input_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=self.config.input_qos_depth, reliability=ReliabilityPolicy.RELIABLE)
        output_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=self.config.output_qos_depth, reliability=ReliabilityPolicy.RELIABLE)

        self.frame_sub = self.create_subscription(RsgFrame, self.config.preprocessed_frame_topic, self.frame_callback, input_qos)

        self.hydra_frame_pub = self.create_publisher(RsgHydraFrame, self.config.hydra_frame_topic, output_qos)
        # Optional classifier-result output is still published for debugging and
        # compatibility, but it is no longer required by the coordinator.
        self.result_pub = self.create_publisher(Phase1ClassificationResult, self.config.perception_result_topic, output_qos)
        self.vlm_result_pub = self.create_publisher(Phase1VlmResult, self.config.vlm_result_topic, output_qos)
        self.status_pub = self.create_publisher(String, self.config.status_topic, output_qos)
        self.unknown_pub = self.create_publisher(String, self.config.unknown_candidates_topic, output_qos)
        self.timing_pub = None
        if self.config.timing_enabled and self.config.publish_timing_topic:
            self.timing_pub = self.create_publisher(Float64MultiArray, self.config.timing_topic, output_qos)

        self.hydra_rgb_pub = None
        self.hydra_depth_pub = None
        self.hydra_camera_info_pub = None
        self.hydra_pose_pub = None
        self.hydra_semantic_pub = None
        self.hydra_instance_pub = None
        self.hydra_metadata_pub = None
        if self.config.publish_hydra_separate_topics:
            self.hydra_rgb_pub = self.create_publisher(Image, self.config.hydra_rgb_topic, output_qos)
            self.hydra_depth_pub = self.create_publisher(Image, self.config.hydra_depth_topic, output_qos)
            self.hydra_camera_info_pub = self.create_publisher(CameraInfo, self.config.hydra_camera_info_topic, output_qos)
            self.hydra_pose_pub = self.create_publisher(PoseStamped, self.config.hydra_pose_topic, output_qos)
            self.hydra_semantic_pub = self.create_publisher(Image, self.config.hydra_semantic_topic, output_qos)
            self.hydra_instance_pub = self.create_publisher(Image, self.config.hydra_instance_topic, output_qos)
            self.hydra_metadata_pub = self.create_publisher(String, self.config.hydra_metadata_topic, output_qos)

        self.debug_semantic_pub = None
        self.debug_instance_pub = None
        if self.config.debug_enabled and self.config.publish_debug_label_images:
            self.debug_semantic_pub = self.create_publisher(Image, self.config.debug_semantic_topic, output_qos)
            self.debug_instance_pub = self.create_publisher(Image, self.config.debug_instance_topic, output_qos)

        self.csv_recorder = Phase1CsvDebugRecorder(
            enabled=self.config.timing_enabled and self.config.write_simple_csv,
            output_dir=self.config.csv_debug_dir,
            node_name="rsg_object_detection",
            logger=self.get_logger(),
        )
        self.timing_recorder = Phase1TimingRecorder(
            enabled=self.config.timing_enabled and self.config.write_timing_excel,
            output_path=self.config.timing_excel_path,
            autosave_every=self.config.timing_excel_autosave_every,
            logger=self.get_logger(),
            sheet_name=self.config.timing_sheet_name,
        )

        self.received_count = 0
        self.processed_count = 0
        self.failed_count = 0
        self.dropped_count = 0
        self.hydra_published_count = 0
        self.unknown_vlm_count = 0

        self._classification_thread.start()
        if self.config.vlm_enabled:
            self._vlm_thread.start()
        self._log_startup_summary()

    def _log_startup_summary(self) -> None:
        self.get_logger().info("rsg_object_detection started in single-process Option-A mode.")
        self.get_logger().info(f"Input frame topic: {self.config.preprocessed_frame_topic}")
        self.get_logger().info(f"Hydra combined output topic: {self.config.hydra_frame_topic}")
        self.get_logger().info(f"Optional classifier result topic: {self.config.perception_result_topic}")
        self.get_logger().info(f"VLM result topic: {self.config.vlm_result_topic}")
        self.get_logger().info(
            f"Profile={self.config.profile}, allow_dummy_fallback={self.config.allow_dummy_fallback}"
        )
        self.get_logger().info(
            f"Frame FIFO size={self.config.request_queue_size}, frame_cache_size={self.config.frame_cache_size}, "
            f"VLM FIFO size={self.config.vlm_queue_size}"
        )
        self.get_logger().info(
            f"SAM backend={self.config.sam_backend}, RAP backend={self.config.rap_backend}, "
            f"VLM enabled={self.config.vlm_enabled}, VLM mode={self.config.vlm_mode}, model={self.config.vlm_model}"
        )
        self.get_logger().info(
            f"Debug enabled: {self.config.debug_enabled} "
            f"(global={self.config.global_debug_enabled}, node={self.config.node_debug_enabled})"
        )

    def frame_callback(self, msg: RsgFrame) -> None:
        """Receive preprocessed frames and enqueue them for SAM/RAP.

        This callback does not run SAM/RAP. It only stores the frame in the
        bounded FIFO so the ROS subscription callback remains lightweight.
        """
        now = time.perf_counter()
        self.received_count += 1
        frame_id = msg.rsg_frame_id
        rgb_time = stamp_to_float(msg.header.stamp)
        self.frame_cache.put(CachedFrame(frame_id=frame_id, sequence=int(msg.sequence), received_monotonic=now, received_stamp_sec=rgb_time, msg=msg))

        if self.frame_fifo.full():
            if self.config.drop_oldest_when_full:
                try:
                    dropped = self.frame_fifo.get_nowait()
                    self.dropped_count += 1
                    self.record_frame_fifo_event("dropped_oldest", dropped, queue_wait_ms=0.0, reason="frame_fifo_full")
                except queue.Empty:
                    pass
            else:
                self.dropped_count += 1
                self.record_frame_fifo_event("dropped_newest", msg, queue_wait_ms=0.0, reason="frame_fifo_full")
                self.publish_status("dropped", frame_id, "frame_fifo_full_drop_newest")
                return

        try:
            self.frame_fifo.put_nowait(msg)
            self.record_frame_fifo_event("enqueued", msg, queue_wait_ms=0.0, reason="preprocessed_frame_received")
        except queue.Full:
            self.dropped_count += 1
            self.record_frame_fifo_event("dropped_newest", msg, queue_wait_ms=0.0, reason="frame_fifo_full_race")
            self.publish_status("dropped", frame_id, "frame_fifo_full")
            return

        if self.received_count % self.config.status_every_n_frames == 0:
            self.publish_status("queued", frame_id, "ok")

    def _classification_loop(self) -> None:
        """Consume the coordinator-owned FIFO and run SAM/RAP sequentially."""
        while not self._stop_event.is_set():
            try:
                frame = self.frame_fifo.get(timeout=0.1)
            except queue.Empty:
                continue

            dequeue_time = time.perf_counter()
            cached = self.frame_cache.get(frame.rsg_frame_id)
            fifo_wait_ms = 0.0
            if cached is not None:
                cached.sent_to_classifier_monotonic = dequeue_time
                cached.sent_to_classifier_delay_ms = (dequeue_time - cached.received_monotonic) * 1000.0
                cached.status = "dequeued_to_sam_rap"
                fifo_wait_ms = cached.sent_to_classifier_delay_ms
            self.record_frame_fifo_event("dequeued_to_sam_rap", frame, queue_wait_ms=fifo_wait_ms, reason="worker_ready")

            try:
                result = self.process_frame(frame, input_age_ms=fifo_wait_ms)

                # The previous implementation wrote the classifier debug row
                # after classifier_delay_ms was finalized but before Hydra
                # publishing. That time was hidden in pipeline_wait_ms. We now
                # measure it explicitly and subtract it from pipeline_wait_ms.
                debug_start = time.perf_counter()
                self.publish_timing_event(result, safe_json_loads(result.metadata_json, default={}))
                result.classifier_debug_record_delay_ms = (time.perf_counter() - debug_start) * 1000.0

                # Optional compatibility/debug output. Keep disabled on the hot
                # path unless explicitly requested because this is a large ROS
                # message containing label images.
                if self.config.publish_debug_label_images:
                    self.result_pub.publish(result)

                self._publish_hydra_from_result(frame, result, cached)
                self.processed_count += 1
                self.publish_status("processed", frame.rsg_frame_id, "ok")
            except Exception as exc:
                self.failed_count += 1
                self.get_logger().error(f"Failed to process frame {frame.rsg_frame_id}: {exc}")
                failed = self.build_failed_result(frame, str(exc))
                if self.config.publish_debug_label_images:
                    self.result_pub.publish(failed)
                self.publish_status("failed", frame.rsg_frame_id, str(exc))

    def _publish_hydra_from_result(self, frame: RsgFrame, result: Phase1ClassificationResult, cached: Optional[CachedFrame]) -> None:
        """Build and publish Hydra-ready output in the same process.

        The method now measures sub-phases explicitly so that a future
        ``pipeline_wait_ms`` spike can be traced to a named phase instead of
        remaining unexplained.
        """
        callback_start = time.perf_counter()

        build_start = time.perf_counter()
        hydra_msg = self.build_hydra_frame(frame, result, build_start, cached)
        hydra_build_delay_ms = (time.perf_counter() - build_start) * 1000.0

        publish_start = time.perf_counter()
        if self.config.publish_hydra_combined:
            self.hydra_frame_pub.publish(hydra_msg)
        if self.config.publish_hydra_separate_topics:
            self.publish_separate_hydra_topics(hydra_msg)
        if self.debug_semantic_pub is not None:
            self.debug_semantic_pub.publish(result.semantic_labels)
        if self.debug_instance_pub is not None:
            self.debug_instance_pub.publish(result.instance_labels)
        hydra_publish_delay_ms = (time.perf_counter() - publish_start) * 1000.0

        unknown_publish_start = time.perf_counter()
        unknowns = safe_json_loads(result.unknown_candidates_json, default=[])
        if unknowns and self.config.include_unknown_objects:
            self.unknown_pub.publish(String(data=result.unknown_candidates_json))
        unknown_publish_delay_ms = (time.perf_counter() - unknown_publish_start) * 1000.0

        # Hydra latency ends here: the Hydra-ready output has been published.
        hydra_publish_complete = time.perf_counter()
        self.hydra_published_count += 1
        coordinator_delay_ms = (hydra_publish_complete - callback_start) * 1000.0
        total_delay_ms = 0.0 if cached is None else (hydra_publish_complete - cached.received_monotonic) * 1000.0
        sent_to_classifier_delay_ms = 0.0 if cached is None else float(cached.sent_to_classifier_delay_ms)
        classifier_debug_ms = float(getattr(result, "classifier_debug_record_delay_ms", 0.0))

        pipeline_wait_ms = max(
            0.0,
            total_delay_ms
            - sent_to_classifier_delay_ms
            - float(result.classifier_delay_ms)
            - classifier_debug_ms
            - coordinator_delay_ms,
        )
        metadata = safe_json_loads(result.metadata_json, default={})

        if self.config.timing_enabled:
            self.csv_recorder.hydra_latency(
                sequence=int(result.sequence),
                frame_id=result.rsg_frame_id,
                status="sent_to_hydra",
                total_delay_ms=total_delay_ms,
                sent_to_classifier_delay_ms=sent_to_classifier_delay_ms,
                coordinator_delay_ms=coordinator_delay_ms,
                classifier_delay_ms=float(result.classifier_delay_ms),
                classifier_debug_record_delay_ms=classifier_debug_ms,
                hydra_build_delay_ms=hydra_build_delay_ms,
                hydra_publish_delay_ms=hydra_publish_delay_ms,
                unknown_publish_delay_ms=unknown_publish_delay_ms,
                evidence_record_delay_ms=0.0,
                pipeline_wait_ms=pipeline_wait_ms,
                num_masks=int(result.num_masks),
                num_known=int(result.num_known),
                num_unknown=int(result.num_unknown),
                num_unknown_tracks=int(metadata.get("num_unknown_tracks", 0) or 0),
                num_vlm_queued=int(metadata.get("num_vlm_queued", 0) or 0),
            )
            self.timing_recorder.add_sample(
                node="rsg_object_detection",
                sequence=int(result.sequence),
                frame_id=result.rsg_frame_id,
                status="sent_to_hydra",
                reason="ok",
                coordinator_delay_ms=coordinator_delay_ms,
                classifier_delay_ms=float(result.classifier_delay_ms),
                total_delay_ms=total_delay_ms,
                sent_to_classifier_delay_ms=sent_to_classifier_delay_ms,
                classifier_debug_record_delay_ms=classifier_debug_ms,
                hydra_build_delay_ms=hydra_build_delay_ms,
                hydra_publish_delay_ms=hydra_publish_delay_ms,
                unknown_publish_delay_ms=unknown_publish_delay_ms,
                pipeline_wait_ms=pipeline_wait_ms,
                num_masks=int(result.num_masks),
                num_known=int(result.num_known),
                num_unknown=int(result.num_unknown),
                num_unknown_tracks=int(metadata.get("num_unknown_tracks", 0) or 0),
                num_vlm_queued=int(metadata.get("num_vlm_queued", 0) or 0),
            )

        evidence_start = time.perf_counter()
        self.add_evidence_record(hydra_msg, result)
        evidence_record_delay_ms = (time.perf_counter() - evidence_start) * 1000.0
        # Evidence is post-Hydra-publish work. It is not added to
        # total_delay_ms, but measuring it prevents confusion if it later causes
        # FIFO wait on following frames.
        if self.config.timing_enabled and evidence_record_delay_ms > 1.0:
            self.get_logger().debug(
                f"Post-Hydra evidence recording took {evidence_record_delay_ms:.3f} ms for {frame.rsg_frame_id}"
            )

        # Keep the cache small: after Hydra output is built, this frame is no
        # longer needed for result matching in the combined mode.
        self.frame_cache.remove(frame.rsg_frame_id)

    def build_hydra_frame(self, frame: RsgFrame, result: Phase1ClassificationResult, callback_start: float, cached: Optional[CachedFrame]) -> RsgHydraFrame:
        """Create the combined Hydra-ready frame message."""
        hydra_msg = RsgHydraFrame()
        hydra_msg.header = frame.header
        hydra_msg.rsg_frame_id = frame.rsg_frame_id
        hydra_msg.source = frame.source
        hydra_msg.sequence = frame.sequence
        hydra_msg.rgb = frame.rgb
        hydra_msg.depth_m = frame.depth_m
        hydra_msg.camera_info = frame.camera_info
        hydra_msg.camera_pose = frame.camera_pose
        hydra_msg.tx = frame.tx
        hydra_msg.rot_m = frame.rot_m
        hydra_msg.semantic_labels = result.semantic_labels
        hydra_msg.instance_labels = result.instance_labels
        hydra_msg.label_table_json = result.label_table_json
        hydra_msg.object_metadata_json = result.object_metadata_json
        hydra_msg.unknown_candidates_json = result.unknown_candidates_json
        hydra_msg.perception_metadata_json = result.metadata_json
        metadata = {
            "phase": "phase1_hydra_input",
            "node": "rsg_object_detection",
            "classifier_success": bool(result.success),
            "classifier_status": result.status,
            "classifier_reason": result.reason,
            "num_masks": int(result.num_masks),
            "num_known": int(result.num_known),
            "num_unknown": int(result.num_unknown),
        }
        if self.config.include_frame_relation_metadata:
            metadata["source_preprocessor_metadata"] = safe_json_loads(frame.metadata_json, default={})
        hydra_msg.metadata_json = safe_json_dumps(metadata)
        hydra_msg.coordinator_delay_ms = (time.perf_counter() - callback_start) * 1000.0
        hydra_msg.classifier_delay_ms = float(result.classifier_delay_ms)
        hydra_msg.total_delay_ms = 0.0 if cached is None else (time.perf_counter() - cached.received_monotonic) * 1000.0
        return hydra_msg

    def publish_separate_hydra_topics(self, hydra_msg: RsgHydraFrame) -> None:
        """Republish the combined Hydra frame fields as separate topics."""
        if self.hydra_rgb_pub is not None:
            self.hydra_rgb_pub.publish(hydra_msg.rgb)
        if self.hydra_depth_pub is not None:
            self.hydra_depth_pub.publish(hydra_msg.depth_m)
        if self.hydra_camera_info_pub is not None:
            self.hydra_camera_info_pub.publish(hydra_msg.camera_info)
        if self.hydra_pose_pub is not None:
            self.hydra_pose_pub.publish(hydra_msg.camera_pose)
        if self.hydra_semantic_pub is not None:
            self.hydra_semantic_pub.publish(hydra_msg.semantic_labels)
        if self.hydra_instance_pub is not None:
            self.hydra_instance_pub.publish(hydra_msg.instance_labels)
        if self.hydra_metadata_pub is not None:
            self.hydra_metadata_pub.publish(String(data=hydra_msg.metadata_json))

    def add_evidence_record(self, hydra_msg: RsgHydraFrame, result: Phase1ClassificationResult) -> None:
        """Store compact frame metadata for future risk-annotation retrieval."""
        if not self.config.store_evidence_frames:
            return
        objects = safe_json_loads(result.object_metadata_json, default=[])
        record = {
            "frame_id": hydra_msg.rsg_frame_id,
            "sequence": int(hydra_msg.sequence),
            "timestamp_sec": stamp_to_float(hydra_msg.header.stamp),
            "num_objects": len(objects) if isinstance(objects, list) else 0,
            "object_ids": [obj.get("candidate_id", "") for obj in objects] if isinstance(objects, list) else [],
            "total_delay_ms": float(hydra_msg.total_delay_ms),
        }
        self.evidence_buffer.add(record)

    def record_frame_fifo_event(self, event: str, frame: RsgFrame, queue_wait_ms: float = 0.0, reason: str = "") -> None:
        """Record compact FIFO events for queue-size and wait-time plots."""
        if not self.config.timing_enabled:
            return
        self._frame_fifo_event_index += 1
        self.csv_recorder.frame_fifo_event(
            event_index=self._frame_fifo_event_index,
            event=event,
            sequence=int(frame.sequence),
            frame_id=frame.rsg_frame_id,
            queue_size=int(self.frame_fifo.qsize()),
            queue_max_size=int(self.config.request_queue_size),
            queue_wait_ms=float(queue_wait_ms),
            reason=reason,
        )

    def process_frame(self, frame: RsgFrame, input_age_ms: float = 0.0) -> Phase1ClassificationResult:
        """Run SAM, RAP, label-map construction, and VLM dispatch for one frame.

        The classifier timing is now split into named sub-phases. This avoids
        hiding expensive operations, such as image conversion or result-message
        construction, inside an unexplained delay column.
        """
        start = time.perf_counter()
        input_age_ms = float(input_age_ms)

        conversion_start = time.perf_counter()
        rgb = self.bridge.imgmsg_to_cv2(frame.rgb, desired_encoding="rgb8")
        depth = self.bridge.imgmsg_to_cv2(frame.depth_m, desired_encoding="32FC1")
        tx = np.array(frame.tx, dtype=np.float64)
        rot_m = np.array(frame.rot_m, dtype=np.float64).reshape(3, 3)
        image_conversion_delay_ms = (time.perf_counter() - conversion_start) * 1000.0

        sam_start = time.perf_counter()
        sam_masks = self.run_sam(rgb)
        sam_delay_ms = (time.perf_counter() - sam_start) * 1000.0

        rap_start = time.perf_counter()
        classified, track_records = self.run_rap_and_metadata(frame, rgb, depth, tx, rot_m, sam_masks)
        rap_delay_ms = (time.perf_counter() - rap_start) * 1000.0

        label_start = time.perf_counter()
        semantic, instance, label_table, objects, unknowns = self.label_map_builder.build(rgb.shape[:2], classified)
        label_map_delay_ms = (time.perf_counter() - label_start) * 1000.0

        metadata_start = time.perf_counter()
        vlm_dispatch = self.dispatch_unknowns_to_vlm(frame, rgb, depth, unknowns, classified)
        metadata = self.build_result_metadata(frame, sam_masks, objects, unknowns, vlm_dispatch, track_records)
        metadata_delay_ms = (time.perf_counter() - metadata_start) * 1000.0

        # Building ROS Image messages and JSON strings is a real cost and can be
        # significant with large label maps/metadata. Measure it separately.
        result_msg_start = time.perf_counter()
        result = Phase1ClassificationResult()
        result.header = frame.header
        result.rsg_frame_id = frame.rsg_frame_id
        result.sequence = frame.sequence
        result.success = True
        result.status = "ok"
        result.reason = "ok"
        result.semantic_labels = self.bridge.cv2_to_imgmsg(semantic, encoding=self.config.semantic_label_encoding)
        result.semantic_labels.header = frame.header
        result.instance_labels = self.bridge.cv2_to_imgmsg(instance, encoding=self.config.instance_label_encoding)
        result.instance_labels.header = frame.header
        result.label_table_json = safe_json_dumps(label_table)
        result.object_metadata_json = safe_json_dumps(objects if self.config.include_object_metadata else [])
        result.unknown_candidates_json = safe_json_dumps(unknowns if self.config.include_unknown_objects else [])
        result.vlm_dispatch_json = safe_json_dumps(vlm_dispatch)
        result.metadata_json = safe_json_dumps(metadata)
        result.input_age_ms = float(input_age_ms)
        result.sam_delay_ms = float(sam_delay_ms)
        result.rap_delay_ms = float(rap_delay_ms)
        result.label_map_delay_ms = float(label_map_delay_ms)
        result.metadata_delay_ms = float(metadata_delay_ms)
        result.image_conversion_delay_ms = float(image_conversion_delay_ms)
        result.result_message_build_delay_ms = (time.perf_counter() - result_msg_start) * 1000.0
        result.classifier_debug_record_delay_ms = 0.0
        result.num_masks = int(len(sam_masks))
        result.num_known = int(len([obj for obj in objects if not str(obj.get("status", "")).startswith("unknown")]))
        result.num_unknown = int(len(unknowns))
        result.classifier_delay_ms = (time.perf_counter() - start) * 1000.0
        return result

    def run_sam(self, rgb: np.ndarray) -> List[SamMask]:
        """Run configured SAM backend and filter tiny masks."""
        if not self.config.sam_enabled:
            return []
        masks = self.sam_backend.segment(rgb)
        filtered: List[SamMask] = []
        for mask in masks[: self.config.sam_max_masks]:
            if int(mask.area_px) >= self.config.sam_min_mask_pixels:
                filtered.append(mask)
        return filtered

    def run_rap_and_metadata(
        self,
        frame: RsgFrame,
        rgb: np.ndarray,
        depth: np.ndarray,
        tx: np.ndarray,
        rot_m: np.ndarray,
        sam_masks: List[SamMask],
    ) -> Tuple[List[ClassifiedMask], List[Dict[str, Any]]]:
        """Run RAP-like retrieval, create metadata, and track unknown objects."""
        classified: List[ClassifiedMask] = []
        track_records: List[Dict[str, Any]] = []
        label_to_id: Dict[str, int] = {"unknown_object": 1000}
        next_label_id = 1
        next_instance_id = 1

        for idx, mask in enumerate(sam_masks):
            rap = self.rap_backend.classify(rgb, mask, idx)
            is_known = bool(rap.is_known and rap.confidence >= self.config.rap_confidence_threshold)
            label = rap.label if is_known else "unknown_object"
            if label not in label_to_id:
                while next_label_id in label_to_id.values() or next_label_id == 1000:
                    next_label_id += 1
                label_to_id[label] = next_label_id
                next_label_id += 1
            label_id = label_to_id[label]
            candidate_id = self.make_candidate_id(frame, mask.mask_id, idx, is_known)
            metadata = self.build_object_metadata(
                frame=frame,
                mask=mask,
                depth=depth,
                tx=tx,
                rot_m=rot_m,
                label=label,
                label_id=label_id,
                instance_id=next_instance_id,
                confidence=float(rap.confidence),
                status="known_by_rap" if is_known else "unknown_pending_vlm",
                candidate_id=candidate_id,
                rap_metadata=rap.metadata,
            )
            if not is_known:
                timestamp_sec = stamp_to_float(frame.header.stamp)
                metadata, track_record = self.unknown_tracker.associate(
                    metadata=metadata,
                    frame_id=frame.rsg_frame_id,
                    sequence=int(frame.sequence),
                    timestamp_sec=timestamp_sec,
                )
                track_record.update({
                    "frame_id": frame.rsg_frame_id,
                    "sequence": int(frame.sequence),
                    "candidate_id": candidate_id,
                })
                track_records.append(track_record)
            classified.append(
                ClassifiedMask(
                    mask_id=mask.mask_id,
                    mask=mask.mask,
                    label=label,
                    label_id=int(label_id),
                    instance_id=int(next_instance_id),
                    confidence=float(rap.confidence),
                    status="known_by_rap" if is_known else "unknown_pending_vlm",
                    candidate_id=candidate_id,
                    metadata=metadata,
                )
            )
            next_instance_id += 1
        return classified, track_records

    def build_object_metadata(
        self,
        frame: RsgFrame,
        mask: SamMask,
        depth: np.ndarray,
        tx: np.ndarray,
        rot_m: np.ndarray,
        label: str,
        label_id: int,
        instance_id: int,
        confidence: float,
        status: str,
        candidate_id: str,
        rap_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create configurable object metadata used by Hydra/fusion/risk nodes."""
        geometry = self.geometry_estimator.estimate(mask.mask, depth, frame.camera_info, tx, rot_m)
        metadata = {
            "source_frame_id": frame.rsg_frame_id,
            "timestamp_sec": stamp_to_float(frame.header.stamp),
            "candidate_id": candidate_id,
            "mask_id": mask.mask_id,
            "label": label,
            "label_id": int(label_id),
            "instance_id": int(instance_id),
            "confidence": float(confidence),
            "status": status,
            "rap": rap_metadata,
            **geometry,
        }
        return filter_metadata(metadata, self.config)

    def dispatch_unknowns_to_vlm(
        self,
        frame: RsgFrame,
        rgb: np.ndarray,
        depth: np.ndarray,
        unknowns: List[Dict[str, Any]],
        classified: List[ClassifiedMask],
    ) -> List[Dict[str, Any]]:
        """Queue one VLM request per persistent unknown track.

        A frame may contain several unknown detections, and the same physical
        unknown may appear in many consecutive frames. The tracker assigns the
        persistent ``unknown_track_id`` and this function only queues the VLM
        task when the track is ready and has not already been queued/done.
        """
        dispatch_records: List[Dict[str, Any]] = []
        mask_lookup = {item.candidate_id: item for item in classified}
        image_area_px = int(rgb.shape[0] * rgb.shape[1]) if rgb.ndim >= 2 else None

        for unknown in unknowns:
            candidate_id = str(unknown.get("candidate_id", ""))
            classified_mask = mask_lookup.get(candidate_id)
            rgb_crop = self.extract_crop(rgb, unknown.get("bbox_2d"))

            if not self.config.vlm_enabled:
                dispatch_info = {"vlm_dispatch_status": "vlm_disabled", "best_frame_score": 0.0}
                task = None
            else:
                task, dispatch_info = self.unknown_tracker.update_evidence_and_build_vlm_task(
                    unknown=unknown,
                    rgb_crop=rgb_crop,
                    frame_header=frame.header,
                    frame_id=frame.rsg_frame_id,
                    sequence=int(frame.sequence),
                    image_area_px=image_area_px,
                )

            dispatch_status = str(dispatch_info.get("vlm_dispatch_status", "not_queued"))
            if task is not None:
                dispatch_status = self.enqueue_vlm_task(task, dispatch_status)

            record = {
                "candidate_id": candidate_id,
                "unknown_track_id": str(unknown.get("unknown_track_id", "")),
                "status": dispatch_status,
                "track_seen_count": int(unknown.get("track_seen_count", 1) or 1),
                "best_frame_score": float(dispatch_info.get("best_frame_score", 0.0) or 0.0),
            }
            dispatch_records.append(record)

            centroid = unknown.get("centroid_3d") if isinstance(unknown, dict) else None
            centroid_x = centroid_y = centroid_z = ""
            if isinstance(centroid, (list, tuple)) and len(centroid) == 3:
                centroid_x, centroid_y, centroid_z = centroid[0], centroid[1], centroid[2]
            self.csv_recorder.unknown_track_observation(
                sequence=int(frame.sequence),
                frame_id=frame.rsg_frame_id,
                candidate_id=candidate_id,
                unknown_track_id=str(unknown.get("unknown_track_id", "")),
                track_event=str(unknown.get("track_event", "")),
                track_seen_count=int(unknown.get("track_seen_count", 1) or 1),
                vlm_status=str(unknown.get("vlm_status", "")),
                vlm_dispatch_status=dispatch_status,
                vlm_queue_size=self.vlm_queue.qsize(),
                vlm_queue_max_size=self.config.vlm_queue_size,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                centroid_z=centroid_z,
                bbox_volume_m3=unknown.get("bbox_volume_m3", ""),
                depth_valid_ratio=unknown.get("depth_valid_ratio", ""),
                mask_area_px=unknown.get("mask_area_px", ""),
                best_frame_score=float(dispatch_info.get("best_frame_score", 0.0) or 0.0),
            )

        return dispatch_records

    def enqueue_vlm_task(self, task: Dict[str, Any], previous_status: str) -> str:
        """Insert a ready unknown-track task into the post-RAP FIFO VLM queue.

        The queue is intentionally after RAP and after persistent unknown-track
        association. Therefore, repeated detections of the same physical unknown
        object do not create repeated VLM jobs. Only one ready task per track is
        inserted, and the VLM worker consumes tasks in FIFO order.
        """
        track_id = str(task.get("unknown_track_id", ""))
        candidate_id = str(task.get("candidate_id", ""))
        frame_id = str(task.get("rsg_frame_id", ""))
        sequence = int(task.get("sequence", 0) or 0)
        task["enqueued_monotonic"] = time.perf_counter()

        try:
            self.vlm_queue.put_nowait(task)
            self.unknown_tracker.mark_vlm_queued(track_id)
            self.unknown_vlm_count += 1
            self.record_vlm_queue_event(
                event="enqueued",
                task=task,
                queue_wait_ms=0.0,
                reason=previous_status,
            )
            return "queued_for_vlm_fifo"
        except queue.Full:
            self.vlm_queue_dropped_count += 1
            self.unknown_tracker.mark_vlm_queue_rejected(track_id, reason="vlm_fifo_queue_full")
            self.record_vlm_queue_event(
                event="dropped_queue_full",
                task=task,
                queue_wait_ms=0.0,
                reason=self.config.vlm_queue_drop_policy,
            )
            return "vlm_fifo_queue_full_dropped"

    def record_vlm_queue_event(self, event: str, task: Dict[str, Any], queue_wait_ms: float = 0.0, reason: str = "") -> None:
        """Record a compact FIFO queue event for plotting and debugging."""
        if not self.config.timing_enabled:
            return
        self._vlm_queue_event_index += 1
        self.csv_recorder.vlm_queue_event(
            event_index=self._vlm_queue_event_index,
            event=event,
            sequence=int(task.get("sequence", 0) or 0),
            frame_id=str(task.get("rsg_frame_id", "")),
            unknown_track_id=str(task.get("unknown_track_id", "")),
            candidate_id=str(task.get("candidate_id", "")),
            queue_size=int(self.vlm_queue.qsize()),
            queue_max_size=int(self.config.vlm_queue_size),
            queue_wait_ms=float(queue_wait_ms),
            track_seen_count=int(task.get("track_seen_count", 0) or 0),
            best_frame_score=float(task.get("best_frame_score", 0.0) or 0.0),
            reason=reason,
        )

    def _vlm_loop(self) -> None:
        """Background loop for unknown-object VLM identification."""
        while not self._stop_event.is_set():
            try:
                task = self.vlm_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            start = time.perf_counter()
            queue_wait_ms = max(0.0, (start - float(task.get("enqueued_monotonic", start))) * 1000.0)
            self.record_vlm_queue_event(event="dequeued", task=task, queue_wait_ms=queue_wait_ms, reason="fifo_order")
            result = self.vlm_backend.identify(task.get("rgb_crop"), task.get("object_metadata", {}))
            rap_update_status = self.rap_memory_updater.update_from_vlm(result, task.get("object_metadata", {}))
            # In real RAP mode, immediately add the VLM-labelled crop back to
            # the VisualRAP memory as the baseline LearningWorker does.  The
            # JSONL log remains enabled as an audit trail.
            try:
                if self.config.rap_update_enabled and bool(result.get("success", False)) and float(result.get("confidence", 0.0)) >= float(self.config.rap_update_min_confidence):
                    label_for_rap = str(result.get("label", "")).replace("_", " ").strip()
                    if label_for_rap and hasattr(self.rap_backend, "add_image"):
                        self.rap_backend.add_image(task.get("rgb_crop"), label_for_rap)
                        rap_update_status["rap_memory_live_update"] = "added_to_visual_rap"
            except Exception as exc:
                rap_update_status["rap_memory_live_update"] = "failed"
                rap_update_status["live_update_error"] = str(exc)
            result["rap_update"] = rap_update_status
            vlm_delay_ms = (time.perf_counter() - start) * 1000.0
            total_age_ms = (time.perf_counter() - float(task.get("created_monotonic", start))) * 1000.0
            msg = Phase1VlmResult()
            msg.header = task["frame_header"]
            msg.rsg_frame_id = str(task["rsg_frame_id"])
            msg.sequence = int(task["sequence"])
            msg.candidate_id = str(task["candidate_id"])
            msg.unknown_track_id = str(task.get("unknown_track_id", ""))
            msg.mask_id = str(task["mask_id"])
            msg.success = bool(result.get("success", False))
            msg.status = "vlm_done" if msg.success else "vlm_failed"
            msg.reason = "ok" if msg.success else str(result.get("raw_response", "vlm_failed"))
            msg.predicted_label = str(result.get("label", "unknown_object"))
            msg.confidence = float(result.get("confidence", 0.0))
            msg.backend = str(result.get("backend", self.config.vlm_mode))
            msg.model = str(result.get("model", self.config.vlm_model))
            msg.vlm_delay_ms = float(vlm_delay_ms)
            msg.total_age_ms = float(total_age_ms)
            msg.object_metadata_json = safe_json_dumps(task.get("object_metadata", {}))
            result["unknown_track_id"] = msg.unknown_track_id
            result["track_seen_count"] = int(task.get("track_seen_count", 0))
            result["best_frame_score"] = float(task.get("best_frame_score", 0.0))
            msg.vlm_metadata_json = safe_json_dumps(result)
            self.unknown_tracker.mark_vlm_result(msg.unknown_track_id, result)
            self.vlm_result_pub.publish(msg)
            self.publish_vlm_timing(msg)
            self.record_vlm_queue_event(event="completed", task=task, queue_wait_ms=queue_wait_ms, reason=msg.status)

    @staticmethod
    def extract_crop(rgb: np.ndarray, bbox_2d: Any) -> Optional[np.ndarray]:
        """Extract an RGB crop from [x, y, w, h] metadata."""
        if not bbox_2d or len(bbox_2d) != 4:
            return None
        x, y, w, h = [int(v) for v in bbox_2d]
        height, width = rgb.shape[:2]
        x0 = max(0, min(width, x))
        y0 = max(0, min(height, y))
        x1 = max(0, min(width, x + max(1, w)))
        y1 = max(0, min(height, y + max(1, h)))
        if x1 <= x0 or y1 <= y0:
            return None
        return rgb[y0:y1, x0:x1].copy()

    @staticmethod
    def make_candidate_id(frame: RsgFrame, mask_id: str, index: int, is_known: bool) -> str:
        prefix = "rsg_known" if is_known else "rsg_unknown"
        safe_frame_id = frame.rsg_frame_id.replace("/", "_")
        return f"{prefix}_{safe_frame_id}_{index:03d}_{mask_id}"

    def build_result_metadata(self, frame: RsgFrame, masks: List[SamMask], objects: List[Dict[str, Any]], unknowns: List[Dict[str, Any]], vlm_dispatch: List[Dict[str, Any]], track_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        queued = [item for item in vlm_dispatch if str(item.get("status", "")).startswith("queued_for_vlm")]
        metadata = {
            "phase": "phase1_object_classification",
            "node": "rsg_object_detection",
            "sam_backend": self.config.sam_backend,
            "rap_backend": self.config.rap_backend,
            "vlm_enabled": self.config.vlm_enabled,
            "vlm_mode": self.config.vlm_mode,
            "num_masks": len(masks),
            "num_objects": len(objects),
            "num_unknown": len(unknowns),
            "num_unknown_tracks": len({str(item.get("unknown_track_id", "")) for item in unknowns if item.get("unknown_track_id")}),
            "num_new_tracks": len([item for item in track_records if item.get("track_event") == "new_track"]),
            "num_matched_tracks": len([item for item in track_records if item.get("track_event") == "matched_existing_track"]),
            "num_vlm_queued": len(queued),
            "vlm_dispatch": vlm_dispatch,
            "unknown_track_records": track_records,
            "source_preprocessor_metadata": safe_json_loads(frame.metadata_json, default={}),
        }
        return metadata

    def build_failed_result(self, frame: RsgFrame, reason: str) -> Phase1ClassificationResult:
        """Build a failure result with empty label maps so downstream nodes can log it."""
        result = Phase1ClassificationResult()
        result.header = frame.header
        result.rsg_frame_id = frame.rsg_frame_id
        result.sequence = frame.sequence
        result.success = False
        result.status = "failed"
        result.reason = reason
        height = int(frame.rgb.height) if frame.rgb.height else 1
        width = int(frame.rgb.width) if frame.rgb.width else 1
        empty = np.zeros((height, width), dtype=np.uint16)
        result.semantic_labels = self.bridge.cv2_to_imgmsg(empty, encoding=self.config.semantic_label_encoding)
        result.semantic_labels.header = frame.header
        result.instance_labels = self.bridge.cv2_to_imgmsg(empty, encoding=self.config.instance_label_encoding)
        result.instance_labels.header = frame.header
        result.label_table_json = safe_json_dumps({"0": "background"})
        result.object_metadata_json = "[]"
        result.unknown_candidates_json = "[]"
        result.vlm_dispatch_json = "[]"
        result.metadata_json = safe_json_dumps({"phase": "phase1_object_classification", "status": "failed", "reason": reason})
        result.image_conversion_delay_ms = 0.0
        result.result_message_build_delay_ms = 0.0
        result.classifier_debug_record_delay_ms = 0.0
        return result

    def publish_status(self, status: str, frame_id: str, reason: str) -> None:
        if not self.config.publish_status:
            return
        payload = {
            "node": "rsg_object_detection",
            "status": status,
            "reason": reason,
            "frame_id": frame_id,
            "received": self.received_count,
            "processed": self.processed_count,
            "failed": self.failed_count,
            "dropped": self.dropped_count,
            "vlm_queued": self.unknown_vlm_count,
            "vlm_queue_dropped": self.vlm_queue_dropped_count,
            "vlm_fifo_queue_size": self.vlm_queue.qsize(),
            "vlm_fifo_queue_max_size": self.config.vlm_queue_size,
        }
        self.status_pub.publish(String(data=safe_json_dumps(payload)))

    def publish_timing_event(self, result: Phase1ClassificationResult, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record one simple classifier phase-latency row per processed frame."""
        if not self.config.timing_enabled:
            return
        metadata = metadata or {}
        if self.timing_pub is not None:
            msg = Float64MultiArray()
            msg.data = [
                float(result.sequence),
                float(result.classifier_delay_ms),
                float(result.sam_delay_ms),
                float(result.rap_delay_ms),
                float(result.num_unknown),
            ]
            self.timing_pub.publish(msg)

        self.csv_recorder.classifier_latency(
            sequence=int(result.sequence),
            frame_id=result.rsg_frame_id,
            status=result.status,
            input_age_ms=float(result.input_age_ms),
            sam_delay_ms=float(result.sam_delay_ms),
            rap_delay_ms=float(result.rap_delay_ms),
            label_map_delay_ms=float(result.label_map_delay_ms),
            metadata_delay_ms=float(result.metadata_delay_ms),
            image_conversion_delay_ms=float(getattr(result, "image_conversion_delay_ms", 0.0)),
            result_message_build_delay_ms=float(getattr(result, "result_message_build_delay_ms", 0.0)),
            classifier_debug_record_delay_ms=float(getattr(result, "classifier_debug_record_delay_ms", 0.0)),
            classifier_delay_ms=float(result.classifier_delay_ms),
            num_masks=int(result.num_masks),
            num_known=int(result.num_known),
            num_unknown=int(result.num_unknown),
            num_new_tracks=int(metadata.get("num_new_tracks", 0)),
            num_matched_tracks=int(metadata.get("num_matched_tracks", 0)),
            num_vlm_queued=int(metadata.get("num_vlm_queued", 0)),
        )
        self.timing_recorder.add_sample(
            node="rsg_object_detection",
            sequence=int(result.sequence),
            frame_id=result.rsg_frame_id,
            status=result.status,
            reason=result.reason,
            input_age_ms=float(result.input_age_ms),
            classifier_delay_ms=float(result.classifier_delay_ms),
            sam_delay_ms=float(result.sam_delay_ms),
            rap_delay_ms=float(result.rap_delay_ms),
            label_map_delay_ms=float(result.label_map_delay_ms),
            metadata_delay_ms=float(result.metadata_delay_ms),
            image_conversion_delay_ms=float(getattr(result, "image_conversion_delay_ms", 0.0)),
            result_message_build_delay_ms=float(getattr(result, "result_message_build_delay_ms", 0.0)),
            classifier_debug_record_delay_ms=float(getattr(result, "classifier_debug_record_delay_ms", 0.0)),
            num_masks=int(result.num_masks),
            num_known=int(result.num_known),
            num_unknown=int(result.num_unknown),
            num_new_tracks=int(metadata.get("num_new_tracks", 0)),
            num_matched_tracks=int(metadata.get("num_matched_tracks", 0)),
            num_vlm_queued=int(metadata.get("num_vlm_queued", 0)),
        )

    def publish_vlm_timing(self, result: Phase1VlmResult) -> None:
        """Record one simple VLM-latency row per persistent unknown track."""
        if not self.config.timing_enabled:
            return
        vlm_meta = safe_json_loads(result.vlm_metadata_json, default={})
        self.csv_recorder.vlm_latency(
            unknown_track_id=result.unknown_track_id,
            candidate_id=result.candidate_id,
            frame_id=result.rsg_frame_id,
            sequence=int(result.sequence),
            status=result.status,
            predicted_label=result.predicted_label,
            confidence=float(result.confidence),
            vlm_delay_ms=float(result.vlm_delay_ms),
            total_age_ms=float(result.total_age_ms),
            track_seen_count=int(vlm_meta.get("track_seen_count", 0) or 0),
            best_frame_score=float(vlm_meta.get("best_frame_score", 0.0) or 0.0),
            backend=result.backend,
            model=result.model,
        )
        self.timing_recorder.add_sample(
            node="rsg_object_detection",
            sequence=int(result.sequence),
            frame_id=result.rsg_frame_id,
            status=result.status,
            reason=result.reason,
            candidate_id=result.candidate_id,
            unknown_track_id=result.unknown_track_id,
            predicted_label=result.predicted_label,
            vlm_delay_ms=float(result.vlm_delay_ms),
            total_age_ms=float(result.total_age_ms),
            track_seen_count=int(vlm_meta.get("track_seen_count", 0) or 0),
            best_frame_score=float(vlm_meta.get("best_frame_score", 0.0) or 0.0),
            backend=result.backend,
            model=result.model,
        )


    def publish_status(self, status: str, frame_id: str, reason: str) -> None:
        if not self.config.publish_status:
            return
        payload = {
            "node": "rsg_object_detection",
            "status": status,
            "reason": reason,
            "frame_id": frame_id,
            "received": self.received_count,
            "processed": self.processed_count,
            "failed": self.failed_count,
            "dropped": self.dropped_count,
            "hydra_published": self.hydra_published_count,
            "frame_fifo_size": self.frame_fifo.qsize(),
            "frame_fifo_max_size": self.config.request_queue_size,
            "vlm_queued": self.unknown_vlm_count,
            "vlm_queue_dropped": self.vlm_queue_dropped_count,
            "vlm_fifo_queue_size": self.vlm_queue.qsize(),
            "vlm_fifo_queue_max_size": self.config.vlm_queue_size,
        }
        self.status_pub.publish(String(data=safe_json_dumps(payload)))

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._classification_thread.is_alive():
            self._classification_thread.join(timeout=1.0)
        if self._vlm_thread.is_alive():
            self._vlm_thread.join(timeout=1.0)
        self.timing_recorder.save()
        self.csv_recorder.close()
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = RSGObjectDetection()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
