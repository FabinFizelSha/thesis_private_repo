"""Configuration loader for Phase 1 RSG nodes.

The Phase 1 nodes use the same central ``rsg_pipeline.yaml`` file as the
preprocessor.  This keeps runtime switches, topic names, debug controls, and
performance settings in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _as_bool(value: Any, default: bool = False) -> bool:
    """Return a robust bool from YAML values."""
    if value is None:
        return default
    return bool(value)


@dataclass
class Phase1Config:
    """Configuration values shared by ``rsg_object_detection`` and classifier."""

    node_key: str
    use_sim_time: bool

    # Topics.
    preprocessed_frame_topic: str
    perception_request_topic: str
    perception_result_topic: str
    vlm_result_topic: str
    hydra_frame_topic: str
    status_topic: str
    timing_topic: str
    unknown_candidates_topic: str
    debug_semantic_topic: str
    debug_instance_topic: str
    hydra_rgb_topic: str
    hydra_depth_topic: str
    hydra_camera_info_topic: str
    hydra_pose_topic: str
    hydra_semantic_topic: str
    hydra_instance_topic: str
    hydra_metadata_topic: str

    # QoS / queueing.
    input_qos_depth: int
    output_qos_depth: int
    request_queue_size: int
    classifier_queue_size: int
    frame_cache_size: int
    max_result_age_sec: float
    drop_oldest_when_full: bool

    # Debug / timing.
    global_debug_enabled: bool
    node_debug_enabled: bool
    timing_measurement_enabled: bool
    publish_timing_topic: bool
    write_timing_excel: bool
    timing_excel_path: str
    timing_sheet_name: str
    timing_excel_autosave_every: int
    publish_status: bool
    status_every_n_frames: int
    publish_debug_label_images: bool
    write_simple_csv: bool
    csv_debug_dir: str
    generate_debug_plots_on_shutdown: bool

    # Hydra output.
    publish_hydra_combined: bool
    publish_hydra_separate_topics: bool
    semantic_label_encoding: str
    instance_label_encoding: str

    # Metadata switches. Turning these off reduces JSON size and processing.
    include_object_metadata: bool
    include_known_objects: bool
    include_unknown_objects: bool
    include_bbox_2d: bool
    include_centroid_2d: bool
    include_centroid_3d: bool
    include_bbox_3d: bool
    include_bbox_volume: bool
    include_depth_stats: bool
    include_mask_area: bool
    include_timing_metadata: bool
    include_frame_relation_metadata: bool

    # SAM / RAP / VLM settings.
    sam_enabled: bool
    sam_backend: str
    sam_min_mask_pixels: int
    sam_max_masks: int
    sam_dummy_num_masks: int

    rap_enabled: bool
    rap_backend: str
    rap_confidence_threshold: float
    rap_distance_threshold: float
    rap_dummy_unknown_every_n: int
    rap_dummy_known_labels: List[str] = field(default_factory=list)
    rap_update_enabled: bool = True
    rap_update_min_confidence: float = 0.50
    rap_memory_path: str = "~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl"

    vlm_enabled: bool = False
    vlm_mode: str = "dummy"
    vlm_async: bool = True
    # FIFO queue after RAP/unknown tracking. Ready unknown tracks enter this
    # queue and are consumed by the VLM worker without blocking SAM/RAP/Hydra.
    vlm_queue_size: int = 16
    vlm_queue_drop_policy: str = "drop_newest"
    vlm_dummy_delay_sec: float = 0.0
    vlm_dummy_label_prefix: str = "vlm_dummy_object"
    vlm_confidence: float = 0.55
    vlm_endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    vlm_model: str = "Qwen2.5-VL-7B-Instruct"
    vlm_timeout_sec: float = 30.0
    vlm_prompt: str = "Identify the main object in this image crop. Return only a short object label."

    # Projection / object geometry.
    estimate_object_geometry: bool = True
    projection_stride: int = 4
    min_valid_depth_points: int = 20
    min_depth_m: float = 0.2
    max_depth_m: float = 6.0
    centroid_method: str = "median"

    # Evidence buffer for future risk annotation.
    store_evidence_frames: bool = True
    evidence_buffer_size: int = 50

    # Persistent unknown-object tracking. This prevents repeated VLM calls for
    # the same physical unknown object across consecutive frames.
    unknown_tracking_enabled: bool = True
    unknown_max_match_distance_m: float = 0.30
    unknown_max_volume_ratio: float = 3.0
    unknown_max_track_age_sec: float = 2.0
    unknown_min_observations_before_vlm: int = 3
    unknown_max_wait_before_vlm_sec: float = 0.75
    unknown_call_vlm_only_once_per_track: bool = True
    unknown_retry_failed_tracks: bool = False
    unknown_update_track_centroid: bool = True
    unknown_centroid_update_alpha: float = 0.7
    unknown_best_frame_selection_enabled: bool = True
    unknown_min_quality_for_vlm: float = 0.0
    unknown_max_tracks: int = 200
    unknown_use_2d_iou_fallback: bool = True
    unknown_min_2d_iou: float = 0.30

    @property
    def debug_enabled(self) -> bool:
        """Return whether global or node-specific debug mode is enabled."""
        return self.global_debug_enabled or self.node_debug_enabled

    @property
    def timing_enabled(self) -> bool:
        """Return whether timing measurement is enabled for this node."""
        return self.debug_enabled and self.timing_measurement_enabled

    @staticmethod
    def from_yaml(path: str, node_key: str) -> "Phase1Config":
        """Load Phase 1 configuration from the central YAML file."""
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as stream:
            root = yaml.safe_load(stream) or {}

        root_debug = root.get("debug", {}) or {}
        preprocessing = root.get("preprocessing", {}) or {}
        phase1 = root.get("phase1", {}) or {}

        runtime = phase1.get("runtime", {}) or {}
        topics = phase1.get("topics", {}) or {}
        qos = phase1.get("qos", {}) or {}
        coordinator = phase1.get("coordinator", {}) or {}
        classifier = phase1.get("object_classifier", {}) or {}
        hydra = phase1.get("hydra_output", {}) or {}
        metadata = phase1.get("metadata", {}) or {}
        sam = phase1.get("sam", {}) or {}
        rap = phase1.get("rap", {}) or {}
        vlm = phase1.get("vlm", {}) or {}
        geometry = phase1.get("object_geometry", {}) or {}
        evidence = phase1.get("evidence_buffer", {}) or {}
        unknown_tracking = phase1.get("unknown_tracking", {}) or {}
        debug = {**(phase1.get("debug", {}) or {}), **root_debug}
        performance = phase1.get("performance", {}) or {}

        creation_time = datetime.now().strftime("%H%M%S")
        session_date = datetime.now().strftime("%Y%m%d")
        timing_excel_path = str(
            performance.get(
                "timing_excel_path",
                f"~/rsg_ros2_ws/debug/{node_key}_debug_{{session_date}}_{{creation_time}}.xlsx",
            )
        ).format(session_date=session_date, creation_time=creation_time, node_name=node_key)
        csv_debug_dir = str(
            performance.get(
                "csv_debug_dir",
                "~/rsg_ros2_ws/debug/phase1_csv_current",
            )
        ).format(session_date=session_date, creation_time=creation_time, node_name=node_key)

        global_debug_enabled = bool(debug.get("global_debug", False) or debug.get("enabled", False))
        node_debug_enabled = bool(debug.get(node_key, False) or debug.get(node_key.replace("rsg_", ""), False))

        preproc_topics = preprocessing.get("topics", {}) or {}
        preproc_runtime = preprocessing.get("runtime", {}) or {}
        preproc_image = preprocessing.get("image", {}) or {}

        return Phase1Config(
            node_key=node_key,
            use_sim_time=bool(runtime.get("use_sim_time", preproc_runtime.get("use_sim_time", True))),
            preprocessed_frame_topic=str(topics.get("preprocessed_frame", preproc_topics.get("prepared_frame", "/rsg/preprocessed/frame"))),
            perception_request_topic=str(topics.get("perception_request", "/rsg/phase1/object_classifier/input")),
            perception_result_topic=str(topics.get("perception_result", "/rsg/phase1/object_classifier/result")),
            vlm_result_topic=str(topics.get("vlm_result", "/rsg/phase1/object_classifier/vlm_result")),
            hydra_frame_topic=str(topics.get("hydra_frame", "/rsg/phase1/hydra/input_frame")),
            status_topic=str(topics.get(f"{node_key}_status", f"/rsg/phase1/{node_key}/status")),
            timing_topic=str(topics.get(f"{node_key}_timing", f"/rsg/phase1/{node_key}/timing")),
            unknown_candidates_topic=str(topics.get("unknown_candidates", "/rsg/phase1/unknown_candidates")),
            debug_semantic_topic=str(topics.get("debug_semantic_labels", "/rsg/phase1/debug/semantic_labels")),
            debug_instance_topic=str(topics.get("debug_instance_labels", "/rsg/phase1/debug/instance_labels")),
            hydra_rgb_topic=str(topics.get("hydra_rgb", "/rsg/hydra/rgb")),
            hydra_depth_topic=str(topics.get("hydra_depth", "/rsg/hydra/depth")),
            hydra_camera_info_topic=str(topics.get("hydra_camera_info", "/rsg/hydra/camera_info")),
            hydra_pose_topic=str(topics.get("hydra_pose", "/rsg/hydra/pose")),
            hydra_semantic_topic=str(topics.get("hydra_semantic_labels", "/rsg/hydra/semantic_labels")),
            hydra_instance_topic=str(topics.get("hydra_instance_labels", "/rsg/hydra/instance_labels")),
            hydra_metadata_topic=str(topics.get("hydra_metadata", "/rsg/hydra/metadata")),
            input_qos_depth=int(qos.get("input_depth", 10)),
            output_qos_depth=int(qos.get("output_depth", 10)),
            request_queue_size=max(1, int(coordinator.get("request_queue_size", 2))),
            classifier_queue_size=max(1, int(classifier.get("queue_size", 1))),
            frame_cache_size=max(1, int(coordinator.get("frame_cache_size", 75))),
            max_result_age_sec=float(coordinator.get("max_result_age_sec", 1.0)),
            drop_oldest_when_full=bool(coordinator.get("drop_oldest_when_full", True)),
            global_debug_enabled=global_debug_enabled,
            node_debug_enabled=node_debug_enabled,
            timing_measurement_enabled=bool(performance.get("measure_timing", True)),
            publish_timing_topic=bool(performance.get("publish_timing", True)),
            write_timing_excel=bool(performance.get("write_timing_excel", True)),
            timing_excel_path=timing_excel_path,
            timing_sheet_name=str(performance.get("timing_sheet_name", node_key[:31])),
            timing_excel_autosave_every=int(performance.get("timing_excel_autosave_every", 0)),
            publish_status=bool(coordinator.get("publish_status", True)),
            status_every_n_frames=max(1, int(coordinator.get("status_every_n_frames", 30))),
            publish_debug_label_images=bool(coordinator.get("publish_debug_label_images", False)),
            write_simple_csv=bool(performance.get("write_simple_csv", True)),
            csv_debug_dir=csv_debug_dir,
            generate_debug_plots_on_shutdown=bool(performance.get("generate_plots_on_shutdown", False)),
            publish_hydra_combined=bool(hydra.get("publish_combined", True)),
            publish_hydra_separate_topics=bool(hydra.get("publish_separate_topics", False)),
            semantic_label_encoding=str(hydra.get("semantic_label_encoding", "16UC1")),
            instance_label_encoding=str(hydra.get("instance_label_encoding", "16UC1")),
            include_object_metadata=bool(metadata.get("include_object_metadata", True)),
            include_known_objects=bool(metadata.get("include_known_objects", True)),
            include_unknown_objects=bool(metadata.get("include_unknown_objects", True)),
            include_bbox_2d=bool(metadata.get("include_bbox_2d", True)),
            include_centroid_2d=bool(metadata.get("include_centroid_2d", True)),
            include_centroid_3d=bool(metadata.get("include_centroid_3d", True)),
            include_bbox_3d=bool(metadata.get("include_bbox_3d", True)),
            include_bbox_volume=bool(metadata.get("include_bbox_volume", True)),
            include_depth_stats=bool(metadata.get("include_depth_stats", True)),
            include_mask_area=bool(metadata.get("include_mask_area", True)),
            include_timing_metadata=bool(metadata.get("include_timing_metadata", True)),
            include_frame_relation_metadata=bool(metadata.get("include_frame_relation_metadata", True)),
            sam_enabled=bool(sam.get("enabled", True)),
            sam_backend=str(sam.get("backend", "dummy")),
            sam_min_mask_pixels=max(1, int(sam.get("min_mask_pixels", 500))),
            sam_max_masks=max(1, int(sam.get("max_masks", 64))),
            sam_dummy_num_masks=max(0, int(sam.get("dummy_num_masks", 1))),
            rap_enabled=bool(rap.get("enabled", True)),
            rap_backend=str(rap.get("backend", "dummy")),
            rap_confidence_threshold=float(rap.get("confidence_threshold", 0.30)),
            rap_distance_threshold=float(rap.get("distance_threshold", 0.30)),
            rap_dummy_unknown_every_n=max(1, int(rap.get("dummy_unknown_every_n", 1))),
            rap_dummy_known_labels=list(rap.get("dummy_known_labels", ["chair", "table", "box"])),
            rap_update_enabled=bool(rap.get("update_memory_from_vlm", True)),
            rap_update_min_confidence=float(rap.get("update_min_confidence", 0.50)),
            rap_memory_path=str(rap.get("memory_update_path", "~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl")),
            vlm_enabled=bool(vlm.get("enabled", True)),
            vlm_mode=str(vlm.get("mode", "dummy")),
            vlm_async=bool(vlm.get("async", True)),
            vlm_queue_size=max(1, int(vlm.get("queue_size", 16))),
            vlm_queue_drop_policy=str(vlm.get("queue_drop_policy", "drop_newest")),
            vlm_dummy_delay_sec=float(vlm.get("dummy_delay_sec", 0.0)),
            vlm_dummy_label_prefix=str(vlm.get("dummy_label_prefix", "vlm_dummy_object")),
            vlm_confidence=float(vlm.get("dummy_confidence", vlm.get("confidence", 0.55))),
            vlm_endpoint=str(vlm.get("endpoint", "http://127.0.0.1:8000/v1/chat/completions")),
            vlm_model=str(vlm.get("model", "Qwen2.5-VL-7B-Instruct")),
            vlm_timeout_sec=float(vlm.get("timeout_sec", 30.0)),
            vlm_prompt=str(vlm.get("prompt", "Identify the main object in this image crop. Return only a short object label.")),
            estimate_object_geometry=bool(geometry.get("enabled", True)),
            projection_stride=max(1, int(geometry.get("projection_stride", 4))),
            min_valid_depth_points=max(1, int(geometry.get("min_valid_depth_points", 20))),
            min_depth_m=float(geometry.get("min_depth_m", preproc_image.get("min_depth_m", 0.2))),
            max_depth_m=float(geometry.get("max_depth_m", preproc_image.get("max_depth_m", 6.0))),
            centroid_method=str(geometry.get("centroid_method", "median")),
            store_evidence_frames=bool(evidence.get("enabled", True)),
            evidence_buffer_size=max(1, int(evidence.get("max_frames", 50))),
            unknown_tracking_enabled=bool(unknown_tracking.get("enabled", True)),
            unknown_max_match_distance_m=float(unknown_tracking.get("max_match_distance_m", 0.30)),
            unknown_max_volume_ratio=float(unknown_tracking.get("max_volume_ratio", 3.0)),
            unknown_max_track_age_sec=float(unknown_tracking.get("max_track_age_sec", 2.0)),
            unknown_min_observations_before_vlm=max(1, int(unknown_tracking.get("min_observations_before_vlm", 3))),
            unknown_max_wait_before_vlm_sec=float(unknown_tracking.get("max_wait_before_vlm_sec", 0.75)),
            unknown_call_vlm_only_once_per_track=bool(unknown_tracking.get("call_vlm_only_once_per_track", True)),
            unknown_retry_failed_tracks=bool(unknown_tracking.get("retry_failed_tracks", False)),
            unknown_update_track_centroid=bool(unknown_tracking.get("update_track_centroid", True)),
            unknown_centroid_update_alpha=float(unknown_tracking.get("centroid_update_alpha", 0.7)),
            unknown_best_frame_selection_enabled=bool(unknown_tracking.get("best_frame_selection", True)),
            unknown_min_quality_for_vlm=float(unknown_tracking.get("min_quality_for_vlm", 0.0)),
            unknown_max_tracks=max(1, int(unknown_tracking.get("max_tracks", 200))),
            unknown_use_2d_iou_fallback=bool(unknown_tracking.get("use_2d_iou_fallback", True)),
            unknown_min_2d_iou=float(unknown_tracking.get("min_2d_iou", 0.30)),
        )
