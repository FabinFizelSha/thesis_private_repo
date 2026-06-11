"""Configuration loader for the RSG preprocessing node."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import yaml


@dataclass
class PreprocessorConfig:
    """Configuration values required by the preprocessing node.

    The configuration is loaded from the central YAML pipeline file. The class
    keeps old YAML keys working while also supporting explicit validation/check
    switches under ``preprocessing.validation``.
    """

    source: str
    use_sim_time: bool

    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    odom_topic: str
    imu_topic: str
    frame_topic: str
    status_topic: str

    rgb_depth_slop_sec: float
    odom_buffer_size: int
    odom_tolerance_sec: float
    use_odom_interpolation: bool
    reject_unsynchronized_frames: bool
    assume_ordered_messages: bool

    use_camera_imu: bool
    imu_buffer_size: int
    imu_tolerance_sec: float
    require_imu_for_frame: bool

    output_rgb_encoding: str
    output_depth_encoding: str
    depth_scale_to_meter: float
    min_depth_m: float
    max_depth_m: float
    max_invalid_depth_ratio: float
    reject_resolution_mismatch: bool

    # Explicit configurable checks / fast-mode switches.
    check_resolution: bool
    convert_rgb: bool
    convert_depth: bool
    compute_invalid_depth_ratio: bool
    check_invalid_depth_ratio: bool
    depth_check_stride: int
    require_odom_for_frame: bool
    publish_status_for_published_frames: bool
    publish_status_for_rejected_frames: bool
    publish_published_status_every_n_frames: int

    world_frame: str
    base_frame: str
    camera_frame: str
    base_to_camera_translation: Tuple[float, float, float]
    base_to_camera_rpy: Tuple[float, float, float]

    frame_id_prefix: str
    session_date: str

    sensor_qos_depth: int
    odom_qos_depth: int
    output_qos_depth: int

    global_debug_enabled: bool
    node_debug_enabled: bool
    timing_measurement_enabled: bool
    publish_timing_topic: bool
    timing_topic: str
    write_timing_excel: bool
    timing_excel_path: str
    timing_sheet_name: str
    timing_excel_autosave_every: int

    # Cooperative latency guard. This guards the preprocessor hot path and
    # prevents frames that have already exceeded the allowed preprocessing
    # budget from being published to downstream workers.
    latency_guard_enabled: bool
    latency_warn_delay_ms: float
    latency_drop_delay_ms: float
    latency_drop_exceeded_frames: bool
    latency_record_dropped_to_excel: bool

    @property
    def debug_enabled(self) -> bool:
        """Return whether global or node-specific debug mode is enabled."""
        return self.global_debug_enabled or self.node_debug_enabled

    @property
    def timing_enabled(self) -> bool:
        """Return whether preprocessing timing measurement should run."""
        return self.debug_enabled and self.timing_measurement_enabled

    @staticmethod
    def from_yaml(path: str) -> "PreprocessorConfig":
        """Load preprocessing configuration from a YAML file."""
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as stream:
            root = yaml.safe_load(stream)

        preprocessing = root["preprocessing"]
        root_debug = root.get("debug", {}) or {}
        preprocessing_debug = preprocessing.get("debug", {}) or {}
        debug = {**preprocessing_debug, **root_debug}
        runtime = preprocessing.get("runtime", {}) or {}
        topics = preprocessing.get("topics", {}) or {}
        sync = preprocessing.get("synchronization", {}) or {}
        imu = preprocessing.get("imu", {}) or {}
        image = preprocessing.get("image", {}) or {}
        validation = preprocessing.get("validation", {}) or {}
        status_output = preprocessing.get("status_output", {}) or {}
        frames = preprocessing.get("frames", {}) or {}
        frame_id = preprocessing.get("frame_id", {}) or {}
        qos = preprocessing.get("qos", {}) or {}
        performance = preprocessing.get("performance", {}) or {}
        latency_guard = preprocessing.get("latency_guard", {}) or {}

        base_to_camera = frames.get("base_to_camera", {}) or {}
        translation = tuple(float(v) for v in base_to_camera.get("translation_m", [0.0, 0.0, 0.0]))
        rotation_rpy = tuple(float(v) for v in base_to_camera.get("rotation_rpy_rad", [0.0, 0.0, 0.0]))

        session_date = str(frame_id.get("session_date", ""))
        if not session_date:
            session_date = datetime.now().strftime("%Y%m%d")

        global_debug_enabled = bool(debug.get("global_debug", False) or debug.get("enabled", False))
        node_debug_enabled = bool(
            debug.get("RSG_pre_processor", False)
            or debug.get("rsg_pre_processor", False)
            or debug.get("preprocessor", False)
        )

        creation_time = datetime.now().strftime("%H%M%S")
        timing_excel_path = str(
            performance.get(
                "timing_excel_path",
                "~/rsg_ros2_ws/debug/RSG_pre_processor_debug_{session_date}_{creation_time}.xlsx",
            )
        ).format(session_date=session_date, creation_time=creation_time)

        # Backward compatible defaults: old keys still work. New validation
        # keys let every rejection check be switched on/off independently.
        check_resolution = bool(validation.get("check_resolution", image.get("reject_resolution_mismatch", True)))
        check_invalid_depth_ratio = bool(validation.get("check_invalid_depth_ratio", True))
        compute_invalid_depth_ratio = bool(validation.get("compute_invalid_depth_ratio", True))
        require_odom_for_frame = bool(validation.get("require_odom_for_frame", True))
        require_imu_for_frame = bool(validation.get("require_imu_for_frame", imu.get("require_for_frame", False)))

        return PreprocessorConfig(
            source=str(runtime.get("source", "rosbag")),
            use_sim_time=bool(runtime.get("use_sim_time", True)),
            rgb_topic=str(topics.get("rgb", "/go1/d455/color/image_raw")),
            depth_topic=str(topics.get("depth", "/go1/d455/aligned_depth_to_color/image_raw")),
            camera_info_topic=str(topics.get("camera_info", "/go1/d455/color/camera_info")),
            odom_topic=str(topics.get("odom", "/odom")),
            imu_topic=str(topics.get("imu", imu.get("topic", "/go1/d455/imu"))),
            frame_topic=str(topics.get("prepared_frame", "/rsg/preprocessed/frame")),
            status_topic=str(topics.get("status", "/rsg/preprocessor/status")),
            rgb_depth_slop_sec=float(sync.get("rgb_depth_slop_sec", 0.03)),
            odom_buffer_size=int(sync.get("odom_buffer_size", 300)),
            odom_tolerance_sec=float(sync.get("odom_tolerance_sec", 0.05)),
            use_odom_interpolation=bool(sync.get("use_odom_interpolation", True)),
            reject_unsynchronized_frames=bool(sync.get("reject_unsynchronized_frames", True)),
            assume_ordered_messages=bool(sync.get("assume_ordered_messages", True)),
            use_camera_imu=bool(imu.get("enabled", False)),
            imu_buffer_size=int(imu.get("buffer_size", 1000)),
            imu_tolerance_sec=float(imu.get("tolerance_sec", 0.03)),
            require_imu_for_frame=require_imu_for_frame,
            output_rgb_encoding=str(image.get("output_rgb_encoding", "rgb8")),
            output_depth_encoding=str(image.get("output_depth_encoding", "32FC1")),
            depth_scale_to_meter=float(image.get("depth_scale_to_meter", 0.001)),
            min_depth_m=float(image.get("min_depth_m", 0.2)),
            max_depth_m=float(image.get("max_depth_m", 6.0)),
            max_invalid_depth_ratio=float(image.get("max_invalid_depth_ratio", 0.70)),
            reject_resolution_mismatch=check_resolution,
            check_resolution=check_resolution,
            convert_rgb=bool(validation.get("convert_rgb", True)),
            convert_depth=bool(validation.get("convert_depth", True)),
            compute_invalid_depth_ratio=compute_invalid_depth_ratio,
            check_invalid_depth_ratio=check_invalid_depth_ratio,
            depth_check_stride=max(1, int(validation.get("depth_check_stride", 1))),
            require_odom_for_frame=require_odom_for_frame,
            publish_status_for_published_frames=bool(status_output.get("publish_published", True)),
            publish_status_for_rejected_frames=bool(status_output.get("publish_rejected", True)),
            publish_published_status_every_n_frames=max(1, int(status_output.get("published_every_n_frames", 1))),
            world_frame=str(frames.get("world_frame", "odom")),
            base_frame=str(frames.get("base_frame", "base_link")),
            camera_frame=str(frames.get("camera_frame", "d455_color_optical_frame")),
            base_to_camera_translation=translation,
            base_to_camera_rpy=rotation_rpy,
            frame_id_prefix=str(frame_id.get("prefix", "rsg")),
            session_date=session_date,
            sensor_qos_depth=int(qos.get("sensor_depth", 10)),
            odom_qos_depth=int(qos.get("odom_depth", 200)),
            output_qos_depth=int(qos.get("output_depth", 10)),
            global_debug_enabled=global_debug_enabled,
            node_debug_enabled=node_debug_enabled,
            timing_measurement_enabled=bool(performance.get("measure_timing", True)),
            publish_timing_topic=bool(performance.get("publish_timing", True)),
            timing_topic=str(performance.get("timing_topic", "/rsg/preprocessor/timing")),
            write_timing_excel=bool(performance.get("write_timing_excel", True)),
            timing_excel_path=timing_excel_path,
            timing_sheet_name=str(performance.get("timing_sheet_name", "timing_debug")),
            timing_excel_autosave_every=int(performance.get("timing_excel_autosave_every", 0)),
            latency_guard_enabled=bool(latency_guard.get("enabled", True)),
            latency_warn_delay_ms=float(latency_guard.get("warn_delay_ms", 40.0)),
            latency_drop_delay_ms=float(latency_guard.get("drop_delay_ms", 66.7)),
            latency_drop_exceeded_frames=bool(latency_guard.get("drop_exceeded_frames", True)),
            latency_record_dropped_to_excel=bool(latency_guard.get("record_dropped_to_excel", True)),
        )
