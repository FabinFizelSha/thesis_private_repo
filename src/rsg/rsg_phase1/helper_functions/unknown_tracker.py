"""Persistent unknown-object tracker for Phase 1.

The classifier still creates a frame-local candidate_id for every unknown mask,
but this tracker assigns a persistent unknown_track_id for the physical object.
The VLM path is keyed by unknown_track_id, which prevents repeated VLM calls when
continuous frames observe the same unknown object.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _as_xyz(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        arr = np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return None
        return arr
    except Exception:
        return None


def _bbox_iou(a: Any, b: Any) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != 4 or len(b) != 4:
        return 0.0
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _volume_ratio(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return 1.0
    hi = max(float(a), float(b))
    lo = max(min(float(a), float(b)), 1e-9)
    return float(hi / lo)


def observation_quality(metadata: Dict[str, Any], image_area_px: Optional[int] = None) -> float:
    """Return a simple score for choosing the best frame/crop for VLM.

    Higher is better. The score favours valid depth and larger object masks.
    It is deliberately lightweight and only uses existing metadata.
    """
    depth_valid_ratio = float(metadata.get("depth_valid_ratio", 0.0) or 0.0)
    mask_area = float(metadata.get("mask_area_px", 0.0) or 0.0)
    if image_area_px and image_area_px > 0:
        area_term = min(mask_area / float(image_area_px), 1.0)
    else:
        # Smoothly compress pixel count without needing full image size.
        area_term = min(math.log1p(mask_area) / math.log1p(25000.0), 1.0)
    valid_geometry_bonus = 0.15 if bool(metadata.get("valid_geometry", False)) else 0.0
    return float(0.65 * depth_valid_ratio + 0.35 * area_term + valid_geometry_bonus)


@dataclass
class UnknownTrack:
    track_id: str
    first_seen_frame_id: str
    first_seen_sequence: int
    first_seen_timestamp_sec: float
    last_seen_frame_id: str
    last_seen_sequence: int
    last_seen_timestamp_sec: float
    centroid_3d: Optional[np.ndarray]
    bbox_volume_m3: Optional[float]
    bbox_2d: Any
    seen_count: int = 1
    candidate_ids: List[str] = field(default_factory=list)
    vlm_status: str = "not_queued"  # not_queued, queued, done, failed
    vlm_result: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    best_candidate_id: str = ""
    best_frame_id: str = ""
    best_sequence: int = 0
    best_metadata: Dict[str, Any] = field(default_factory=dict)
    best_crop: Any = None
    best_header: Any = None
    created_monotonic: float = field(default_factory=time.perf_counter)
    queued_monotonic: Optional[float] = None


class UnknownObjectTracker:
    """Associate unknown detections over time and gate VLM dispatch per track."""

    def __init__(self, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self._tracks: Dict[str, UnknownTrack] = {}
        self._next_track_index = 1
        self._lock = Lock()

    def associate(self, metadata: Dict[str, Any], frame_id: str, sequence: int, timestamp_sec: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Assign a persistent track ID to one unknown detection."""
        if not self.config.unknown_tracking_enabled:
            track_id = str(metadata.get("candidate_id", "unknown_track_disabled"))
            metadata.update({
                "unknown_track_id": track_id,
                "track_event": "tracking_disabled",
                "track_seen_count": 1,
                "vlm_status": "not_queued",
            })
            return metadata, {"track_event": "tracking_disabled", "unknown_track_id": track_id}

        centroid = _as_xyz(metadata.get("centroid_3d"))
        volume = self._safe_float(metadata.get("bbox_volume_m3"))
        bbox_2d = metadata.get("bbox_2d")
        candidate_id = str(metadata.get("candidate_id", ""))

        with self._lock:
            self._remove_old_tracks(timestamp_sec)
            match_id, match_reason, match_distance = self._find_match(centroid, volume, bbox_2d, timestamp_sec)
            if match_id is None:
                track_id = f"unknown_track_{self._next_track_index:04d}"
                self._next_track_index += 1
                track = UnknownTrack(
                    track_id=track_id,
                    first_seen_frame_id=frame_id,
                    first_seen_sequence=int(sequence),
                    first_seen_timestamp_sec=float(timestamp_sec),
                    last_seen_frame_id=frame_id,
                    last_seen_sequence=int(sequence),
                    last_seen_timestamp_sec=float(timestamp_sec),
                    centroid_3d=centroid.copy() if centroid is not None else None,
                    bbox_volume_m3=volume,
                    bbox_2d=bbox_2d,
                    candidate_ids=[candidate_id],
                    best_score=0.0,
                    best_candidate_id=candidate_id,
                    best_frame_id=frame_id,
                    best_sequence=int(sequence),
                    best_metadata=dict(metadata),
                )
                self._tracks[track_id] = track
                track_event = "new_track"
                track_seen_count = 1
            else:
                track = self._tracks[match_id]
                track_id = track.track_id
                track.seen_count += 1
                track.last_seen_frame_id = frame_id
                track.last_seen_sequence = int(sequence)
                track.last_seen_timestamp_sec = float(timestamp_sec)
                if candidate_id:
                    track.candidate_ids.append(candidate_id)
                self._update_track_geometry(track, centroid, volume, bbox_2d)
                track_event = "matched_existing_track"
                track_seen_count = track.seen_count

            metadata.update({
                "unknown_track_id": track_id,
                "track_event": track_event,
                "track_match_reason": match_reason,
                "track_match_distance_m": None if match_distance is None else float(match_distance),
                "track_seen_count": int(track_seen_count),
                "first_seen_frame_id": track.first_seen_frame_id,
                "last_seen_frame_id": frame_id,
                "vlm_status": track.vlm_status,
            })
            record = {
                "unknown_track_id": track_id,
                "track_event": track_event,
                "track_seen_count": int(track_seen_count),
                "vlm_status": track.vlm_status,
                "match_reason": match_reason,
                "match_distance_m": match_distance,
            }
            return metadata, record

    def update_evidence_and_build_vlm_task(
        self,
        unknown: Dict[str, Any],
        rgb_crop: Any,
        frame_header: Any,
        frame_id: str,
        sequence: int,
        image_area_px: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Update best crop for a track and return a VLM task only when ready.

        The method returns at most one VLM task per persistent track unless the
        config allows retries. This is the core behavior needed to avoid repeated
        VLM calls for the same unknown object across consecutive frames.
        """
        track_id = str(unknown.get("unknown_track_id", ""))
        if not track_id:
            return None, {"vlm_dispatch_status": "missing_track_id"}

        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return None, {"vlm_dispatch_status": "track_not_found"}

            score = observation_quality(unknown, image_area_px=image_area_px)
            if score >= track.best_score or not self.config.unknown_best_frame_selection_enabled:
                track.best_score = float(score)
                track.best_candidate_id = str(unknown.get("candidate_id", ""))
                track.best_frame_id = frame_id
                track.best_sequence = int(sequence)
                track.best_metadata = dict(unknown)
                track.best_crop = None if rgb_crop is None else rgb_crop.copy()
                track.best_header = frame_header

            if track.vlm_status == "done":
                return None, {"vlm_dispatch_status": "already_done", "best_frame_score": track.best_score}
            if track.vlm_status == "queued" and self.config.unknown_call_vlm_only_once_per_track:
                return None, {"vlm_dispatch_status": "already_queued", "best_frame_score": track.best_score}
            if track.vlm_status == "failed" and not self.config.unknown_retry_failed_tracks:
                return None, {"vlm_dispatch_status": "failed_no_retry", "best_frame_score": track.best_score}

            first_seen_age = float(unknown.get("timestamp_sec", 0.0) or 0.0) - track.first_seen_timestamp_sec
            enough_observations = track.seen_count >= self.config.unknown_min_observations_before_vlm
            waited_long_enough = first_seen_age >= self.config.unknown_max_wait_before_vlm_sec
            quality_ok = track.best_score >= self.config.unknown_min_quality_for_vlm

            if not quality_ok:
                return None, {"vlm_dispatch_status": "waiting_for_better_frame", "best_frame_score": track.best_score}
            if not (enough_observations or waited_long_enough):
                return None, {"vlm_dispatch_status": "waiting_for_more_observations", "best_frame_score": track.best_score}

            task = {
                "frame_header": track.best_header if track.best_header is not None else frame_header,
                "rsg_frame_id": track.best_frame_id or frame_id,
                "sequence": int(track.best_sequence or sequence),
                "candidate_id": track.best_candidate_id or str(unknown.get("candidate_id", "")),
                "unknown_track_id": track.track_id,
                "mask_id": str(track.best_metadata.get("mask_id", unknown.get("mask_id", ""))),
                "rgb_crop": track.best_crop,
                "object_metadata": dict(track.best_metadata or unknown),
                "created_monotonic": time.perf_counter(),
                "track_seen_count": int(track.seen_count),
                "best_frame_score": float(track.best_score),
            }
            # Do not mark the track as queued here. The FIFO queue may be full.
            # The classifier marks the track as queued only after successful
            # insertion into the post-RAP VLM FIFO queue.
            return task, {"vlm_dispatch_status": "ready_for_vlm", "best_frame_score": track.best_score}

    def mark_vlm_queued(self, track_id: str) -> None:
        """Mark a persistent unknown track as inserted into the VLM FIFO queue."""
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return
            track.vlm_status = "queued"
            track.queued_monotonic = time.perf_counter()

    def mark_vlm_queue_rejected(self, track_id: str, reason: str = "queue_full") -> None:
        """Release a track if its VLM task could not be enqueued.

        This allows the same persistent track to be retried in a later frame
        rather than getting stuck in queued state when the FIFO queue is full.
        """
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return
            if track.vlm_status == "queued":
                track.vlm_status = "not_queued"
                track.queued_monotonic = None

    def mark_vlm_result(self, track_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return
            track.vlm_status = "done" if bool(result.get("success", False)) else "failed"
            track.vlm_result = dict(result)

    def _find_match(self, centroid: Optional[np.ndarray], volume: Optional[float], bbox_2d: Any, timestamp_sec: float) -> Tuple[Optional[str], str, Optional[float]]:
        best_id: Optional[str] = None
        best_distance: Optional[float] = None
        best_reason = "no_match"
        for track_id, track in self._tracks.items():
            age = timestamp_sec - track.last_seen_timestamp_sec
            if age > self.config.unknown_max_track_age_sec:
                continue
            volume_ok = _volume_ratio(volume, track.bbox_volume_m3) <= self.config.unknown_max_volume_ratio
            if centroid is not None and track.centroid_3d is not None:
                dist = float(np.linalg.norm(centroid - track.centroid_3d))
                if dist <= self.config.unknown_max_match_distance_m and volume_ok:
                    if best_distance is None or dist < best_distance:
                        best_id, best_distance, best_reason = track_id, dist, "centroid_3d"
            elif self.config.unknown_use_2d_iou_fallback:
                iou = _bbox_iou(bbox_2d, track.bbox_2d)
                if iou >= self.config.unknown_min_2d_iou and volume_ok:
                    dist_proxy = 1.0 - iou
                    if best_distance is None or dist_proxy < best_distance:
                        best_id, best_distance, best_reason = track_id, dist_proxy, "bbox_2d_iou"
        return best_id, best_reason, best_distance

    def _update_track_geometry(self, track: UnknownTrack, centroid: Optional[np.ndarray], volume: Optional[float], bbox_2d: Any) -> None:
        alpha = float(self.config.unknown_centroid_update_alpha)
        if self.config.unknown_update_track_centroid and centroid is not None:
            if track.centroid_3d is None:
                track.centroid_3d = centroid.copy()
            else:
                track.centroid_3d = alpha * track.centroid_3d + (1.0 - alpha) * centroid
        if volume is not None and volume > 0.0:
            if track.bbox_volume_m3 is None or track.bbox_volume_m3 <= 0.0:
                track.bbox_volume_m3 = volume
            else:
                track.bbox_volume_m3 = alpha * float(track.bbox_volume_m3) + (1.0 - alpha) * float(volume)
        if bbox_2d:
            track.bbox_2d = bbox_2d

    def _remove_old_tracks(self, timestamp_sec: float) -> None:
        if self.config.unknown_max_tracks <= 0:
            return
        # First remove stale tracks that never received a VLM result and are far outside matching age.
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if (timestamp_sec - track.last_seen_timestamp_sec) > max(self.config.unknown_max_track_age_sec * 5.0, self.config.unknown_max_track_age_sec + 1.0)
            and track.vlm_status not in {"queued"}
        ]
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)
        # If still too large, drop oldest not-queued track.
        while len(self._tracks) > self.config.unknown_max_tracks:
            removable = [track for track in self._tracks.values() if track.vlm_status != "queued"]
            if not removable:
                break
            oldest = min(removable, key=lambda tr: tr.last_seen_timestamp_sec)
            self._tracks.pop(oldest.track_id, None)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            f = float(value)
            if math.isfinite(f):
                return f
        except Exception:
            pass
        return None
