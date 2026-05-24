"""
ROS 2 replacement for the original FastAPI APIWorker.

This node replaces the external HTTP API layer with ROS 2 communication while
keeping the existing QueueWorker, LearningWorker, SceneGraph, VLMHelper,
SAMSegmenter, and VisualRAP code as normal Python library code.

Role of this node:
    - Subscribe to ROS RGB and depth image topics.
    - Synchronize RGB/depth frames.
    - Convert ROS image messages into OpenCV / NumPy arrays.
    - Get the camera pose as tx and rotM from TF.
    - Create the same internal queue item format used by the old APIWorker.
    - Push that item into the existing QueueWorker queue.
    - Publish shared-state information back into ROS topics.


Old APIWorker behavior:
    HTTP /frame request
        -> decode RGB/depth
        -> prepare {frame_id, timestamp, rgb_img, depth_img, tx, rotM, source}
        -> queue.put(item)

New ROS behavior:
    ROS RGB/depth callback
        -> convert RGB/depth
        -> read TF pose
        -> prepare {frame_id, timestamp, rgb_img, depth_img, tx, rotM, source}
        -> queue.put(item)
"""

import json
import math
import queue as queue_module
import time
from multiprocessing import Manager, Process, Queue
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as RosTime
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def _json_default(obj: Any) -> Any:
    """
    Convert non-standard Python / NumPy objects into JSON-serializable objects.

    Args:
        obj:
            Object passed by `json.dumps`.

    Returns:
        JSON-compatible representation.

    Raises:
        TypeError:
            If the object cannot be serialized.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    Convert a quaternion into a 3x3 rotation matrix.

    ROS quaternions are ordered as x, y, z, w. The returned rotation matrix
    maps camera-frame coordinates into the target/world frame when it is built
    from the TF transform returned by:

        lookup_transform(target_frame, camera_frame, ...)

    Args:
        x:
            Quaternion x component.
        y:
            Quaternion y component.
        z:
            Quaternion z component.
        w:
            Quaternion w component.

    Returns:
        3x3 NumPy rotation matrix.
    """
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _fill_invalid_depth_nearest(depth_img: np.ndarray) -> np.ndarray:
    """
    Fill invalid depth values using nearest valid depth values.

    The old APIWorker converted depth images into meters and then called
    `fill_invalid_depth_nearest`. This local implementation keeps equivalent
    behavior inside the ROS wrapper without depending on a specific helper
    import path.

    Invalid depth values are:
        - NaN
        - Inf
        - values <= 0

    Args:
        depth_img:
            Depth image in meters.

    Returns:
        Depth image with invalid values filled when possible.
    """
    depth_img = depth_img.astype(np.float32, copy=False)
    invalid = ~np.isfinite(depth_img) | (depth_img <= 0.0)

    if not np.any(invalid):
        return depth_img

    if np.all(invalid):
        return depth_img

    try:
        from scipy import ndimage

        indices = ndimage.distance_transform_edt(
            invalid,
            return_distances=False,
            return_indices=True,
        )
        return depth_img[tuple(indices)].astype(np.float32)

    except Exception:
        # Safe fallback: leave invalid depth unchanged if scipy is unavailable.
        return depth_img


class RsgPipelineNode(Node):
    """
    ROS 2 API-replacement node for RiskSceneGraph.

    This node replaces the original FastAPI `APIWorker` interface. It owns the
    shared queue and shared state, starts the existing QueueWorker process, and
    feeds ROS camera frames into the original processing pipeline.

    Parameters:
        rgb_topic:
            RGB image topic.

        depth_topic:
            Depth image topic.

        target_frame:
            World/map frame used as the persistent scene-graph reference frame.

        camera_frame:
            Camera optical frame. TF is queried from `target_frame` to
            `camera_frame`.

        queue_size:
            Maximum number of pending frames in the internal multiprocessing
            queue.

        learning_queue_size:
            Maximum number of unknown-object learning tasks.

        sync_queue_size:
            Queue size used by the RGB/depth approximate synchronizer.

        sync_slop_sec:
            Allowed timestamp difference between RGB and depth messages.

        enqueue_every_n_frames:
            Enqueue only every Nth synchronized RGB/depth pair. This prevents
            VLM overload when the camera runs at high frame rate.

        use_tf_pose:
            If true, get `tx` and `rotM` from TF.

        allow_identity_pose:
            If true and TF is unavailable, use zero translation and identity
            rotation. This is useful for early testing.

        depth_scale:
            Divisor used for uint16 depth images. RealSense Z16 depth commonly
            uses millimeters, so 1000.0 converts to meters.

        start_queue_worker:
            If true, start the existing QueueWorker as a child process.

        start_learning_worker:
            If true, start the existing LearningWorker as a child process.

        use_offline:
            Forwarded to QueueWorker / LearningWorker.

        use_sam_rap:
            Forwarded to QueueWorker. Enables SAM + VisualRAP path.

        save_output:
            Forwarded to QueueWorker.

        small_vlm:
            Forwarded to QueueWorker.

    Subscriptions:
        rgb_topic:
            RGB image stream.

        depth_topic:
            Depth image stream.

        /rsg/robot_info_json:
            JSON string that replaces the old `/robot/info` API route.

    Publishers:
        /rsg/status:
            General status and enqueue events.

        /rsg/scene_graph_json:
            Latest scene graph from `shared_state["latest"]`.

        /rsg/frame_status_json:
            Recent frame statuses from `shared_state["frame_<id>"]`.
    """

    def __init__(self) -> None:
        """Initialize ROS subscriptions, publishers, queues, shared state, and workers."""
        super().__init__("rsg_pipeline_node")

        self.bridge = CvBridge()

        self.rgb_topic = self.declare_parameter(
            "rgb_topic",
            "/camera/color/image_raw",
        ).value

        self.depth_topic = self.declare_parameter(
            "depth_topic",
            "/camera/depth/image_raw",
        ).value

        self.target_frame = self.declare_parameter(
            "target_frame",
            "map",
        ).value

        self.camera_frame = self.declare_parameter(
            "camera_frame",
            "camera_color_optical_frame",
        ).value

        self.queue_size = int(self.declare_parameter("queue_size", 256).value)
        self.learning_queue_size = int(self.declare_parameter("learning_queue_size", 256).value)
        self.sync_queue_size = int(self.declare_parameter("sync_queue_size", 10).value)
        self.sync_slop_sec = float(self.declare_parameter("sync_slop_sec", 0.05).value)
        self.enqueue_every_n_frames = int(self.declare_parameter("enqueue_every_n_frames", 30).value)

        self.use_tf_pose = bool(self.declare_parameter("use_tf_pose", True).value)
        self.allow_identity_pose = bool(self.declare_parameter("allow_identity_pose", True).value)
        self.depth_scale = float(self.declare_parameter("depth_scale", 1000.0).value)

        self.start_queue_worker = bool(self.declare_parameter("start_queue_worker", True).value)
        self.start_learning_worker = bool(self.declare_parameter("start_learning_worker", False).value)

        self.use_offline = bool(self.declare_parameter("use_offline", False).value)
        self.use_sam_rap = bool(self.declare_parameter("use_sam_rap", False).value)
        self.save_output = bool(self.declare_parameter("save_output", True).value)
        self.small_vlm = bool(self.declare_parameter("small_vlm", False).value)

        self.manager = Manager()
        self.shared_state = self.manager.dict()
        self.shared_state["latest"] = {"nodes": [], "edges": []}
        self.shared_state["_counter"] = 0

        self.work_queue: Queue = Queue(maxsize=self.queue_size)
        self.learning_queue: Queue = Queue(maxsize=self.learning_queue_size)

        self.queue_worker_process: Optional[Process] = None
        self.learning_worker_process: Optional[Process] = None

        self.status_pub = self.create_publisher(String, "/rsg/status", 10)
        self.scene_graph_pub = self.create_publisher(String, "/rsg/scene_graph_json", 10)
        self.frame_status_pub = self.create_publisher(String, "/rsg/frame_status_json", 10)

        self.robot_info_sub = self.create_subscription(
            String,
            "/rsg/robot_info_json",
            self.robot_info_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.sync.registerCallback(self.frame_callback)

        self.total_synced_frames = 0
        self.last_published_latest = None

        self.state_timer = self.create_timer(1.0, self.publish_shared_state)

        if self.start_queue_worker:
            self.start_existing_queue_worker()

        if self.start_learning_worker:
            self.start_existing_learning_worker()

        self.get_logger().info("Risk Scene Graph ROS API replacement started.")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Target frame: {self.target_frame}")
        self.get_logger().info(f"Camera frame: {self.camera_frame}")
        self.get_logger().info(f"QueueWorker enabled: {self.start_queue_worker}")
        self.get_logger().info(f"LearningWorker enabled: {self.start_learning_worker}")
        self.get_logger().info(f"Enqueue every N frames: {self.enqueue_every_n_frames}")

    def start_existing_queue_worker(self) -> None:
        """
        Start the original QueueWorker in a child process.

        This preserves the original architecture:
            APIWorker / ROS wrapper -> multiprocessing queue -> QueueWorker

        The ROS node replaces only the APIWorker side of the design.
        """
        try:
            from risk_scene_graph_core.scripts.Worker import QueueWorker

            queue_worker = QueueWorker(
                queue=self.work_queue,
                shared_state=self.shared_state,
                use_offline=self.use_offline,
                use_sam_rap=self.use_sam_rap,
                save_output=self.save_output,
                learning_queue=self.learning_queue,
                small_vlm=self.small_vlm,
            )

            self.queue_worker_process = Process(
                target=queue_worker.run,
                name="rsg-queue-worker",
                daemon=True,
            )
            self.queue_worker_process.start()

            self.get_logger().info(
                f"Started QueueWorker process with PID {self.queue_worker_process.pid}"
            )

        except Exception as error:
            self.get_logger().error(f"Failed to start QueueWorker: {error}")
            raise

    def start_existing_learning_worker(self) -> None:
        """
        Start the original LearningWorker in a child process.

        Enable this only when `use_sam_rap` is true and the VisualRAP /
        ChromaDB path is ready.
        """
        try:
            from risk_scene_graph_core.scripts.Worker import LearningWorker

            learning_worker = LearningWorker(
                learning_queue=self.learning_queue,
                use_offline=self.use_offline,
            )

            self.learning_worker_process = Process(
                target=learning_worker.run,
                name="rsg-learning-worker",
                daemon=True,
            )
            self.learning_worker_process.start()

            self.get_logger().info(
                f"Started LearningWorker process with PID {self.learning_worker_process.pid}"
            )

        except Exception as error:
            self.get_logger().error(f"Failed to start LearningWorker: {error}")
            raise

    def robot_info_callback(self, msg: String) -> None:
        """
        Receive robot metadata as JSON.

        This replaces the old FastAPI `/robot/info` endpoint. Publish JSON to:

            /rsg/robot_info_json

        Example:
            ros2 topic pub --once /rsg/robot_info_json std_msgs/String \\
              "{data: '{\"battery\": 85, \"mode\": \"autonomous\"}'}"

        Args:
            msg:
                ROS string message containing JSON.
        """
        try:
            robot_info = json.loads(msg.data)
            self.shared_state["robot_info"] = robot_info

            self.publish_status(
                {
                    "status": "robot_info_updated",
                    "robot_info": robot_info,
                }
            )

        except json.JSONDecodeError as error:
            self.publish_status(
                {
                    "status": "robot_info_rejected",
                    "error": str(error),
                    "raw": msg.data,
                }
            )

    def frame_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """
        Receive synchronized RGB and depth images and enqueue a frame.

        This method is the ROS replacement for the old `/frame` API route.
        It creates the same queue item format expected by the existing
        QueueWorker:

            {
                "frame_id": int,
                "timestamp": str,
                "rgb_img": np.ndarray,
                "depth_img": np.ndarray,
                "tx": np.ndarray,
                "rotM": np.ndarray,
                "source": "ros2"
            }

        Args:
            rgb_msg:
                Synchronized RGB image.

            depth_msg:
                Synchronized depth image.
        """
        self.total_synced_frames += 1

        if self.enqueue_every_n_frames > 1:
            if self.total_synced_frames % self.enqueue_every_n_frames != 0:
                return

        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_img_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_img = self.prepare_depth_image(depth_img_raw)

            tx, rotM = self.get_camera_pose(rgb_msg.header.stamp)

            frame_id = self.get_next_frame_id()
            timestamp = self.ros_time_to_string(rgb_msg.header.stamp)

            frame_status = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "status": "queued",
                "queued_at": time.time(),
                "source": "ros2",
            }
            self.shared_state[f"frame_{frame_id}"] = frame_status

            item = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "rgb_img": rgb_img,
                "depth_img": depth_img,
                "tx": tx,
                "rotM": rotM,
                "source": "ros2",
            }

            self.work_queue.put_nowait(item)

            self.publish_status(
                {
                    "status": "queued",
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "queue_size": self.safe_queue_size(self.work_queue),
                    "rgb_shape": list(rgb_img.shape),
                    "depth_shape": list(depth_img.shape),
                    "source": "ros2",
                }
            )

        except queue_module.Full:
            self.publish_status(
                {
                    "status": "dropped",
                    "reason": "work_queue_full",
                    "queue_size": self.safe_queue_size(self.work_queue),
                }
            )

        except Exception as error:
            self.publish_status(
                {
                    "status": "frame_rejected",
                    "error": str(error),
                }
            )
            self.get_logger().error(f"Failed to enqueue ROS frame: {error}")

    def prepare_depth_image(self, depth_img: np.ndarray) -> np.ndarray:
        """
        Convert a ROS depth image to float32 meters.

        The old APIWorker converted uint16 depth into meters by dividing by
        1000.0 and treated non-uint16 depth as already metric-like. This method
        keeps that behavior but exposes the divisor as the `depth_scale`
        parameter.

        Args:
            depth_img:
                Raw depth image from `cv_bridge`.

        Returns:
            Float32 depth image in meters.
        """
        if depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / self.depth_scale
        else:
            depth_m = depth_img.astype(np.float32)

        return _fill_invalid_depth_nearest(depth_m)

    def get_camera_pose(self, stamp: RosTime) -> tuple[np.ndarray, np.ndarray]:
        """
        Get camera translation and rotation matrix.

        QueueWorker expects:
            - tx: camera/world translation as shape (3,)
            - rotM: camera-to-world rotation matrix as shape (3, 3)

        Args:
            stamp:
                ROS timestamp from the RGB image.

        Returns:
            Tuple `(tx, rotM)`.

        Raises:
            RuntimeError:
                If TF is enabled, no transform is available, and identity
                fallback is disabled.
        """
        if not self.use_tf_pose:
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )

            translation = transform.transform.translation
            rotation = transform.transform.rotation

            tx = np.array(
                [translation.x, translation.y, translation.z],
                dtype=np.float64,
            )

            rotM = _quaternion_to_rotation_matrix(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            )

            return tx, rotM

        except TransformException as error:
            if self.allow_identity_pose:
                self.get_logger().warn(
                    f"TF unavailable; using identity pose. Reason: {error}"
                )
                return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

            raise RuntimeError(f"TF lookup failed: {error}") from error

    def get_next_frame_id(self) -> int:
        """
        Increment and return the shared frame counter.

        This mirrors the old APIWorker counter stored under `_counter` in
        shared state.
        """
        current = int(self.shared_state.get("_counter", 0))
        next_id = current + 1
        self.shared_state["_counter"] = next_id
        return next_id

    def publish_shared_state(self) -> None:
        """
        Publish latest scene graph and frame statuses from shared state.

        This replaces API read endpoints such as:
            - get latest scene graph
            - get frame status
            - get system status
        """
        latest = self.shared_state.get("latest", {"nodes": [], "edges": []})

        latest_json = json.dumps(latest, default=_json_default)
        if latest_json != self.last_published_latest:
            self.scene_graph_pub.publish(String(data=latest_json))
            self.last_published_latest = latest_json

        frame_statuses = {}
        for key in list(self.shared_state.keys()):
            if str(key).startswith("frame_"):
                frame_statuses[str(key)] = self.shared_state[key]

        status_msg = {
            "status": "running",
            "queue_size": self.safe_queue_size(self.work_queue),
            "latest_frame_id": self.shared_state.get("_counter", 0),
            "frames": frame_statuses,
            "queue_worker_alive": (
                self.queue_worker_process.is_alive()
                if self.queue_worker_process is not None
                else False
            ),
            "learning_worker_alive": (
                self.learning_worker_process.is_alive()
                if self.learning_worker_process is not None
                else False
            ),
        }

        self.frame_status_pub.publish(
            String(data=json.dumps(status_msg, default=_json_default))
        )

    def publish_status(self, payload: dict[str, Any]) -> None:
        """
        Publish a JSON status message.

        Args:
            payload:
                JSON-compatible status dictionary.
        """
        self.status_pub.publish(
            String(data=json.dumps(payload, default=_json_default))
        )

    @staticmethod
    def safe_queue_size(q: Queue) -> int:
        """
        Return multiprocessing queue size when supported.

        Some platforms do not support `qsize()`. In that case, this method
        returns -1.

        Args:
            q:
                Multiprocessing queue.

        Returns:
            Queue size, or -1 if unavailable.
        """
        try:
            return int(q.qsize())
        except Exception:
            return -1

    @staticmethod
    def ros_time_to_string(stamp: RosTime) -> str:
        """
        Convert ROS time into a stable string timestamp.

        Args:
            stamp:
                ROS time message.

        Returns:
            Timestamp string in `sec.nanosec` format.
        """
        return f"{stamp.sec}.{stamp.nanosec:09d}"

    def shutdown_workers(self) -> None:
        """
        Stop child worker processes cleanly.

        QueueWorker exits when it receives a sentinel `None` queue item.
        """
        try:
            self.work_queue.put_nowait(None)
        except Exception:
            pass

        try:
            self.learning_queue.put_nowait(None)
        except Exception:
            pass

        if self.queue_worker_process is not None:
            self.queue_worker_process.join(timeout=5.0)
            if self.queue_worker_process.is_alive():
                self.queue_worker_process.terminate()

        if self.learning_worker_process is not None:
            self.learning_worker_process.join(timeout=5.0)
            if self.learning_worker_process.is_alive():
                self.learning_worker_process.terminate()

        try:
            self.manager.shutdown()
        except Exception:
            pass


def main(args: Optional[list[str]] = None) -> None:
    """
    Entry point for the ROS 2 API-replacement node.

    Args:
        args:
            Optional ROS command-line arguments.
    """
    rclpy.init(args=args)

    node = RsgPipelineNode()

    try:
        rclpy.spin(node)

    finally:
        node.shutdown_workers()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()