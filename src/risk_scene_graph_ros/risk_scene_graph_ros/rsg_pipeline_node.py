"""
ROS 2 wrapper node for the Risk Annotated Scene Graph pipeline.

This module implements the first migration step from the original
FastAPI / QueueWorker-based RiskSceneGraph system to ROS 2.

In Step 1, this node does not yet run the full scene-graph pipeline.
Its purpose is to verify that ROS 2 camera data can be received,
converted into OpenCV / NumPy format, and exposed to the rest of the
RiskSceneGraph codebase later.

Current responsibilities:
    - Subscribe to an RGB image topic.
    - Subscribe to a depth image topic.
    - Convert ROS 2 `sensor_msgs/Image` messages into OpenCV arrays.
    - Store the latest received depth image.
    - Publish a lightweight JSON status message on `/rsg/status`.

Future responsibilities:
    - Synchronize RGB and depth frames.
    - Convert ROS frames into the existing RiskSceneGraph frame format.
    - Call the existing QueueWorker-style processing pipeline.
    - Publish the generated scene graph as a ROS 2 topic.
"""

import json
from typing import Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RsgPipelineNode(Node):
    """
    ROS 2 node that receives camera images and prepares them for RiskSceneGraph.

    The node acts as the ROS-facing wrapper around the existing Python
    RiskSceneGraph library. In this first version, it only verifies that
    RGB and depth images can be received and converted successfully.

    Parameters:
        rgb_topic:
            Name of the ROS 2 topic that publishes RGB images.
            Default: `/camera/color/image_raw`

        depth_topic:
            Name of the ROS 2 topic that publishes depth images.
            Default: `/camera/depth/image_raw`

    Subscriptions:
        rgb_topic:
            Receives RGB images as `sensor_msgs/Image`.

        depth_topic:
            Receives depth images as `sensor_msgs/Image`.

    Publishers:
        /rsg/status:
            Publishes a JSON string containing basic frame reception status.
    """

    def __init__(self) -> None:
        """
        Initialize the Risk Scene Graph ROS wrapper node.

        This constructor:
            - creates a `CvBridge` object,
            - declares ROS parameters,
            - creates RGB and depth image subscribers,
            - creates a status publisher,
            - initializes storage for the latest depth image.
        """
        super().__init__("rsg_pipeline_node")

        self.bridge = CvBridge()

        self.rgb_topic = self.declare_parameter(
            "rgb_topic",
            "/camera/color/image_raw"
        ).value

        self.depth_topic = self.declare_parameter(
            "depth_topic",
            "/camera/depth/image_raw"
        ).value

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            "/rsg/status",
            10
        )

        self.latest_depth: Optional[object] = None

        self.get_logger().info("Risk Scene Graph ROS wrapper started.")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")

    def depth_callback(self, msg: Image) -> None:
        """
        Receive and convert a depth image message.

        The converted depth image is stored in `self.latest_depth`.
        This allows the RGB callback to report whether a depth frame
        has already been received.

        Args:
            msg:
                ROS 2 depth image message of type `sensor_msgs/Image`.

        Notes:
            The depth image is converted using `passthrough` encoding
            because depth cameras may publish different formats, for example:
                - `16UC1`
                - `32FC1`

            `passthrough` preserves the original image encoding.
        """
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough"
            )

        except Exception as error:
            self.get_logger().error(f"Depth conversion failed: {error}")

    def rgb_callback(self, msg: Image) -> None:
        """
        Receive and convert an RGB image message.

        This callback converts the incoming ROS image into an OpenCV BGR image.
        After conversion, it publishes a JSON status message on `/rsg/status`.

        Args:
            msg:
                ROS 2 RGB image message of type `sensor_msgs/Image`.

        Published status format:
            Example:

            ```json
            {
                "status": "received_frame",
                "rgb_shape": [480, 640, 3],
                "has_depth": true
            }
            ```

        Notes:
            OpenCV uses BGR channel order by default. Therefore, the image is
            converted using `bgr8` encoding instead of `rgb8`.
        """
        try:
            rgb = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            status = {
                "status": "received_frame",
                "rgb_shape": list(rgb.shape),
                "has_depth": self.latest_depth is not None,
            }

            self.status_pub.publish(String(data=json.dumps(status)))
            self.get_logger().info(json.dumps(status))

        except Exception as error:
            self.get_logger().error(f"RGB conversion failed: {error}")


def main(args: Optional[list[str]] = None) -> None:
    """
    Entry point for the `rsg_pipeline_node` executable.

    This function initializes ROS 2, creates the `RsgPipelineNode`,
    spins it until shutdown, and then cleans up resources.

    Args:
        args:
            Optional command-line arguments passed by ROS 2.
    """
    rclpy.init(args=args)

    node = RsgPipelineNode()

    try:
        rclpy.spin(node)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()