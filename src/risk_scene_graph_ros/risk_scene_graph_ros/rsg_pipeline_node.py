import json
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RsgPipelineNode(Node):
    def __init__(self):
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

        self.latest_depth = None

        self.get_logger().info("Risk Scene Graph ROS wrapper started.")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def rgb_callback(self, msg: Image):
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

        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RsgPipelineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()