#!/usr/bin/env python3
"""Monitor Hydra DSG update traffic and publish a small RViz status marker.

This node does not deserialize DsgUpdate.layer_contents. Hydra publishes DSG data
as serialized bytes, so the actual graph rendering is handled by hydra_visualizer_node.
"""

import rclpy
from rclpy.node import Node
from hydra_msgs.msg import DsgUpdate
from visualization_msgs.msg import Marker


class DsgUpdateMonitor(Node):
    def __init__(self):
        super().__init__("dsg_update_monitor")

        self.declare_parameter("graph", "backend")
        self.declare_parameter("dsg_topic", "")
        self.declare_parameter("frame_id", "")
        self.declare_parameter("publish_status_marker", True)

        graph = self.get_parameter("graph").value
        configured_topic = self.get_parameter("dsg_topic").value
        configured_frame = self.get_parameter("frame_id").value
        self.publish_status_marker = self.get_parameter("publish_status_marker").value

        if configured_topic:
            self.dsg_topic = configured_topic
        elif graph == "frontend":
            self.dsg_topic = "/hydra/frontend/dsg"
        else:
            self.dsg_topic = "/hydra/backend/dsg"

        if configured_frame:
            self.frame_id = configured_frame
        elif graph == "frontend":
            self.frame_id = "world"
        else:
            self.frame_id = "map"

        self.sub = self.create_subscription(
            DsgUpdate,
            self.dsg_topic,
            self._on_dsg_update,
            10,
        )

        self.marker_pub = self.create_publisher(
            Marker,
            "dsg_update_status_marker",
            10,
        )

        self.get_logger().info(
            f"Monitoring Hydra DSG topic: {self.dsg_topic} "
            f"(status marker frame: {self.frame_id})"
        )

    def _on_dsg_update(self, msg: DsgUpdate):
        text = (
            f"{self.dsg_topic}: seq={msg.sequence_number}, "
            f"full={msg.full_update}, bytes={len(msg.layer_contents)}, "
            f"deleted_nodes={len(msg.deleted_nodes)}, "
            f"deleted_edges={len(msg.deleted_edges)}"
        )
        self.get_logger().info(text)

        if not self.publish_status_marker:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = "hydra_dsg_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.35
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 1.0

        marker.text = text
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = DsgUpdateMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
