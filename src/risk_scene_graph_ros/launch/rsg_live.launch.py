from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("risk_scene_graph_ros")
    params_file = os.path.join(pkg_share, "config", "rsg_ros_params.yaml")

    return LaunchDescription([
        Node(
            package="risk_scene_graph_ros",
            executable="rsg_pipeline_node",
            name="rsg_pipeline_node",
            output="screen",
            parameters=[params_file],
        )
    ])
