"""Launch Phase 1 object detection node.

The active Phase 1 design is a single-process object-detection node. It owns
one FIFO frame queue before SAM/RAP, performs SAM/RAP/unknown tracking in the
same process, and publishes Hydra-ready semantic RGB-D frames.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare("rsg"),
        "config",
        "rsg_pipeline.yaml",
    ])

    config_file = LaunchConfiguration("config_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="Path to the central RSG pipeline YAML configuration file.",
        ),
        Node(
            package="rsg",
            executable="rsg_object_detection",
            name="rsg_object_detection",
            output="screen",
            parameters=[{"config_file": config_file}],
        ),
    ])
