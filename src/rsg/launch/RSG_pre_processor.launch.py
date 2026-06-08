"""Launch file for the RSG_pre_processor ROS 2 node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the launch description for RSG_pre_processor."""
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
            executable="RSG_pre_processor",
            name="RSG_pre_processor",
            output="screen",
            parameters=[{"config_file": config_file}],
        ),
    ])
