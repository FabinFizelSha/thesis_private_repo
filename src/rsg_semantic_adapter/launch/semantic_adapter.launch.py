from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_file = PathJoinSubstitution([FindPackageShare("rsg_semantic_adapter"), "config", "semantic_adapter_uhumans2.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=config_file),
        DeclareLaunchArgument("segmenter_mode", default_value="dummy_grid"),
        DeclareLaunchArgument("rap_mode", default_value="disabled"),
        DeclareLaunchArgument("vlm_mode", default_value="disabled"),
        Node(package="rsg_semantic_adapter", executable="semantic_adapter_node", name="rsg_semantic_adapter", output="screen", parameters=[LaunchConfiguration("config_file"), {"segmenter_mode": LaunchConfiguration("segmenter_mode"), "rap_mode": LaunchConfiguration("rap_mode"), "vlm_mode": LaunchConfiguration("vlm_mode")}]),
    ])
