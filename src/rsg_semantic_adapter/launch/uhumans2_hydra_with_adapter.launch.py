"""Launch semantic adapter + Hydra using delayed synchronized adapter outputs.

The adapter receives raw rosbag RGB/depth/CameraInfo, runs SAM/RAP, publishes a
semantic image, and republishes RGB/depth/CameraInfo with matching timestamps.
Hydra is remapped to the delayed /rsg/hydra/... topics.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hydra_launch = PathJoinSubstitution(
        [FindPackageShare("hydra_ros"), "launch", "hydra.launch.yaml"]
    )
    adapter_launch = PathJoinSubstitution(
        [FindPackageShare("rsg_semantic_adapter"), "launch", "semantic_adapter.launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("segmenter_mode", default_value="sam"),
            DeclareLaunchArgument("rap_mode", default_value="disabled"),
            DeclareLaunchArgument("vlm_mode", default_value="disabled"),
            DeclareLaunchArgument("start_visualizer", default_value="true"),
            DeclareLaunchArgument("labelspace", default_value="ade20k_full"),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(adapter_launch),
                launch_arguments={
                    "segmenter_mode": LaunchConfiguration("segmenter_mode"),
                    "rap_mode": LaunchConfiguration("rap_mode"),
                    "vlm_mode": LaunchConfiguration("vlm_mode"),
                }.items(),
            ),

            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="world_to_map",
                arguments=["0", "0", "0", "0", "0", "0", "world", "map"],
                output="screen",
            ),

            GroupAction(
                [
                    SetRemap(
                        src="hydra/input/left_cam/rgb/image_raw",
                        dst="/rsg/hydra/left_cam/rgb/image_raw",
                    ),
                    SetRemap(
                        src="hydra/input/left_cam/depth_registered/image_rect",
                        dst="/rsg/hydra/left_cam/depth_registered/image_rect",
                    ),
                    SetRemap(
                        src="hydra/input/left_cam/rgb/camera_info",
                        dst="/rsg/hydra/left_cam/rgb/camera_info",
                    ),
                    SetRemap(
                        src="hydra/input/left_cam/semantic/image_raw",
                        dst="/rsg/semantic/image_raw",
                    ),
                    IncludeLaunchDescription(
                        AnyLaunchDescriptionSource(hydra_launch),
                        launch_arguments={
                            "dataset": "uhumans2",
                            "labelspace": LaunchConfiguration("labelspace"),
                            "robot_frame": "base_link_gt",
                            "odom_frame": "world",
                            "map_frame": "map",
                            "use_sim_time": "true",
                            "start_visualizer": LaunchConfiguration("start_visualizer"),
                            "enable_lcd": "false",
                            "extra_yaml": "{}",
                        }.items(),
                    ),
                ]
            ),
        ]
    )
