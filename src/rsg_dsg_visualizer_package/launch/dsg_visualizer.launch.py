"""Launch Hydra's existing visualizer for either frontend or backend DSG.

Usage:
  ros2 launch rsg_dsg_visualizer dsg_visualizer.launch.py graph:=backend
  ros2 launch rsg_dsg_visualizer dsg_visualizer.launch.py graph:=frontend
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _make_nodes(context, *args, **kwargs):
    graph = LaunchConfiguration("graph").perform(context).strip().lower()
    start_rviz = LaunchConfiguration("start_rviz").perform(context).strip().lower()
    start_monitor = LaunchConfiguration("start_monitor").perform(context).strip().lower()
    use_color_adapter = LaunchConfiguration("use_color_adapter").perform(context).strip().lower()
    verbosity = LaunchConfiguration("verbosity").perform(context).strip()
    fixed_frame = LaunchConfiguration("fixed_frame").perform(context).strip()

    if graph not in ("frontend", "backend"):
        raise RuntimeError("graph must be either 'frontend' or 'backend'")

    dsg_topic = f"/hydra/{graph}/dsg"

    if not fixed_frame:
        # Hydra frontend uses odom_frame. Backend uses map_frame.
        # For uHumans2: odom_frame=world, map_frame=map.
        fixed_frame = "world" if graph == "frontend" else "map"

    visualizer_config = PathJoinSubstitution([
        FindPackageShare("hydra_visualizer"),
        "config",
        "visualizer_config.yaml",
    ])

    visualizer_plugins = PathJoinSubstitution([
        FindPackageShare("hydra_visualizer"),
        "config",
        "visualizer_plugins.yaml",
    ])

    external_plugins = PathJoinSubstitution([
        FindPackageShare("hydra_ros"),
        "config",
        "hydra_ros_visualizer_plugins.yaml",
    ])

    rviz_config = PathJoinSubstitution([
        FindPackageShare("hydra_visualizer"),
        "rviz",
        "streaming_visualizer.rviz",
    ])

    nodes = [
        Node(
            package="hydra_visualizer",
            executable="hydra_visualizer_node",
            name="hydra_visualizer",
            output="screen",
            remappings=[
                ("dsg", dsg_topic),
                ("hydra_visualizer/dsg", dsg_topic),
            ],
            arguments=[
                "--config-utilities-file", visualizer_config,
                "--config-utilities-file", visualizer_plugins,
                "--config-utilities-file", external_plugins,
                "--config-utilities-yaml",
                f"{{glog_level: 1, glog_verbosity: {verbosity}}}",
                "--config-utilities-yaml",
                f"{{graph: {{type: GraphFromRos, frame_id: {fixed_frame}}}}}",
                "--config-utilities-yaml",
                f"{{plugins: {{mesh: {{use_color_adapter: {use_color_adapter}}}}}}}",
            ],
        )
    ]

    if start_monitor in ("true", "1", "yes"):
        nodes.append(
            Node(
                package="rsg_dsg_visualizer",
                executable="dsg_update_monitor",
                name=f"{graph}_dsg_update_monitor",
                output="screen",
                parameters=[
                    {
                        "graph": graph,
                        "dsg_topic": dsg_topic,
                        "frame_id": fixed_frame,
                        "publish_status_marker": True,
                    }
                ],
            )
        )

    if start_rviz in ("true", "1", "yes"):
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name=f"rviz_{graph}_dsg",
                output="screen",
                arguments=["-d", rviz_config],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "graph",
            default_value="backend",
            description="Which Hydra DSG to visualize: frontend or backend.",
        ),
        DeclareLaunchArgument(
            "fixed_frame",
            default_value="",
            description="Empty chooses world for frontend and map for backend.",
        ),
        DeclareLaunchArgument(
            "start_rviz",
            default_value="true",
            description="Start RViz with Hydra's streaming visualizer config.",
        ),
        DeclareLaunchArgument(
            "start_monitor",
            default_value="true",
            description="Start a small DsgUpdate monitor/status marker node.",
        ),
        DeclareLaunchArgument(
            "use_color_adapter",
            default_value="false",
            description="Use Hydra visualizer mesh color adapter.",
        ),
        DeclareLaunchArgument(
            "verbosity",
            default_value="0",
            description="Hydra visualizer glog verbosity.",
        ),
        OpaqueFunction(function=_make_nodes),
    ])
