# rsg_dsg_visualizer

ROS 2 helper package for visualizing Hydra frontend or backend Dynamic Scene Graphs.

This package launches Hydra's existing `hydra_visualizer_node` and remaps its DSG input to either:

- `/hydra/frontend/dsg`
- `/hydra/backend/dsg`

It also starts a lightweight monitor node that subscribes to the selected `hydra_msgs/msg/DsgUpdate` topic and logs update status.

The package does **not** deserialize `DsgUpdate.layer_contents` in Python. Actual DSG rendering is handled by `hydra_visualizer_node`.

## Install

Copy this package into:

```bash
~/rsg_ros2_ws/src/rsg_dsg_visualizer
```

Then build:

```bash
cd ~/rsg_ros2_ws
deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select rsg_dsg_visualizer
source install/setup.bash
```

## Run backend DSG visualizer

Start Hydra without the default visualizer:

```bash
ros2 launch hydra_ros uhumans2.launch.yaml start_visualizer:=false
```

Then:

```bash
ros2 launch rsg_dsg_visualizer dsg_visualizer.launch.py graph:=backend
```

## Run frontend DSG visualizer

Start Hydra without the default visualizer:

```bash
ros2 launch hydra_ros uhumans2.launch.yaml start_visualizer:=false
```

Then:

```bash
ros2 launch rsg_dsg_visualizer dsg_visualizer.launch.py graph:=frontend
```

## Launch arguments

```bash
graph:=frontend|backend
fixed_frame:=world|map|odom|...
start_rviz:=true|false
start_monitor:=true|false
use_color_adapter:=true|false
verbosity:=0|1|2
```

Defaults:

- `graph:=backend`
- empty `fixed_frame` means:
  - frontend -> `world`
  - backend -> `map`
- `start_rviz:=true`
- `start_monitor:=true`

## Notes

For uHumans2:

- frontend DSG usually uses frame `world`
- backend DSG usually uses frame `map`

If RViz shows nothing, check the RViz Fixed Frame.
