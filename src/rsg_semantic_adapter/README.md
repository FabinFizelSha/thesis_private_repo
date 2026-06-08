# rsg_semantic_adapter

ROS2 package that converts raw RGB-D input into a Hydra-compatible dense semantic image using a SAM/RAP/VLM-style pipeline.

This version adds an asynchronous unknown-object FIFO queue:

```text
RGB-D + CameraInfo + TF
    -> SAM masks
    -> RAP fast label retrieval
    -> known object: publish known label to Hydra
    -> unknown object: publish unknown_object to Hydra immediately
                       enqueue crop for Qwen/VLM in background
    -> /rsg/semantic/image_raw
    -> Hydra

Qwen/VLM background worker
    -> /rsg/semantic/vlm_updates
```

Hydra does not wait for Qwen. The VLM result is published later with the same `adapter_object_id` that was used in `/rsg/semantic/object_updates`.
A future fusion node can match `adapter_object_id` to a Hydra object node using timestamp, class ID, and 3D centroid.

## Important topics

```text
/rsg/semantic/image_raw
  Dense Hydra-compatible semantic label image.

/rsg/semantic/overlay
  RGB visualization overlay for rqt_image_view.

/rsg/semantic/object_updates
  Per-frame SAM/RAP metadata. Includes adapter_object_id, bbox, centroid, label_id, and queued_for_vlm.

/rsg/semantic/vlm_updates
  Delayed Qwen/VLM label and risk result. Includes adapter_object_id.

/rsg/hydra/left_cam/rgb/image_raw
/rsg/hydra/left_cam/depth_registered/image_rect
/rsg/hydra/left_cam/rgb/camera_info
  Delayed synchronized Hydra input bundle. Hydra is remapped to these topics.
```

## Install

```bash
cd ~/rsg_ros2_ws/src
unzip /path/to/rsg_semantic_adapter_async_unknown_package.zip

cd ~/rsg_ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select rsg_semantic_adapter
source install/setup.bash
```

## Run adapter + Hydra

```bash
cd ~/rsg_ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ros2 launch rsg_semantic_adapter uhumans2_hydra_with_adapter.launch.py \
  segmenter_mode:=sam \
  rap_mode:=disabled \
  vlm_mode:=disabled \
  start_visualizer:=true
```

Play your bag:

```bash
ros2 bag play /home/fabin/rsg_ros2_ws/data/office_data \
  --clock \
  --qos-profile-overrides-path ~/.tf_overrides.yaml
```

## Test with dummy mode

```bash
ros2 launch rsg_semantic_adapter uhumans2_hydra_with_adapter.launch.py \
  segmenter_mode:=dummy_grid \
  rap_mode:=disabled \
  vlm_mode:=disabled \
  start_visualizer:=true
```

## Enable asynchronous VLM processing

Start your local Qwen/OpenAI-compatible server first, then launch:

```bash
ros2 launch rsg_semantic_adapter uhumans2_hydra_with_adapter.launch.py \
  segmenter_mode:=sam \
  rap_mode:=http \
  vlm_mode:=qwen \
  start_visualizer:=true
```

If RAP returns low confidence, the frame callback publishes `unknown_object` to Hydra immediately and places the crop in the FIFO queue. The VLM worker later publishes `/rsg/semantic/vlm_updates`.

Monitor queue behavior:

```bash
ros2 topic echo /rsg/semantic/object_updates --once
ros2 topic echo /rsg/semantic/vlm_updates --once
```

## Verify synchronization

```bash
ros2 topic hz /rsg/hydra/left_cam/rgb/image_raw
ros2 topic hz /rsg/hydra/left_cam/depth_registered/image_rect
ros2 topic hz /rsg/hydra/left_cam/rgb/camera_info
ros2 topic hz /rsg/semantic/image_raw
ros2 topic hz /hydra/backend/dsg
```

Hydra should subscribe to `/rsg/hydra/...` topics:

```bash
ros2 node info /hydra | sed -n '/Subscribers:/,/Publishers:/p'
```

## Visualize SAM output

```bash
ros2 run rqt_image_view rqt_image_view
```

Select:

```text
/rsg/semantic/overlay
```

## Generate Python API documentation with pdoc

Install pdoc:

```bash
python3 -m pip install --user --break-system-packages pdoc
```

Generate documentation from the source package:

```bash
cd ~/rsg_ros2_ws/src/rsg_semantic_adapter_package/rsg_semantic_adapter
python3 -m pdoc ./rsg_semantic_adapter -o docs
```

Open the docs:

```bash
xdg-open docs/rsg_semantic_adapter.html
```

Main modules worth documenting:

```text
rsg_semantic_adapter.semantic_adapter_node
rsg_semantic_adapter.segmenters
rsg_semantic_adapter.unknown_object_queue
rsg_semantic_adapter.rap_client
rsg_semantic_adapter.qwen_vlm_client
rsg_semantic_adapter.utils
```

## Architecture note

Do not create a new Hydra pixel class for every VLM phrase. Use a stable class such as `unknown_object` in the semantic image, and store open-vocabulary VLM labels as metadata:

```text
semantic image label: unknown_object
VLM metadata: extension cable, trip_hazard, risk_score=0.88
```

A later fusion node should map:

```text
adapter_object_id -> hydra_node_id
```

using 3D centroid, timestamp, and semantic class.
