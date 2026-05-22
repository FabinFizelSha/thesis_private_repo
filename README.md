# Risk Scene Graph ROS 2 Wrapper — Step 1 Setup

This README documents the setup completed so far for the **Step 1 migration** of the Risk Annotated Scene Graph project into a ROS 2 Jazzy workspace.

The goal of Step 1 is to use ROS 2 only as the **data input/output wrapper** while keeping the existing RiskSceneGraph pipeline as a Python library.

```text
ROS 2 camera / depth / TF input
        ↓
risk_scene_graph_ros/rsg_pipeline_node.py
        ↓
existing RiskSceneGraph library code
        ↓
ROS 2 status / scene graph output
```

At this stage, we are **not** converting every Python module into an individual ROS 2 node. The existing modules such as `SceneGraph`, `VlmHelper`, `SamSegmenter`, `VisualRAP`, and `Worker` are kept as library code.

---

## 1. System used

This setup uses:

```text
Ubuntu: 24.04
ROS 2: Jazzy
Workspace: ~/rsg_ros2_ws
```

ROS 2 Jazzy is the correct ROS 2 distribution for Ubuntu 24.04.

---

## 2. ROS 2 Jazzy installation

The following system packages were installed:

```bash
sudo apt update
sudo apt install software-properties-common curl gnupg lsb-release
sudo add-apt-repository universe
sudo apt update
sudo apt install ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep python3-vcstool
```

ROS 2 environment setup:

```bash
source /opt/ros/jazzy/setup.bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```

`rosdep` was initialized:

```bash
sudo rosdep init
rosdep update
```

---

## 3. ROS image, TF, and build dependencies

The following ROS 2 Jazzy dependencies were installed:

```bash
sudo apt install \
  ros-jazzy-cv-bridge \
  ros-jazzy-image-transport \
  ros-jazzy-message-filters \
  ros-jazzy-tf2-ros \
  ros-jazzy-tf-transformations \
  ros-jazzy-vision-opencv
```

Purpose of these packages:

| Package | Purpose |
|---|---|
| `cv_bridge` | Convert ROS image messages to OpenCV/Numpy images |
| `image_transport` | Image transport support |
| `message_filters` | Synchronize RGB and depth messages |
| `tf2_ros` | Read robot/camera transforms |
| `tf-transformations` | Quaternion and transform utilities |
| `vision-opencv` | OpenCV support for ROS 2 |

---

## 4. Workspace creation

The new ROS 2 workspace was created as:

```bash
mkdir -p ~/rsg_ros2_ws/src
cd ~/rsg_ros2_ws/src
```

Two ROS 2 Python packages were created.

### Core package

```bash
ros2 pkg create risk_scene_graph_core \
  --build-type ament_python
```

This package stores the existing RiskSceneGraph code as normal importable Python library code.

### ROS wrapper package

```bash
ros2 pkg create risk_scene_graph_ros \
  --build-type ament_python \
  --dependencies rclpy sensor_msgs std_msgs geometry_msgs nav_msgs tf2_ros cv_bridge message_filters
```

This package stores the ROS 2 node wrappers, launch files, and ROS-specific configuration.

---

## 5. Current workspace structure

The intended structure is:

```text
~/rsg_ros2_ws/
├── src/
│   ├── risk_scene_graph_core/
│   │   ├── risk_scene_graph_core/
│   │   │   ├── __init__.py
│   │   │   ├── config/
│   │   │   ├── scripts/
│   │   │   ├── data/
│   │   │   └── visual_memory/
│   │   ├── package.xml
│   │   └── setup.py
│   │
│   └── risk_scene_graph_ros/
│       ├── risk_scene_graph_ros/
│       │   ├── __init__.py
│       │   ├── rsg_pipeline_node.py
│       │   ├── ros_frame_converter.py
│       │   └── graph_publisher.py
│       ├── launch/
│       │   └── rsg_live.launch.py
│       ├── config/
│       │   └── rsg_ros_params.yaml
│       ├── package.xml
│       ├── setup.py
│       └── README.md
```

---

## 6. Copying the existing RiskSceneGraph code

The old project path was assumed to be:

```bash
~/PycharmProjects/Risk-Annotated-Scene-Graph/Risk-Annotated-Scene-Graph-main
```

Set environment variables:

```bash
export OLD_RSG=~/PycharmProjects/Risk-Annotated-Scene-Graph/Risk-Annotated-Scene-Graph-main
export NEW_CORE=~/rsg_ros2_ws/src/risk_scene_graph_core/risk_scene_graph_core
```

Create folders inside the new core package:

```bash
cd ~/rsg_ros2_ws/src/risk_scene_graph_core

mkdir -p risk_scene_graph_core/scripts
mkdir -p risk_scene_graph_core/config
mkdir -p risk_scene_graph_core/data
mkdir -p risk_scene_graph_core/visual_memory
```

Copy original source files:

```bash
cp -r $OLD_RSG/scripts/* risk_scene_graph_core/scripts/
cp -r $OLD_RSG/config/* risk_scene_graph_core/config/
```

Copy optional project data:

```bash
cp -r $OLD_RSG/visual_memory/* risk_scene_graph_core/visual_memory/ 2>/dev/null || true
cp -r $OLD_RSG/data/* risk_scene_graph_core/data/ 2>/dev/null || true
```

Copy useful documentation/dependency files:

```bash
cp $OLD_RSG/requirements.txt . 2>/dev/null || true
cp $OLD_RSG/requirement_without_rap.txt . 2>/dev/null || true
cp $OLD_RSG/README.md ORIGINAL_README.md 2>/dev/null || true
```

Add Python package initialization files:

```bash
touch risk_scene_graph_core/__init__.py
touch risk_scene_graph_core/scripts/__init__.py
touch risk_scene_graph_core/config/__init__.py
```

---

## 7. Import cleanup

Old imports may look like this:

```python
from scripts.SceneGraph import SceneGraph
from scripts.VlmHelper import VLMHelper
from config.parameters import *
```

Inside the new ROS 2 workspace, these should be changed to:

```python
from risk_scene_graph_core.scripts.SceneGraph import SceneGraph
from risk_scene_graph_core.scripts.VlmHelper import VLMHelper
from risk_scene_graph_core.config.parameters import *
```

To find old import patterns:

```bash
cd ~/rsg_ros2_ws/src/risk_scene_graph_core/risk_scene_graph_core

grep -R "from scripts\|import scripts\|from config\|import config" -n .
```

Edit the results manually. Avoid blind global replacement until the imports are inspected.

---

## 8. `risk_scene_graph_core/setup.py`

The core package should use this `setup.py` structure:

```python
from setuptools import setup, find_packages

package_name = "risk_scene_graph_core"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fabin",
    maintainer_email="fabin@example.com",
    description="Core Python library for Risk Annotated Scene Graph.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
```

---

## 9. ROS wrapper files

The ROS wrapper package contains:

```text
risk_scene_graph_ros/
├── rsg_pipeline_node.py
├── ros_frame_converter.py
└── graph_publisher.py
```

Create the files:

```bash
cd ~/rsg_ros2_ws/src/risk_scene_graph_ros/risk_scene_graph_ros

touch rsg_pipeline_node.py
touch ros_frame_converter.py
touch graph_publisher.py
```

For Step 1, `rsg_pipeline_node.py` only checks that RGB and depth images can be received and converted.

```python
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
```

---

## 10. `risk_scene_graph_ros/setup.py`

The ROS wrapper package should have this `setup.py`:

```python
from setuptools import setup, find_packages
import os
from glob import glob

package_name = "risk_scene_graph_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fabin",
    maintainer_email="fabin@example.com",
    description="ROS 2 wrapper for Risk Annotated Scene Graph.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rsg_pipeline_node = risk_scene_graph_ros.rsg_pipeline_node:main",
        ],
    },
)
```

---

## 11. ROS parameter file

Create the parameter file:

```bash
mkdir -p ~/rsg_ros2_ws/src/risk_scene_graph_ros/config
nano ~/rsg_ros2_ws/src/risk_scene_graph_ros/config/rsg_ros_params.yaml
```

Content:

```yaml
rsg_pipeline_node:
  ros__parameters:
    rgb_topic: /camera/color/image_raw
    depth_topic: /camera/depth/image_raw
    camera_info_topic: /camera/color/camera_info
    target_frame: map
    camera_frame: camera_color_optical_frame
    publish_debug: true
    use_sam_rap: false
    save_output: true
```

For Step 1:

```yaml
use_sam_rap: false
```

This allows the ROS wrapper to be tested without immediately running SAM, VisualRAP, ChromaDB, or VLM inference.

---

## 12. ROS launch file

Create:

```bash
mkdir -p ~/rsg_ros2_ws/src/risk_scene_graph_ros/launch
nano ~/rsg_ros2_ws/src/risk_scene_graph_ros/launch/rsg_live.launch.py
```

Content:

```python
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
```

---

## 13. Python dependencies

General Python dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel

python3 -m pip install \
  numpy \
  opencv-python \
  pillow \
  matplotlib \
  networkx \
  scipy \
  requests \
  pydantic \
  fastapi \
  uvicorn \
  openai \
  chromadb \
  transformers \
  accelerate \
  sentencepiece
```

PyTorch installation depends on GPU/CUDA setup.

For CUDA 12.1:

```bash
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only testing:

```bash
python3 -m pip install torch torchvision torchaudio
```

Segment Anything:

```bash
python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git
```

Optional transform utilities:

```bash
python3 -m pip install pyquaternion transforms3d
```

---

## 14. Build the workspace

From the workspace root:

```bash
cd ~/rsg_ros2_ws

rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
```

Source the workspace:

```bash
source install/setup.bash
```

Optional: add workspace sourcing to `.bashrc`:

```bash
echo "source ~/rsg_ros2_ws/install/setup.bash" >> ~/.bashrc
```

---

## 15. Run the ROS wrapper

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
source ~/rsg_ros2_ws/install/setup.bash

ros2 launch risk_scene_graph_ros rsg_live.launch.py
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
source ~/rsg_ros2_ws/install/setup.bash

ros2 topic echo /rsg/status
```

Expected output format:

```json
{
  "status": "received_frame",
  "rgb_shape": [480, 640, 3],
  "has_depth": true
}
```

The exact image shape may differ depending on the camera resolution.

---

## 16. RealSense camera support

If using an Intel RealSense camera, install:

```bash
sudo apt install ros-jazzy-realsense2-camera
```

Launch the camera:

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

Then verify image topics:

```bash
ros2 topic list
```

Expected topics may include:

```text
/camera/color/image_raw
/camera/depth/image_raw
/camera/color/camera_info
```

If your RealSense driver publishes different topic names, update:

```text
risk_scene_graph_ros/config/rsg_ros_params.yaml
```

---

## 17. Current success condition

The success condition for Step 1 is:

```text
ROS 2 RGB/depth topics
        ↓
rsg_pipeline_node
        ↓
OpenCV/Numpy conversion
        ↓
/rsg/status publisher
```

This confirms that ROS 2 data can enter the new workspace correctly.

The full scene graph pipeline is not expected to run yet.

---

## 18. Next migration step

After Step 1 works, Step 2 will connect the converted ROS frame into the existing RiskSceneGraph processing logic.

Planned Step 2 target:

```text
ROS 2 image/depth input
        ↓
ROS frame converter
        ↓
existing QueueWorker-style processing function
        ↓
SceneGraph JSON output
        ↓
/rsg/scene_graph_json
```

SAM, VLM, VisualRAP, and LearningWorker will still remain as library components during Step 2.
