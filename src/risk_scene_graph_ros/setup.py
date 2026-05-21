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