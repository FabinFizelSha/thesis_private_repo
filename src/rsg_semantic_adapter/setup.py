import os
from glob import glob
from setuptools import setup
package_name = "rsg_semantic_adapter"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Fabin",
    maintainer_email="fabin@example.com",
    description="RGB-D to Hydra semantic image adapter with SAM/RAP/VLM hooks.",
    license="MIT",
    entry_points={"console_scripts": ["semantic_adapter_node = rsg_semantic_adapter.semantic_adapter_node:main"]},
)
