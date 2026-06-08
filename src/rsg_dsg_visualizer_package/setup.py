import os
from glob import glob
from setuptools import setup

package_name = "rsg_dsg_visualizer"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Fabin",
    maintainer_email="fabin@example.com",
    description="Hydra frontend/backend DSG visualizer wrapper and status monitor.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "dsg_update_monitor = rsg_dsg_visualizer.dsg_update_monitor:main",
        ],
    },
)
