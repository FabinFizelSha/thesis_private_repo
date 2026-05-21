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