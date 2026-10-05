from glob import glob

from setuptools import setup

package_name = "lemon_reach"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hidayat",
    maintainer_email="mhieda.robotics@gmail.com",
    description="Single-shot MoveIt approach to the largest detected Lemon.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "lemon_target_node = lemon_reach.lemon_target_node:main",
            "lemon_approach_node = lemon_reach.lemon_approach_node:main",
            "arm_home_node = lemon_reach.arm_home_node:main",
            "lemon_harvest_manager_node = lemon_reach.lemon_harvest_manager_node:main",
            "lemon_ellipse_node = lemon_reach.lemon_ellipse_node:main",
        ],
    },
)
