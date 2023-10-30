"""
Setup for carla_ad_agent
"""
import os
from glob import glob
ROS_VERSION = int(os.environ['ROS_VERSION'])

if ROS_VERSION == 1:
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(packages=['yolov7_trt_ros'], package_dir={'': 'script'})

    setup(**d)