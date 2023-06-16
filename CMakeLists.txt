cmake_minimum_required(VERSION 3.0.2)
project(yolov7_trt_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  vision_msgs
  cv_bridge
)

# find_package(Boost REQUIRED python3)

catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES add_detection
  CATKIN_DEPENDS cv_bridge
)

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
)

catkin_python_setup()

catkin_install_python(PROGRAMS script/detect_ros.py script/tensorrt_python/utils/utils.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})