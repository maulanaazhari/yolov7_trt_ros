cmake_minimum_required(VERSION 3.0.2)
project(yolov7_trt_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

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

find_package(CUDA 11.8 REQUIRED)
include_directories(/usr/local/cuda-11.8/include)
link_directories(/usr/local/cuda-11.8/lib64)
# execute_process(COMMAND uname -m COMMAND tr -d '\n'
#     OUTPUT_VARIABLE ARCHITECTURE
#   )
# if( ${ARCHITECTURE} STREQUAL "x86_64" )
#   # tensorrt
#   include_directories(/usr/include/x86_64-linux-gnu/)
#   link_directories(/usr/lib/x86_64-linux-gnu/)
# elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
#   # tensorrt
#   include_directories(/usr/include/x86_64-linux-gnu/)
#   link_directories(/usr/lib/x86_64-linux-gnu/)
# endif()


# for grabcut
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# find_package(Boost REQUIRED python3)

catkin_package(
  INCLUDE_DIRS include
  # LIBRARIES add_detection
  CATKIN_DEPENDS cv_bridge
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_end2end 
            src/end2end_detect.cpp)
target_include_directories(${PROJECT_NAME}_end2end PUBLIC
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_libraries(${PROJECT_NAME}_end2end
            ${catkin_LIBRARIES} 
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            nvinfer
            nvinfer_plugin
            cudart
            cuda)

add_executable(end2end_detect src/end2end_detect)
target_link_libraries(end2end_detect ${catkin_LIBRARIES} ${PROJECT_NAME}_end2end)

# catkin_python_setup()

catkin_install_python(PROGRAMS script/detect_ros.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

add_definitions(-O2 -pthread)