#pragma once
#include <string>
#include <yolov7_trt_ros/norm.h>
#include <ros/ros.h>

class YOLONormNode{

};

int main(int argc, char** argv) {
    // ros::init(argc, argv, "norm_detect");
    // ros::NodeHandle nh, nhp("~");
    // // ROS_INFO_STREAM("RUNNING");
    
    // // tag_detector::SingleTagDet tag_det(nh, nhp);

    // ros::spin();
    
    // return 0;

    std::string engine_file_path = "";
    if (argc == 4 && std::string(argv[2]) == "-i") 
    {
        engine_file_path = argv[1];
    } 
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolo ../model_trt.engine -i ../*.jpg  // deserialize file and run inference" << std::endl;
        return -1;
    }

    ROS_INFO_STREAM("engine path: " << std::string(argv[1]));
    ROS_INFO_STREAM("image path: " << std::string(argv[3]));

    const std::string input_image_path {argv[3]};
    YOLO yolo(engine_file_path);
    // ROS_INFO_STREAM("engine is created!");
    yolo.detect_img(input_image_path);
    return 0;
}