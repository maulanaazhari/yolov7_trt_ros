# yolov7_trt_ros
ROS Node for running object detection with yolov7 + TensorRT

convert yolov7 model to tensortrt: https://github.com/Linaom1214/TensorRT-For-YOLO-Series##YOLOv7
- clone the repository, go inside the repo root and install the requirements
- convert onnx to trt by: python3 export.py -o yolov7v59-tiny.onnx -e yolov7v59-tiny.trt --end2end
- test the converted model by: python3 trt.py -e yolov7-tiny.trt  -i ../mbz_ws/src/yolov7_trt_ros/models/frame0141.jpg -o yolov7v59-tiny-1.jpg --end2end
