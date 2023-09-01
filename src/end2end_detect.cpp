#include <yolov7_trt_ros/end2end.h>

int main(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "-model_path" && std::string(argv[3]) == "-image_path") {
    char* model_path = argv[2];
    char* image_path = argv[4];
    float* Boxes = new float[4000];
    int* BboxNum = new int[1];
    int* ClassIndexs = new int[1000];
    Yolo yolo(model_path);
    cv::Mat img;
    img = cv::imread(image_path);
    // warmup 
    for (int num =0; num < 10; num++) {
      yolo.Infer(img.cols, img.rows, img.channels(), img.data, Boxes, ClassIndexs, BboxNum);
    }
    // run inference
    auto start = std::chrono::system_clock::now();
    for (int num =0; num < 1; num++) {
      yolo.Infer(img.cols, img.rows, img.channels(), img.data, Boxes, ClassIndexs, BboxNum);
    };
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    yolo.draw_objects(img, Boxes, ClassIndexs, BboxNum);

  } else {
    std::cerr << "--> arguments not right!" << std::endl;
    std::cerr << "--> yolo -model_path ./output.trt -image_path ./demo.jpg" << std::endl;
    return -1;
  }
}