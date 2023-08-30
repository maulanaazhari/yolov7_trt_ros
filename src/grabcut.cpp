#include <opencv2/core/core.hpp>
#include <grabcut/grabcut.h>
#include <iostream>

int main(int argc, char **argv) {
  const std::string imagePath = "/home/usrg/mbz_ws/src/yolov7_trt_ros/data/tool_512x409.png"; 
  const std::string trimapPath = "/home/usrg/mbz_ws/src/yolov7_trt_ros/data/trimap_512x409.png";
  const std::string outputPath = "/home/usrg/mbz_ws/src/yolov7_trt_ros/data/output_512x409_trimap_iter_5_gamma_10xxxx.png";
  int maxIter = 5;
  float gamma = 10.;

  // Read image and trimap
  cv::Mat im = cv::imread(imagePath);
  cv::Mat imBgra;
  cv::cvtColor(im, imBgra, cv::COLOR_BGR2BGRA);
  cv::Mat trimap = cv::imread(trimapPath, cv::IMREAD_GRAYSCALE);

  // Perform segmentation
  cv::Mat segmentation;
  GrabCut gc(maxIter);
  for (int i = 0; i <= 10; i++){
    segmentation = gc.estimateSegmentationFromTrimap(imBgra, trimap, gamma);
    // segmentation = gc.estimateSegmentationFromRect(imBgra, cv::Rect(0, 0, 512, 409), gamma);
  }
  auto initStart = std::chrono::high_resolution_clock::now();
  for (int i = 0; i <= 100; i++){
    segmentation = gc.estimateSegmentationFromTrimap(imBgra, trimap, gamma);
    // segmentation = gc.estimateSegmentationFromRect(imBgra, cv::Rect(0, 0, 512, 409), gamma);
  }
  auto initEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> initElapsed = initEnd - initStart;
  std::cout << "Elapsed: " << initElapsed.count()/100.0 << " s\n";

  // Save segmentation
  segmentation = segmentation;
  cv::imwrite(outputPath, segmentation);
  

return 0;
}