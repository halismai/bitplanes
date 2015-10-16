#include "bitplanes/core/config.h"
#include "bitplanes/core/tracker.h"

#include "bitplanes/utils/timer.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace bp;

int main()
{
  auto motion_type = MotionType::Homography;
  auto alg_params = AlgorithmParameters::FromConfigFile("../config/test.cfg");


  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);

  Tracker tracker(motion_type, alg_params);
  tracker.setTemplate(I, cv::Rect(10,10,100,100));
  auto ret = tracker.track(I);
  std::cout << ret << std::endl;

  return 0;
}
