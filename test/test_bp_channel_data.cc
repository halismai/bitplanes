#include "bitplanes/core/internal/bitplanes_channel_data.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/timer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>

int main()
{
  //Eigen::initParallel();
  //std::cout << "num threads: " << Eigen::nbThreads() << std::endl;
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(80, 50, 320, 240);

  bp::BitPlanesChannelData<bp::Homography> cdata;

  std::cout << roi << std::endl;
  cdata.set(I, roi);

  return 0;

  auto t_ms = bp::TimeCode(10, [&]() { cdata.set(I,roi); });
  printf("time %0.2f ms\n", t_ms);

  cv::Mat I1;
  I(roi).copyTo(I1);

  typename bp::BitPlanesChannelData<bp::Homography>::Pixels residuals;
  cdata.computeResiduals(I1, residuals);

  printf("ERROR [inf]: %f\n", residuals.template lpNorm<Eigen::Infinity>());
  printf("ERROR [L_2]: %f\n", residuals.norm());

  t_ms = bp::TimeCode(100, [&]() { cdata.computeResiduals(I1, residuals); });
  printf("time %0.2f ms\n", t_ms);

  return 0;
}

