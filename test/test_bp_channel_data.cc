#include "bitplanes/core/internal/bitplanes_channel_data.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/timer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>


int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(80, 50, 320, 240);

  bp::BitPlanesChannelData<bp::Homography> cdata;

  cdata.set(I, roi);

  auto t_ms = bp::TimeCode(100, [&]() { cdata.set(I,roi); });
  printf("time %0.2f ms\n", t_ms);

  return 0;
}

