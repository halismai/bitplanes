#include "bitplanes/core/config.h"
#include "bitplanes/core/internal/channel_data_dense.h"
#include "bitplanes/utils/timer.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef bp::ChannelDataDense<bp::Homography> CData;

#if defined(BITPLANES_WITH_PROFILER)
#include <gperftools/profiler.h>
#endif

#include <iostream>

int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(10, 10, I.cols-10, I.rows-10);

  std::cout << roi << std::endl;

  CData cdata;
  cdata.set(I, roi);

#if defined(BITPLANES_WITH_PROFILER)
  ProfilerFlush();
  ProfilerStart("/tmp/prof");
#endif

  auto t_ms = bp::TimeCode(10, [&]() { cdata.set(I, roi); });
  printf("time %0.2f ms\n", t_ms);

  typename CData::Pixels residuals;
  cdata.computeResiduals(I, residuals);

  t_ms = bp::TimeCode(100, [&]() { cdata.computeResiduals(I, residuals); });
  printf("time %0.2f ms\n", t_ms);

  printf("ERROR: %g\n", residuals.lpNorm<Eigen::Infinity>());

#if defined(BITPLANES_WITH_PROFILER)
  ProfilerFlush();
  ProfilerStop();
#endif

  return 0;
}

