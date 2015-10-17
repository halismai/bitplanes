#include "bitplanes/core/internal/channel_data_dense.h"
#include "bitplanes/utils/timer.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef bp::ChannelDataDense<bp::Homography> CData;

int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(10, 10, 320, 240);

  CData cdata;
  cdata.set(I, roi);

  auto t_ms = bp::TimeCode(100, [&]() { cdata.set(I, roi); });
  printf("time %0.2f ms\n", t_ms);

  typename CData::Pixels residuals;
  cdata.computeResiduals(I, residuals);
  printf("ERROR: %g\n", residuals.lpNorm<Eigen::Infinity>());

  t_ms = bp::TimeCode(100, [&]() { cdata.computeResiduals(I, residuals); });
  printf("time %0.2f ms\n", t_ms);

  return 0;
}
