#include <bitplanes/core/internal/bitplanes_channel_data_subsampled.h>
#include <bitplanes/core/homography.h>

#include <bitplanes/utils/timer.h>

#include <opencv2/highgui.hpp>

using namespace bp;

int main()
{
  cv::Mat I0 = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(10, 10, 256, 256);

  BitPlanesChannelDataSubSampled<Homography> cdata(1);
  cdata.set(I0, roi);

  {
    auto t = TimeCode(10, [&]() { cdata.set(I0, roi); });
    printf("set() time %f\n", t);
  }


  cv::Mat Iw;
  {
    Matrix33f T(Matrix33f::Identity());
    T(0,2) = 2.5;
    T(1,2) = 0.5;

    auto t = TimeCode(100, [&]() { cdata.warpImage(I0, T, roi, Iw); });
    printf("warpImage time %f\n", t);
  }

  {
    typename BitPlanesChannelDataSubSampled<Homography>::Residuals residuals;
    typename BitPlanesChannelDataSubSampled<Homography>::Gradient gradient;
    auto t = TimeCode(100, [&]() {
                      cdata.computeResiduals(Iw, residuals);
                      gradient = cdata.jacobian().transpose() * residuals;
                      });
    printf("computeResiduals %f\n", t);
  }

  {
    typename BitPlanesChannelDataSubSampled<Homography>::Gradient gradient;
    auto t = TimeCode(100, [&]() {
                      cdata.doLinearize(Iw, gradient);
                      });
    printf("doLinearize %f\n", t);
  }

  return 0;
}


