#include "bitplanes/core/internal/bitplanes_channel_data2.h"
#include "bitplanes/core/internal/bitplanes_channel_data.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/timer.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace bp;

int main()
{
  cv::Mat I = cv::imread("/home/halismai/code/mclk/cvpr/code/data/tracking/br/zm/00000.png", cv::IMREAD_GRAYSCALE);

  cv::Rect roi(120, 110, 300, 230);

  {
    BitPlanesChannelData2<Homography> cdata;
    cdata.set(I, roi);

    auto t_ms = TimeCode(100, [&]() {cdata.set(I, roi); } );
    printf("time: %0.2f ms\n", t_ms);


    cv::Mat I1;
    I(roi).copyTo(I1);

    cv::Mat diff;
    cv::absdiff(I(roi), I1, diff);

    Vector_<float> res;
    t_ms = TimeCode(100, [&]() { cdata.computeResiduals(I1, res); });
    printf("time: %0.2f ms ERROR: %g\n", t_ms, res.norm());
  }

  {
    BitPlanesChannelData<Homography> cdata;

    cdata.set(I, roi);
    auto t_ms = TimeCode(100, [&]() {cdata.set(I, roi); } );
    printf("time: %0.2f ms\n", t_ms);
    Vector_<float> res;
    t_ms = TimeCode(100, [&]() { cdata.computeResiduals(I(roi), res); });
    printf("time: %0.2f ms ERROR: %g\n", t_ms, res.norm());
  }

  return 0;
}

