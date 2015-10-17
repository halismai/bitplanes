#include "bitplanes/core/config.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/bitplanes_tracker.h"
#include "bitplanes/core/internal/census.h"

#include "bitplanes/utils/timer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include <Eigen/LU>

#if defined(BITPLANES_WITH_PROFILER)
#include <gperftools/profiler.h>
#endif

using namespace bp;

int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(80, 50, 320, 240);

  AlgorithmParameters alg_params;
  alg_params.verbose = false;
  alg_params.parameter_tolerance = 1e-6;
  alg_params.function_tolerance = 5e-5;

  BitplanesTracker<Homography> tracker(alg_params);
  tracker.setTemplate(I, roi);

  auto t_ms = TimeCode(100, [&]() { tracker.setTemplate(I, roi); });
  printf("setTemplate time %0.2f ms\n", t_ms);

  cv::Mat C1, C2, D;
  simd::CensusTransform(I, roi, C1);
  CensusTransform(I, roi, C2);

  cv::absdiff(C1, C2, D);
  cv::imshow("D", D);
  cv::waitKey();

  return 0;
}

