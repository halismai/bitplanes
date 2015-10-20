#include "bitplanes/core/config.h"
#include "bitplanes/core/intensity_tracker.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/translation.h"
#include "bitplanes/core/affine.h"
#include "bitplanes/utils/timer.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include <Eigen/LU>

#if BITPLANES_WITH_PROFILER
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

  IntensityTracker<Homography> tracker(alg_params);
  tracker.setTemplate(I, roi);

#if BITPLANES_WITH_PROFILER
  ProfilerFlush();
  ProfilerStart("/tmp/prof");
#endif


  auto t_ms = TimeCode(100, [&]() { tracker.setTemplate(I, roi); });
  printf("setTemplate time: %0.2f ms\n", t_ms);

  Matrix33f T_true;
  T_true <<
      1.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0;
  T_true = T_true.inverse();

  cv::Mat I1, xmap, ymap;
  imwarp<Homography>(I, I1, T_true, cv::Rect(0,0,I.cols,I.rows));

  Matrix33f T_init( Matrix33f::Identity() );
  auto ret = tracker.track(I1);
  //std::cout << ret << std::endl;

  t_ms = TimeCode(100, [&]() { tracker.track(I1); });
  printf("track time: %0.2f ms\n", t_ms);

#if BITPLANES_WITH_PROFILER
  ProfilerFlush();
  ProfilerStop();
#endif

  //cv::imshow("I0", I);
  //cv::imshow("I1", I1);

  //std::cout << "T_true:\n" << T_true << std::endl;
  //std::cout << "ret.T:\n" << ret.T << std::endl;

  Matrix33f T_error = (T_true * ret.T  - Matrix33f::Identity());
  //std::cout << "T_error: " << T_error << std::endl;

  std::cout << "ERROR: " << T_error.template lpNorm<Eigen::Infinity>() << std::endl;

  std::cout << ret << std::endl;

  //cv::waitKey();
  return 0;
}

