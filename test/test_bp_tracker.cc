#include "bitplanes/core/config.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/bitplanes_tracker.h"
#include "bitplanes/core/internal/census.h"
#include "bitplanes/core/internal/imwarp.h"

#include "bitplanes/utils/timer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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
  alg_params.verbose = true;
  alg_params.parameter_tolerance = 5e-6;
  alg_params.function_tolerance  = 5e-6;
  alg_params.max_iterations = 1000;
  alg_params.sigma = 1.2;

  BitplanesTracker<Homography> tracker(alg_params);

  Matrix33f T_true;
  T_true <<
      1.0, 0.00, 2.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0;
  T_true = T_true.inverse();

  cv::Mat I1, xmap, ymap;
  imwarp<Homography>(I, I1, T_true, cv::Rect(0,0,I.cols,I.rows));

  tracker.setTemplate(I1, roi);
  auto ret = tracker.track(I1);
  if(!alg_params.verbose)
    printf("track time %0.2f ms\n", TimeCode(100, [&]() {tracker.track(I);}));

  return 0;


  std::cout << ret << std::endl;

  Matrix33f T_err = (ret.T.inverse() * T_true) - Matrix33f::Identity();
  std::cout << "\n" << T_err << std::endl;
  std::cout << "\nParameter error: " << T_err.norm() << std::endl;


  std::cout << "\nGOT:\n" << ret.T << std::endl;
  std::cout << "\nture:\n" << T_true << std::endl;

  return 0;
}

