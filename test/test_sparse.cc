#include <bitplanes/core/config.h>
#include <bitplanes/core/internal/bitplanes_sparse_data.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/bitplanes_tracker_sparse.h>
#include <bitplanes/utils/timer.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>

#if BITPLANES_WITH_PROFILER
#include <gperftools/profiler.h>
#endif

typedef bp::BitPlanesSparseData<bp::Homography> TrackerData;
typedef bp::BitplanesTrackerSparse<bp::Homography> Tracker;

int main()
{
  cv::Mat I = cv::imread("/home/halismai/code/mclk/cvpr/code/data/tracking/br/zm/00000.png", cv::IMREAD_GRAYSCALE);

  const cv::Rect roi(120, 110, 300, 230);

  bp::AlgorithmParameters params;
  params.verbose = false;
  params.function_tolerance = -1;
  params.max_iterations = 1000;

  Tracker tracker(params);
  tracker.setTemplate(I, roi);

  bp::Matrix33f T_init;
  T_init <<
      1.0, 0.0, 2.5,
      0.0, 1.0, -1.0,
      0.0, 0.0, 1.0;

  std::cout << tracker.track(I, T_init) << std::endl;

#if BITPLANES_WITH_PROFILER
  ProfilerStart("/tmp/prof");
#endif

  auto t_ms = bp::TimeCode(100, [&]() { tracker.track(I, T_init); } );
  printf("time: %0.2f ms\n", t_ms);

#if BITPLANES_WITH_PROFILER
  ProfilerStop();
#endif

#if 0
  TrackerData tdata;
  tdata.set(I, roi, -1);

  auto t_ms = bp::TimeCode(100, [&]() { tdata.set(I, roi); });
  printf("time %0.2f ms\n", t_ms);

  {
    std::ofstream ofs("J");
    ofs << tdata.jacobian() << std::endl;
  }

  bp::Vector_<float> residuals;
  bp::Matrix33f T(bp::Matrix33f::Identity());
  tdata.computeResiduals(T, I, residuals);

  t_ms = bp::TimeCode(1000, [&]() { tdata.computeResiduals(T, I, residuals); });
  printf("time %0.2f\n", t_ms);

  printf("ERROR: %g\n", residuals.norm());


  typename TrackerData::Gradient g;
  t_ms = bp::TimeCode(1000, [&]() { tdata.linearize(I, T, g); });
  printf("time: %0.2f\n", t_ms);
#endif


  return 0;
}
