/*
  This file is part of bitplanes.

  bitplanes is free software: you can redistribute it and/or modify
  it under the terms of the Lesser GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  bitplanes is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  Lesser GNU General Public License for more details.

  You should have received a copy of the Lesser GNU General Public License
  along with bitplanes.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "bitplanes/core/config.h"
#include "bitplanes/core/tracker.h"

#include "bitplanes/utils/timer.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace bp;

#if defined(BITPLANES_WITH_PROFILER)
#include <gperftools/profiler.h>
#endif

#include <iostream>
#include <Eigen/LU>

static inline cv::Mat WarpImage(const cv::Mat& I, const Matrix33f& T_true)
{
  cv::Mat M; cv::eigen2cv<float,3,3>(T_true, M);

  cv::Mat ret;
  cv::warpPerspective(I, ret, M, cv::Size());

  return ret;
}

int main()
{
  auto motion_type = MotionType::Homography;
  auto alg_params = AlgorithmParameters::FromConfigFile("../config/test.cfg");

  Tracker tracker(motion_type, alg_params);

  cv::Mat I(cv::Size(640,480), CV_8UC1);
  {
    auto* p = reinterpret_cast<uint8_t*>(I.data);
    for(int i = 0; i < 640*480; ++i) {
      p[i] = 255.0 * ( rand() / (float) RAND_MAX );
    }
  }

  I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  THROW_ERROR_IF(I.empty(), "couldn not read image");

  std::cout << I.size() << std::endl;

  cv::Rect bbox(1, 1, I.cols-2, I.rows-2);
  tracker.setTemplate(I, bbox);

#if defined(BITPLANES_WITH_PROFILER)
  ProfilerFlush();
  ProfilerStart("/tmp/prof");
#endif

  Matrix33f T_true, T_init;
  T_true <<
      1.0, 0.0, 0.5,
      0.0, 1.0, 0.5,
      0.0, 0.0, 1.0;
  auto I1 = WarpImage(I, T_true);

  T_init.setIdentity();
  auto ret = tracker.track(I, T_init);
  std::cout << ret << std::endl;

  auto t_ms = TimeCode(0, [&]() { tracker.track(I1); });
  Info("Time: %0.2f ms [%0.2f Hz]\n", t_ms, 1.0 /(t_ms / 1000.0));

#if defined(BITPLANES_WITH_PROFILER)
  ProfilerFlush();
  ProfilerStop();
#endif


  ret.T /= ret.T(2,2);

  std::cout << "ERROR: " <<
      ((T_true.inverse() * ret.T) - Matrix33f::Identity()).lpNorm<Eigen::Infinity>()
      << std::endl;

  std::cout << ret.T << std::endl;

  return 0;
}



