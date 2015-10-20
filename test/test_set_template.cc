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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace bp;

#if BITPLANES_WITH_PROFILER
#include <gperftools/profiler.h>
#endif

int main()
{
  auto motion_type = MotionType::Translation;
  auto alg_params = AlgorithmParameters::FromConfigFile("../config/test.cfg");

  Tracker tracker(motion_type, alg_params);

  cv::Mat I(cv::Size(640,480), CV_8UC1);
  {
    auto* p = reinterpret_cast<uint8_t*>(I.data);
    for(int i = 0; i < 640*480; ++i) {
      p[i] = 255.0 * ( rand() / (float) RAND_MAX );
    }
  }

  cv::Rect bbox(10, 10, 600, 400);
  tracker.setTemplate(I, bbox);

  std::cout << bbox << std::endl;

#if BITPLANES_WITH_PROFILER
  ProfilerFlush();
  ProfilerStart("/tmp/prof");
#endif

  auto t_ms = TimeCode(100, [&]() { tracker.setTemplate(I, bbox); });
  Info("Time: %0.2f ms [%0.2f Hz]\n", t_ms, 1.0 /(t_ms / 1000.0));

#if BITPLANES_WITH_PROFILER
  printf("done\n");
  ProfilerFlush();
  ProfilerStop();
#endif

  return 0;
}

