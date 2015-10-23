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

#include <bitplanes/core/debug.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/viz.h>
#include <bitplanes/utils/timer.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

#if BITPLANES_WITH_PROFILER
#include <gperftools/profiler.h>
#endif

static const double SCALE = 0.25;

std::vector<cv::Mat> LoadData()
{
  static const char* DATA_DIR = "../data/zm/";

  std::vector<cv::Mat> ret(50);
  for(int i = 0; i < 50; ++i)
  {
    char fn[128];
    snprintf(fn, sizeof(fn)-1, "%s/%05d.png", DATA_DIR, i);
    ret[i] = cv::imread(fn, cv::IMREAD_GRAYSCALE);
    assert( !ret[i].empty() );
    if(SCALE > 0.0)
      cv::resize(ret[i], ret[i], cv::Size(), SCALE, SCALE);
  }

  return ret;
}

using namespace bp;

int main()
{
  const auto images = LoadData();

  //cv::setNumThreads(0);
  //cv::setUseOptimized(0);

  AlgorithmParameters params;
  params.num_levels = 2;
  params.max_iterations = 50;
  params.parameter_tolerance = 1e-5;
  params.function_tolerance = 1e-4;
  params.verbose = false;

  cv::Rect bbox = SCALE > 0.0 ?
      cv::Rect(SCALE*120, SCALE*110, SCALE*300, SCALE*230) :
      cv::Rect(120, 110, 300, 230);

  std::cout << bbox << std::endl;

  BitPlanesTrackerPyramid<Homography> tracker(params);
  tracker.setTemplate(images[0], bbox);

#if BITPLANES_WITH_PROFILER
  ProfilerStart("/tmp/prof");
#endif

  double total_time = 0.0;
  cv::Mat dimg;
  Matrix33f H(Matrix33f::Identity());
  for(size_t i = 1; i < images.size(); ++i)
  {
    Timer timer;
    auto result = tracker.track(images[i], H);
    total_time += timer.stop().count();

    H = result.T;

    DrawTrackingResult(dimg, images[i], bbox, H.data());
    cv::imshow("bitplanes", dimg);
    int k = 0xff & cv::waitKey(2);
    if(k == 'q')
      break;
  }

#if BITPLANES_WITH_PROFILER
  ProfilerStop();
#endif

  Info("Runtime %0.2f Hz\n", images.size() / (total_time / 1000.0));

  return 0;
}

