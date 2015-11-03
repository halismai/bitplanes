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

#ifndef BITPLANES_CORE_FEATURE_BASED_H
#define BITPLANES_CORE_FEATURE_BASED_H

#include "bitplanes/utils/memory.h"
#include "bitplanes/core/types.h"

#include <opencv2/features2d.hpp>
#include <vector>

namespace bp {

class FeatureBasedPlaneTracker
{
 public:
  struct Config
  {
    inline Config()
        : max_num_pts(1024), ransac_max_iters(2000), ransac_reproj_threshold(2) {}

    int max_num_pts;
    int ransac_max_iters;
    double ransac_reproj_threshold;
  }; // Config

  typedef bp::Matrix33f Transform;

 public:
  FeatureBasedPlaneTracker(cv::Ptr<cv::Feature2D>, cv::Ptr<cv::DescriptorMatcher>,
                           Config = Config());

  ~FeatureBasedPlaneTracker();

  void setTemplate(const cv::Mat&, const cv::Rect&);
  Result track(const cv::Mat&);

  inline const std::vector<cv::Point2f> getInliersPoints() const { return _inlier_points; }

 private:
  cv::Ptr<cv::Feature2D> _features;
  cv::Ptr<cv::DescriptorMatcher> _matcher;
  Config _config;

  std::vector<cv::KeyPoint> _keypoints;
  cv::Mat _descriptors;
  cv::Rect _bbox;

  std::vector<cv::Point2f> _inlier_points;
}; // FeatureBasedPlaneTracker

}; // bp

#endif // BITPLANES_CORE_FEATURE_BASED_H
