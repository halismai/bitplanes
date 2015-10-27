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

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

#include "bitplanes/core/feature_based.h"
#include "bitplanes/utils/error.h"

namespace bp {

FeatureBasedPlaneTracker::
FeatureBasedPlaneTracker(UniquePointer<cv::Feature2D> f,
                         UniquePointer<cv::DescriptorMatcher> m, Config conf)
    : _features(std::move(f)), _matcher(std::move(m)), _config(conf) {}

static inline cv::Mat make_mask(const cv::Size& image_size, const cv::Rect& bbox)
{
  cv::Mat ret(image_size, CV_8UC1);
  ret.setTo( cv::Scalar(0) );
  ret(bbox).setTo(cv::Scalar(255));
  return ret;
}

void FeatureBasedPlaneTracker::setTemplate(const cv::Mat& image, const cv::Rect& bbox)
{
  THROW_ERROR_IF( bbox.x < 1 || bbox.x > image.cols - 2 ||
                  bbox.y < 1 || bbox. y > image.rows - 2,
                  "bounding box is out of image boundaries" );

  _bbox = bbox;
  _features->detectAndCompute(image, make_mask(image.size(), bbox), _keypoints,
                              _descriptors, false);
  _matcher->add(_descriptors);
}

static inline cv::Rect make_bigger_box(cv::Size image_size, cv::Rect src, int buff)
{
  int b = buff / 2;
  int x = std::max(1, src.x - b);
  int y = std::max(1, src.y - b);
  int w = std::min(image_size.width - 2, src.width + b);
  int h = std::min(image_size.height - 2, src.height + b);

  return cv::Rect( x, y, w, h );
}

Result FeatureBasedPlaneTracker::track(const cv::Mat& image)
{
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  auto bbox = make_bigger_box(image.size(), _bbox, 200);

  _features->detectAndCompute(image, make_mask(image.size(), bbox),
                              keypoints, descriptors, false);

  std::vector<cv::DMatch> matches;
  _matcher->match(descriptors, matches);

  std::vector<cv::Point2f> x1, x2;
  x1.reserve(matches.size());
  x2.reserve(matches.size());

  for(const auto& m : matches)
  {
    x1.push_back( _keypoints[m.trainIdx].pt );
    x2.push_back( keypoints[m.queryIdx].pt );
  }

  cv::Mat H;
  cv::findHomography(x1, x2, cv::RANSAC, _config.ransac_reproj_threshold,
                     cv::noArray(), _config.ransac_max_iters);

  bp::Matrix33f H_ret;
  cv::cv2eigen(H, H_ret);
  return Result(H_ret);
}

} // bp

