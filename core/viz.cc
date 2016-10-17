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

#include "bitplanes/core/viz.h"
#include "bitplanes/core/types.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <array>

namespace {

/**
 * returns the tranformed corners of the bounding box
 */
static inline
std::array<cv::Point2f,4> RectToPoints(const cv::Rect& r, const float* H_ptr)
{
  std::array<cv::Point2f,4> ret;
  const Eigen::Matrix3f H = Eigen::Matrix3f::Map(H_ptr);

  Eigen::Vector3f p;
  p = H * Eigen::Vector3f(r.x,           r.y, 1.0); p /= p[2];
  ret[0] = cv::Point2f(p.x(), p.y());

  p = H * Eigen::Vector3f(r.x + r.width, r.y, 1.0); p /= p[2];
  ret[1] = cv::Point2f(p.x(), p.y());

  p = H * Eigen::Vector3f(r.x + r.width, r.y + r.height, 1.0); p /= p[2];
  ret[2] = cv::Point2f(p.x(), p.y());

  p = H * Eigen::Vector3f(r.x          , r.y + r.height, 1.0); p /= p[2];
  ret[3] = cv::Point2f(p.x(), p.y());

  return ret;

}

cv::Scalar ToOpenCV(bp::ColorByName clr, int alpha = 128)
{
  cv::Scalar ret;
  switch(clr)
  {
    case bp::ColorByName::Red:
      ret = cv::Scalar(0, 0, 255, alpha);
      break;

    case bp::ColorByName::Green:
      ret = cv::Scalar(0, 255, 0, alpha);
      break;

    case bp::ColorByName::Blue:
      ret = cv::Scalar(255, 0, 0, alpha);
      break;

    case bp::ColorByName::Black:
      ret = cv::Scalar(0, 0, 0, alpha);
      break;

    case bp::ColorByName::White:
      ret = cv::Scalar(255, 255, 255, alpha);
      break;

    case bp::ColorByName::Yellow:
      //ret = cv::Scalar(0, 255, 255, alpha);
      ret = cv::Scalar(0, 217, 255, alpha);
  }

  return ret;
}

} // namespace


namespace bp {

void DrawTrackingResult(cv::Mat& dst, const cv::Mat& src, const cv::Rect& r,
                        const float* H, ColorByName clr, int thickness, int type,
                        int shift)
{
  if(src.channels() == 1)
    cv::cvtColor(src, dst, CV_GRAY2BGRA);
  else
    src.copyTo(dst);

  const auto cv_clr = ToOpenCV(clr);
  const auto x = RectToPoints(r, H);

  cv::line(dst, x[0], x[1], cv_clr, thickness, type, shift);
  cv::line(dst, x[1], x[2], cv_clr, thickness, type, shift);
  cv::line(dst, x[2], x[3], cv_clr, thickness, type, shift);
  cv::line(dst, x[3], x[0], cv_clr, thickness, type, shift);
  cv::line(dst, x[0], x[2], cv_clr, thickness, type, shift);
  cv::line(dst, x[1], x[3], cv_clr, thickness, type, shift);
}


} // bp
