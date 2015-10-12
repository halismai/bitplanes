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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace bp {

void imsmooth(const cv::Mat& src, cv::Mat& dst, double sigma)
{
  if(sigma > 0.0)
    cv::GaussianBlur(src, dst, cv::Size(), sigma);
  else
    dst = src;
}

void imsmooth(cv::Mat& image, double sigma)
{
  imsmooth(image, image, sigma);
}

} // bp


