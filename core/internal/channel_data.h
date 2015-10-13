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

#ifndef BITPLANES_CORE_INTERNAL_CHANNEL_DATA_H
#define BITPLANES_CORE_INTERNAL_CHANNEL_DATA_H

#include "bitplanes/core/internal/cvfwd.h"
#include "bitplanes/core/types.h"
#include "bitplanes/utils/memory.h"

namespace bp {

class ChannelData
{
 public:
  typedef Matrix_<float> Matrix;
  typedef Vector_<float> Vector;

 public:
  ChannelData(MotionType);

  void set(const cv::Mat& image, const PointVector& points,
           float s = 1.0, float c1 = 0.0f, float c2 = 0.0f);

  void computeResiduals(const cv::Mat& Cw, Vector& residuals) const;

  inline const std::vector<size_t>& indices() const { return _inds; }
  inline size_t size() const { return _inds.size(); }

  inline const Vector& pixels() const { return _pixels; }

 protected:
  MotionType _motion_type;
  Vector _pixels;
  Matrix _jacobian;
  std::vector<size_t> _inds;
}; // ChannelData

}; // bp

#endif // BITPLANES_CORE_INTERNAL_CHANNEL_DATA_H
