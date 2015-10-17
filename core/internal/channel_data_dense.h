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

#ifndef BITPLANES_CORE_INTERNAL_CHANNEL_DATA_DENSE_H
#define BITPLANES_CORE_INTERNAL_CHANNEL_DATA_DENSE_H

#include "bitplanes/core/types.h"
#include "bitplanes/utils/memory.h"
#include "bitplanes/core/motion_model.h"

#include <opencv2/core/core.hpp>

namespace bp {

template <class MotionModelType>
class ChannelDataDense
{
 public:
  typedef MotionModel<MotionModelType> Motion;
  typedef typename Motion::Jacobian       Jacobian;
  typedef typename Motion::Hessian        Hessian;
  typedef typename Motion::Gradient       Gradient;
  typedef typename Motion::JacobianMatrix JacobianMatrix;

  typedef Vector_<float> Pixels;

 public:
  inline ChannelDataDense() {}

  /**
   * sets the data (pixels and jacobians)
   */
  void set(const cv::Mat&, const cv::Rect&,
           float s=1.0, float c1=0.0f, float c2=0.0);

  /**
   * \param Iw warped image
   * \param residuals residual values
   */
  void computeResiduals(const cv::Mat& Iw, Pixels& residuals) const;

  inline const JacobianMatrix& jacobian() const { return _jacobian; }
  inline const Pixels& pixels() const { return _pixels; }

  inline size_t size() const { return static_cast<size_t>(_pixels.size()); }

 protected:
  Pixels _pixels;
  JacobianMatrix _jacobian;

  cv::Rect _bbox;
}; // ChannelDataDense

}; // bp


#endif // BITPLANES_CORE_INTERNAL_CHANNEL_DATA_DENSE_H
