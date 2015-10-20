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

#ifndef BITPLANES_INTERNAL_CHANNEL_DATA_H
#define BITPLANES_INTERNAL_CHANNEL_DATA_H

#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/types.h"
#include "bitplanes/core/internal/cvfwd.h"

namespace bp {

template <class M>
class BitPlanesChannelData
{
 public:
  typedef MotionModel<M> MotionModelType;
  typedef typename MotionModelType::WarpJacobian WarpJacobian;
  typedef typename MotionModelType::JacobianMatrix JacobianMatrix;
  typedef typename MotionModelType::Hessian Hessian;
  typedef Vector_<float> Pixels;

 public:
  BitPlanesChannelData() = default;

  void set(const cv::Mat& I, const cv::Rect& bbox,
           float s = 1.0f, float c1 = 0.0f, float c2 = 0.0f);

  void computeResiduals(const cv::Mat& Iw, Pixels&) const;

  inline const Vector_<uint8_t>& pixels() const { return _pixels; }
  inline const JacobianMatrix& jacobian() const { return _jacobian; }
  inline const Hessian& hessian() const { return _hessian; }

 protected:
  //Pixels _pixels;
  Vector_<uint8_t> _pixels;
  JacobianMatrix _jacobian;
  Hessian _hessian;
  int _roi_stride;
}; // BitPlanesChannelData
}; // bp

#endif // BITPLANES_INTERNAL_CHANNEL_DATA_H

