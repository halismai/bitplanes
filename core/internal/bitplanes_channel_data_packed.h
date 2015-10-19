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

#ifndef BITPLANES_CORE_INTERNAL_BITPLANES_CHANNEL_DATA_PACKED_H
#define BITPLANES_CORE_INTERNAL_BITPLANES_CHANNEL_DATA_PACKED_H

#include "bitplanes/core/types.h"
#include "bitplanes/core/motion_model.h"
#include <opencv2/core.hpp>

namespace bp {

/**
 * stores the census transform as a single channel and *unpacks* the planes when
 * needed.
 *
 * The Jacobian is stored per pixel (8-jacobians per pixel) consecutive in memory
 */
template <class M>
class BitPlanesChannelDataPacked
{
 public:
  typedef MotionModel<M> Motion;
  typedef typename Motion::WarpJacobian WarpJacobian;
  typedef typename Motion::Hessian Hessian;
  typedef typename Motion::Gradient Gradient;
  typedef typename Motion::JacobianMatrix JacobianMatrix;
  typedef Vector_<float> Pixels;

 public:
  inline BitPlanesChannelDataPacked() {}

  void set(const cv::Mat&, const cv::Rect&, float = 1.0f, float = 0.0f, float = 0.0f);

  void computeResiduals(const cv::Mat&, Pixels& residuals);

  inline const JacobianMatrix& jacobian() const { return _jacobian; }
  inline const Hessian& hessian() const { return _hessian; }
  inline const Pixels& pixels() const { return _pixels; }
  inline size_t size() const  { return _pixels.size(); }

 protected:
  Pixels _pixels;
  JacobianMatrix _jacobian;
  Hessian _hessian;
  cv::Rect _bbox;
  cv::Mat _C0;
  cv::Mat _I0;
}; // BitPlanesChannelDataPacked

}; // bp



#endif // BITPLANES_CORE_INTERNAL_BITPLANES_CHANNEL_DATA_PACKED_H
