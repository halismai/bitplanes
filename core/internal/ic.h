/*
  This file is part of bitplanes.

  the Free Software Foundation, either version 3 of the License, or
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

#ifndef BITPLANES_CORE_INTERNAL_IC_H
#define BITPLANES_CORE_INTERNAL_IC_H

#include "bitplanes/core/internal/tracker_impl.h"
#include "bitplanes/core/internal/cvfwd.h"

#include "bitplanes/core/motion_model.h"

namespace bp {

template <class Motion>
class InverseCompositionalImpl : public Tracker::Impl
{
 public:
  typedef MotionModel<Motion> MotionModelType;
  typedef typename MotionModelType::Transform Transform;
  typedef typename MotionModelType::Hessian Hessian;
  typedef typename MotionModelType::Gradient Gradient;
  typedef typename MotionModelType::JacobianMatrix JacobianMatrix;
  typedef typename MotionModelType::ParameterVector ParameterVector;

 public:
  inline InverseCompositionalImpl(MotionType m, AlgorithmParameters p)
       : Tracker::Impl(m, p) {}

  inline ~InverseCompositionalImpl() {}

  /**
   * track the template (does the optimization)
   */
  Result track(const cv::Mat&, const Transform&);

  /**
   * set the template data
   */
  void setTemplate(const cv::Mat&, const cv::Rect& box);

 protected:
  /**
   * linearize (called at every iteration of the optimization)
   */
  float linearize(const cv::Mat&, const Transform&);

 protected:
  Hessian  _hessian;
  Gradient _gradient;
}; // InverseCompositionalImpl
}; // bp

#endif // BITPLANES_CORE_INTERNAL_IC_H

