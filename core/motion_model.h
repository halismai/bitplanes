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

#ifndef BP_CORE_MOTION_MODEL_H
#define BP_CORE_MOTION_MODEL_H

#include "bitplanes/core/motion_model_traits.h"

namespace bp {

template <class Derived>
class MotionModel
{
 public:
  typedef motion_model_traits<Derived> traits;

  typedef typename traits::Transform Transform;
  typedef typename traits::Hessian   Hessian;
  typedef typename traits::Gradient Gradient;
  typedef typename traits::Jacobian Jacobian;
  typedef typename traits::JacobianMatrix JacobianMatrix;
  typedef typename traits::ParameterVector ParameterVector;

  static constexpr int DOF = traits::DOF;

 public:

  static inline Transform Scale(const Transform& T, float s)
  {
    return Derived::Scale(T, s);
  }

  static inline Transform ParamsToMatrix(const ParameterVector& p)
  {
    return Derived::ParamsToMatrix(p);
  }

  static inline ParameterVector MatrixToParams(const Transform& p)
  {
    return Derived::MatrixToParams(p);
  }

  static inline ParameterVector Solve(const Hessian& H, const Gradient& g)
  {
    return Derived::Solve(H, g);
  }

  template <class ... Args> static inline
  Jacobian ComputeJacobian(float x, float y, float Ix, float Iy, Args&... args)
  {
    return Derived::ComputeJacobian(x, y, Ix, Iy, args...);
  }

 protected:
  inline const Derived* derived() const { return static_cast<const Derived*>(this); }
  inline       Derived* derived()       { return static_cast<Derived*>(this); }
}; // MotionModel

}; // bp

#endif // BP_CORE_MOTION_MODEL_H
