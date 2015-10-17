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

#ifndef BITPLANES_CORE_AFFINE_H
#define BITPLANES_CORE_AFFINE_H

#include "bitplanes/core/motion_model.h"

namespace bp {

class Affine : public MotionModel<Affine>
{
 public:
  typedef MotionModel<Affine> Base;

  typedef typename Base::Transform Transform;
  typedef typename Base::Hessian   Hessian;
  typedef typename Base::Gradient  Gradient;
  typedef typename Base::Jacobian  Jacobian;
  typedef typename Base::JacobianMatrix JacobianMatrix;
  typedef typename Base::ParameterVector ParameterVector;

 public:

  /**
   * scale the transform with the given value
   */
  static Transform Scale(const Transform&, float);

  /**
   * convert the parameter vector to a matrix
   */
  static Transform ParamsToMatrix(const ParameterVector&);

  /**
   * convert the matrix to a parameter vector
   */
  static ParameterVector MatrixToParams(const Transform&);

  /**
   * solve the linear system
   */
  static ParameterVector Solve(const Hessian&, const Gradient&);

  /**
   */
  static Jacobian ComputeJacobian(float x, float y, float Ix, float Iy,
                                  float s = 1.0f, float c1=0.0f, float c2=0.0f);

  static inline void ComputeJacobian(Eigen::Ref<Jacobian> J, float x, float y,
                                     float Ix, float Iy, float s = 1.0f,
                                     float c1 = 0.0f, float c2 = 0.0f)
  {
    J = Affine::ComputeJacobian(x, y, Ix, Iy, s, c1, c2);
  }
}; // Affine

}; // bp

#endif // BITPLANES_CORE_AFFINE_H
