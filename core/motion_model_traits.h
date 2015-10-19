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

#ifndef BITPLANES_CORE_MOTION_MODEL_TRAITS_H
#define BITPLANES_CORE_MOTION_MODEL_TRAITS_H

#include "bitplanes/core/types.h"

namespace bp {

class Homography;
class Affine;
class Translation;

template <class> struct motion_model_traits;

template<> struct motion_model_traits<Homography>
{
  static constexpr int DOF = 8;

  typedef Eigen::Matrix<float, 3, 3>         Transform;
  typedef Eigen::Matrix<float, DOF, DOF>     Hessian;
  typedef Eigen::Matrix<float, DOF, 1>       ParameterVector;
  typedef Eigen::Matrix<float, 1, DOF>       Jacobian;
  typedef ParameterVector                    Gradient;
  typedef Eigen::Matrix<float, Dynamic, DOF> JacobianMatrix;
  typedef Eigen::Matrix<float, 2, DOF>       WarpJacobian;
};

template<> struct motion_model_traits<Affine>
{
  static constexpr int DOF = 6;

  typedef Eigen::Matrix<float, 3, 3>         Transform;
  typedef Eigen::Matrix<float, DOF, DOF>     Hessian;
  typedef Eigen::Matrix<float, DOF, 1>       ParameterVector;
  typedef Eigen::Matrix<float, 1, DOF>       Jacobian;
  typedef ParameterVector                    Gradient;
  typedef Eigen::Matrix<float, Dynamic, DOF> JacobianMatrix;
  typedef Eigen::Matrix<float, 2, DOF>       WarpJacobian;
};

template<> struct motion_model_traits<Translation>
{
  static constexpr int DOF = 2;

  typedef Eigen::Matrix<float, 3, 3>         Transform;
  typedef Eigen::Matrix<float, DOF, DOF>     Hessian;
  typedef Eigen::Matrix<float, DOF, 1>       ParameterVector;
  typedef Eigen::Matrix<float, 1, DOF>       Jacobian;
  typedef ParameterVector                    Gradient;
  typedef Eigen::Matrix<float, Dynamic, DOF> JacobianMatrix;
  typedef Eigen::Matrix<float, 2, DOF>       WarpJacobian;
};

}; // bp

#endif // BITPLANES_CORE_MOTION_MODEL_TRAITS_H

