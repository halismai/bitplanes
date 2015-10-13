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

#include <unsupported/Eigen/MatrixFunctions> // for exp and log
#include <Eigen/Cholesky>

#include "bitplanes/core/affine.h"

namespace bp {


auto Affine::Scale(const Transform& T, float scale) -> Transform
{
  Transform S; //(Transform::Identity()), S_i(Transform::Identity());
  // TODO

  return S;
}

auto Affine::MatrixToParams(const Transform& H) -> ParameterVector
{
  ParameterVector p;
  // TODO
  return p;
}

auto Affine::ParamsToMatrix(const ParameterVector& p) -> Transform
{
  Transform H;
  // TODO
  return H;
}

auto Affine::Solve(const Hessian& A, const Gradient& b) -> ParameterVector
{
  return -A.ldlt().solve(b);
}

auto Affine::ComputeJacobian(float x, float y, float Ix, float Iy,
                                 float s, float c1, float c2) -> Jacobian
{
  Jacobian J;
  // TODO

  return J;
}

} // bp

