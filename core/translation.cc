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

#include "bitplanes/core/translation.h"

namespace bp {

auto Translation::Scale(const Transform& T, float scale) -> Transform
{
  return T * scale;
}

auto Translation::MatrixToParams(const Transform& H) -> ParameterVector
{
  ParameterVector p(H(0,2), H(1,2));
  return p;
}

auto Translation::ParamsToMatrix(const ParameterVector& p) -> Transform
{
  Transform T;
  T <<
      1.0, 0.0, p[0],
      0.0, 1.0, p[1],
      0.0, 0.0, 1.0;
  return T;
}

auto Translation::Solve(const Hessian& A, const Gradient& b) -> ParameterVector
{
  return -A.ldlt().solve(b);
}

auto Translation::ComputeWarpJacobian(float /*x*/, float /*y*/, float /*s*/,
                                      float /*c1*/, float /*c2*/) -> WarpJacobian
{
  WarpJacobian Jw;
  Jw << 1.0, 1.0f;
  return Jw;
}


} // bp


