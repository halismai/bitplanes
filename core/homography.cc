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

#include "bitplanes/core/homography.h"

namespace bp {


auto Homography::Scale(const Transform& T, float scale) -> Transform
{
  Transform S(Transform::Identity()), S_i(Transform::Identity());

  S(0,0) = scale; S_i(0,0) = 1.0f / scale;
  S(1,1) = scale; S_i(1,1) = 1.0f / scale;

  return S * T * S_i;
}

auto Homography::MatrixToParams(const Transform& H) -> ParameterVector
{
  const Transform L = H.log();

  ParameterVector p;
  p[0] = L(0,2);
  p[1] = L(1,2);
  p[2] = -L(1,0);
  p[3] = -3*L(2,2)/2;
  p[4] = -1*(L(1,1) - p[3]/3);
  p[5] = L(0,1) - p[2];
  p[6] = L(2,0);
  p[7] = L(2,1);

  return p;
}

auto Homography::ParamsToMatrix(const ParameterVector& p) -> Transform
{
  Transform H;
  H <<
      p[3]/3 + p[4], p[2]+p[5]    , p[0],
      -p[2]        , p[3]/3 - p[4], p[1],
      p[6]         , p[7]         , -2*p[3]/3;

  return H.exp();
}

auto Homography::Solve(const Hessian& A, const Gradient& b) -> ParameterVector
{
  return -A.ldlt().solve(b);
}


auto Homography::ComputeJacobian(float x, float y, float Ix, float Iy,
                                 float s, float c1, float c2) -> Jacobian
{
  Jacobian J;
  J <<
      Ix/s,
      Iy/s,
      Iy*(c1 - x) - Ix*(c2 - y),
      Ix*x - Iy*c2 - Ix*c1 + Iy*y,
      Iy*(c2 - y) - Ix*(c1 - x),
      -Ix*(c2 - y),
      -s*(c1 - x)*(Ix*c1 + Iy*c2 - Ix*x - Iy*y),
      -s*(c2 - y)*(Ix*c1 + Iy*c2 - Ix*x - Iy*y);

  return J;
}

#if 0
void Homography::ComputeJacobian(Eigen::Ref<Jacobian> J, float x, float y,
                                 float Ix, float Iy, float s, float c1, float c2)
{
  /*
  J[0] = Ix/s;
  J[1] = Iy/s;
  J[2] = Iy*(c1 - x) - Ix*(c2 - y);
  J[3] = Ix*x - Iy*c2 - Ix*c1 + Iy*y;
  J[4] = Iy*(c2 - y) - Ix*(c1 - x);
  J[5] = -Ix*(c2 - y);
  J[6] = -s*(c1 - x)*(Ix*c1 + Iy*c2 - Ix*x - Iy*y);
  J[7] = -s*(c2 - y)*(Ix*c1 + Iy*c2 - Ix*x - Iy*y);
  */
  J = Homography::ComputeJacobian(x, y, Ix, Iy, s, c1, c2);
}
#endif

} // bp

