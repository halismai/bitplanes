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

#ifndef BITPLANES_CORE_INTERNAL_FIT_HOMOGRAPHY_H
#define BITPLANES_CORE_INTERNAL_FIT_HOMOGRAPHY_H

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "bitplanes/utils/error.h"

#include <iostream>
#include <iomanip>

namespace bp {

template <class PointVector> inline
Eigen::Matrix<typename PointVector::value_type::Scalar,
    PointVector::value_type::RowsAtCompileTime,
    PointVector::value_type::RowsAtCompileTime>
NormalizePoints(const PointVector& pts, PointVector& dst)
{
  typedef typename PointVector::value_type PointType;
  typedef typename PointType::Scalar Scalar;

  static_assert(PointType::RowsAtCompileTime == 3, "points must be 3x1");

  PointType c(PointType::Zero());
  for(const auto& p : pts)
    c += p;
  c /= (Scalar) pts.size();

  Scalar m = Scalar(0);
  for(const auto& p : pts)
    m += (p - c).norm();
  m /= (Scalar) pts.size();
  auto s = sqrt(Scalar(2)) / std::max(m, Scalar(1e-6));

  Eigen::Matrix<typename PointVector::value_type::Scalar,
    PointVector::value_type::RowsAtCompileTime,
    PointVector::value_type::RowsAtCompileTime> ret;

  ret.setZero();
  for(int i = 0; i < ret.rows() - 1; ++i) {
      ret(i, i) = s;
      ret(i, ret.cols()-1) = -s*c[i];
  }
  ret(ret.rows()-1, ret.rows()-1) = 1.0;

  return ret;
}

/**
 * Fit a homography given a set of corresponding points
 */
template <class PointVector> inline
Eigen::Matrix<typename PointVector::value_type::Scalar,3,3>
FitHomography(const PointVector& X1, const PointVector& X2)
{
  THROW_ERROR_IF( X1.size() != X2.size(), "size mismatch" );
  THROW_ERROR_IF( X1.size() == 0, "empty" );
  THROW_ERROR_IF( X1.size() < 4, "need at least 4 points to fit homography" );
  THROW_ERROR_IF( X1[0].rows() != 3, "data must be 3xN" );

  typedef typename PointVector::value_type::Scalar T;

  const int N = X1.size();
  Eigen::Matrix<T, Eigen::Dynamic, 9> A(3*N, 9);
  Eigen::Matrix<T,1,3> a, b;
  Eigen::Matrix<T,1,3> O(0,0,0);

  for(int i = 0; i < N; ++i)
  {
    const auto& x1 = X1[i];
    const auto& x2 = X2[i];

    const Eigen::Matrix<T,1,3> X = x1.transpose();
    const float x = x2[0], y = x2[1], w = x2[2];

    A.template block<1,3>(3*i, 0) = O;
    A.template block<1,3>(3*i, 3) = -w*X;
    A.template block<1,3>(3*i, 6) = y*X;

    A.template block<1,3>(3*i+1, 0) = w*X;
    A.template block<1,3>(3*i+1, 3) = O;
    A.template block<1,3>(3*i+1, 6) = -x*X;

    A.template block<1,3>(3*i+2, 0) = -y*X;
    A.template block<1,3>(3*i+2, 3) = x*X;
    A.template block<1,3>(3*i+2, 6) = O;
  }

#if defined(EIGEN_DEFAULT_TO_ROW_MAJOR)
  return Eigen::Matrix<T,3,3>(
      Eigen::JacobiSVD<decltype(A)>(A, Eigen::ComputeFullV).matrixV().data() + 8*9);
#else
  return Eigen::Matrix<T,3,3>(
      Eigen::JacobiSVD<decltype(A)>(A, Eigen::ComputeFullV).matrixV().data() + 8*9).transpose();
#endif

}

}; // bp

#endif // BITPLANES_CORE_INTERNAL_FIT_HOMOGRAPHY_H

