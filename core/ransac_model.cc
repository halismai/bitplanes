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

#include "bitplanes/core/ransac_model.h"
#include "bitplanes/utils/error.h"
#include <iostream>

#include <Eigen/SVD>
#include <Eigen/Dense>

namespace bp {

template <class PointVector> inline
Eigen::Matrix<float,3,3> FitHomography(const PointVector& X1, const PointVector& X2)
{
  THROW_ERROR_IF( X1.size() != X2.size(), "size mismatch" );

  Eigen::Matrix<float, Eigen::Dynamic, 9> A(3*X1.size(), 9);
  Eigen::Matrix<float,1,3> a, b;
  Eigen::Matrix<float,1,3> O(0,0,0);

  for(size_t i = 0; i < X1.size(); ++i)
  {
    const auto& x1 = X1[i];
    const auto& x2 = X2[i];

    const Eigen::Matrix<float,1,3> X = x1.transpose();
    const float x = x2[0], y = x2[1], w = x2[2];

    A.row(3*i+0) << O, -w*X, y*X;
    A.row(3*i+1) << w*X, O, -x*X;
    A.row(3+i+2) << -y*X, x*X, O;
  }

  Eigen::JacobiSVD<decltype(A)> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  svd.computeV();
  Eigen::Matrix<float,9,9> V = svd.matrixV();

  Eigen::Matrix<float,3,3> ret(V.data() + 8*9);
  return ret;
}

template <class PointVector, class Indices> static inline
void CopyCorrespondences(const typename RansacHomography::CorrespondencesType& corrs,
                         const Indices& inds, PointVector& x1, PointVector& x2)
{
  const size_t N = x1.size();

  THROW_ERROR_IF( N > corrs.size(), "indices.size() > correspondences.size()");

  x1.resize(N);
  x2.resize(N);

  for(size_t i = 0; i < N; ++i)
  {
    const auto ii = inds[i];
    x1[i] = corrs[ii].x1;
    x2[i] = corrs[ii].x2;
  }

}

auto RansacHomography::run(const SampleIndices& inds) const -> Result
{
  typedef EigenStdVector<typename CorrespondenceType::Point>::type PointVector;

  const size_t N = inds.size();
  PointVector x1(N), x2(N);
  CopyCorrespondences(_corrs, inds, x1, x2);

  return FitHomography(x1, x2);
}

auto RansacHomography::fitFinal(const Result& H_init, const Indices& inds) const -> Result
{
  const size_t N = inds.size();

  if(N >= MinSampleSize)
  {
    PointVector x1(N), x2(N);
    CopyCorrespondences(_corrs, inds, x1, x2);
    return FitHomography(x1, x2);
  } else
  {
    return H_init;
  }
}

auto RansacHomography::findInliers(const Result& H, float t) const -> Indices
{
  Indices ret;
  ret.reserve(size());

  auto Normalize = [=](const Eigen::Matrix<float, 3, 1>& x)
  {
    Eigen::Matrix<float,3,1> ret(x);
    ret /= x[2];
    return ret;
  };

  const Result H_inv = H.inverse();
  for(size_t i = 0; i < _corrs.size(); ++i)
  {
    float d =
        (Normalize(_corrs[i].x1) - Normalize(H_inv*_corrs[i].x2)).squaredNorm() +
        (Normalize(_corrs[i].x2) - Normalize(H*_corrs[i].x1)).squaredNorm();
    if(d < t)
      ret.push_back(i);
  }

  return ret;
}

}

