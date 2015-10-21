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


#ifndef BITPLANES_INTERNAL_BITPLANES_SPARSE_DATA_H
#define BITPLANES_INTERNAL_BITPLANES_SPARSE_DATA_H

#include "bitplanes/core/types.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/internal/cvfwd.h"

#include <opencv2/core.hpp>

namespace bp {

template <class M>
class BitPlanesSparseData
{
 public:
  typedef MotionModel<M> MotionModelType;
  typedef typename MotionModelType::Gradient Gradient;
  typedef typename MotionModelType::Jacobian Jacobian;
  typedef typename MotionModelType::Hessian Hessian;
  typedef typename MotionModelType::Transform Transform;
  typedef typename MotionModelType::ParameterVector ParameterVector;
  typedef typename MotionModelType::JacobianMatrix JacobianMatrix;
  typedef typename EigenStdVector<Jacobian>::type JacobianVector;
  typedef Eigen::Matrix<float,3,1> Point;
  typedef typename EigenStdVector<Point>::type PointVector;

 public:
  BitPlanesSparseData() = default;

  Hessian set(const cv::Mat& I, const cv::Rect& roi,
              float = 1.0, float = 0.0, float = 0.0);

  void computeResiduals(const Matrix33f& T, const cv::Mat& I,
                        Vector_<float>& residuals) const;

  float linearize(const cv::Mat& I, const Matrix33f& T, Gradient& g) const;

  inline const JacobianMatrix& jacobian() const { return _jacobian; }

  inline Transform toTransform(const ParameterVector dp) const
  {
    return _T_inv * MotionModelType::ParamsToMatrix(dp) * _T;
  }

 protected:
  std::vector<uint8_t> _flags;
  Vector_<uint8_t> _pixels;
  JacobianMatrix _jacobian;
  PointVector _points;
  cv::Rect _roi;

  Transform _T=Transform::Identity(),
            _T_inv=Transform::Identity();
}; // BitPlanesSparseData

}; // bp

#endif // BITPLANES_INTERNAL_BITPLANES_SPARSE_DATA_H
