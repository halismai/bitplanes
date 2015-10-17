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

#ifndef BITPLANES_CORE_BITPLANES_TRACKER_H
#define BITPLANES_CORE_BITPLANES_TRACKER_H

#include "bitplanes/core/config.h"
#include "bitplanes/core/types.h"
#include "bitplanes/core/algorithm_parameters.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/internal/channel_data_dense.h"
#include "bitplanes/core/internal/normalization.h"
#include "bitplanes/core/internal/optim_common.h"
#include "bitplanes/core/internal/imwarp.h"
#include "bitplanes/core/internal/census.h"
#include "bitplanes/utils/timer.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core.hpp>

#include <limits>
#include <iostream>
#include <array>

namespace bp {

template <class M>
class BitplanesTracker
{
 public:
  typedef Matrix33f Transform;
  typedef MotionModel<M> MotionModelType;

  typedef typename MotionModelType::ParameterVector ParameterVector;
  typedef typename MotionModelType::Jacobian        Jacobian;
  typedef typename MotionModelType::JacobianMatrix  JacobianMatrix;
  typedef typename MotionModelType::Hessian         Hessian;
  typedef typename MotionModelType::Gradient        Gradient;

 public:
  inline BitplanesTracker(AlgorithmParameters p = AlgorithmParameters())
      : _alg_params(p), _T(Matrix33f::Identity()), _T_inv(Matrix33f::Identity()) {}

  void setTemplate(const cv::Mat& image, const cv::Rect& bbox);

  Result track(const cv::Mat& image, const Transform& T_init = Transform::Identity());

 protected:
  AlgorithmParameters _alg_params;
  cv::Rect _bbox;
  cv::Mat _interp_maps[2];
  cv::Mat _Iw, _I0, _C;
  Matrix33f _T, _T_inv;

  Hessian _hessian;
  Gradient _gradient;
  JacobianMatrix _jacobian;
  Vector_<float> _pixels;
  Vector_<float> _residuals;

  float linearize(const cv::Mat&, const Transform& T_init);

  void setNormalization(const cv::Rect&)
  {
    _T.setIdentity();
    _T_inv.setIdentity();
  }

  int _interp = cv::INTER_AREA;
}; // BitplanesTracker

template<> void BitplanesTracker<Homography>::setNormalization(const cv::Rect& bbox)
{
  HartlyNormalization(bbox, _T, _T_inv);
}


template <class M> inline
void BitplanesTracker<M>::setTemplate(const cv::Mat& image, const cv::Rect& bbox)
{
  setNormalization(bbox);
  _bbox = bbox;

  {
    const int npts = bbox.area() * 8;
    _jacobian.resize(npts, M::DOF);
    _residuals.resize(npts);
    _pixels.resize(npts);
  }

  image.copyTo(_I0);
  cv::GaussianBlur(_I0(bbox), _I0(bbox), cv::Size(3,3), _alg_params.sigma);
  simd::CensusTransform(_I0, bbox, _C);
}

}; // bp

#endif // BITPLANES_CORE_BITPLANES_TRACKER_H

