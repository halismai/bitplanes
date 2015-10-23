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
#include "bitplanes/core/internal/bitplanes_channel_data.h"
#include "bitplanes/core/internal/bitplanes_channel_data_packed.h"
#include <opencv2/core.hpp>

#include <limits>
#include <iostream>
#include <fstream>
#include <array>

#include <Eigen/Cholesky>

namespace bp {

template <class M>
class BitplanesTracker
{
 public:
  typedef Matrix33f Transform;
  typedef MotionModel<M> MotionModelType;

  typedef typename MotionModelType::Hessian         Hessian;
  typedef typename MotionModelType::Gradient        Gradient;
  typedef typename MotionModelType::ParameterVector ParameterVector;

  typedef BitPlanesChannelDataPacked<M> ChannelDataType;

 public:
  BitplanesTracker(AlgorithmParameters p = AlgorithmParameters());

  void setTemplate(const cv::Mat& image, const cv::Rect& bbox);

  Result track(const cv::Mat& image, const Transform& T_init = Transform::Identity());

 protected:
  AlgorithmParameters _alg_params;
  ChannelDataType _cdata;
  cv::Rect _bbox;
  cv::Mat _I0, _I1;
  cv::Mat _Iw;
  cv::Mat _interp_maps[2];
  Matrix33f _T, _T_inv;
  Gradient _gradient;
  Vector_<float> _residuals;

  Eigen::LDLT<Hessian> _ldlt;

  float linearize(const cv::Mat&, const Transform& T_init);

  void setNormalization(const cv::Rect&)
  {
    _T.setIdentity();
    _T_inv.setIdentity();
  }

  void smoothImage(cv::Mat& I, const cv::Rect& roi);

  int _interp;
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // BitplanesTracker


}; // bp

#endif // BITPLANES_CORE_BITPLANES_TRACKER_H

