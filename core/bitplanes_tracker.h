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
#include "bitplanes/core/internal/bitplanes_channel_data_base.h"
#include "bitplanes/core/internal/bitplanes_channel_data_subsampled.h"
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

  typedef typename Eigen::LDLT<Hessian> Solver;

  typedef BitPlanesChannelDataSubSampled<M> ChannelDataType;

 public:
  /**
   */
  BitplanesTracker(AlgorithmParameters p = AlgorithmParameters());

  /**
   * Sets the template
   *
   * \param image the template image (I_0)
   * \param bbox  location of the template in image
   */
  void setTemplate(const cv::Mat& image, const cv::Rect& bbox);

  /**
   * Tracks the template that was set during the call setTemplate
   *
   * \param image the input image (I_1)
   * \param T_init initialization of the transform
   */
  Result track(const cv::Mat& image, const Transform& T_init = Transform::Identity());

 protected:
  /**
   * Performs the linearization step, which is:
   *  - warp the image
   *  - re-compute the multi-channel descriptors
   *  - compute the cost function gradient (J^T * error)
   */
  float linearize(const cv::Mat&, const Transform& T_init);

  /**
   * applies smoothing to the image at the specified ROI
   */
  void smoothImage(cv::Mat& I, const cv::Rect& roi);

 protected:
  AlgorithmParameters _alg_params; //< AlgorithmParameters
  ChannelDataType _cdata;          //< holds the multi-channel data
  cv::Rect _bbox;                  //< the template's bounding box
  cv::Mat _I, _Iw;                 //< buffers for input image and warped image
  Matrix33f _T, _T_inv;            //< normalization matrices
  Gradient _gradient;              //< gradient of the cost function
  Vector_<float> _residuals;       //< vector of residuals
  Solver _solver;                  //< the linear solver
  int _interp;                     //< interpolation, e.g. cv::INTER_LINEAR

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // BitplanesTracker
}; // bp

#endif // BITPLANES_CORE_BITPLANES_TRACKER_H


