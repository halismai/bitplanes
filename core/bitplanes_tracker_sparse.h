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

#ifndef BITPLANES_CORE_BITPLANES_TRACKER_SPARSE_H
#define BITPLANES_CORE_BITPLANES_TRACKER_SPARSE_H

#include "bitplanes/core/types.h"
#include "bitplanes/core/algorithm_parameters.h"
#include "bitplanes/core/internal/bitplanes_sparse_data.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/internal/optim_common.h"
#include "bitplanes/utils/timer.h"

#include <Eigen/Cholesky>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <cmath>

namespace bp {

template <class M, class Solver = Eigen::LDLT<typename M::Hessian>>
class BitplanesTrackerSparse
{
 public:
  typedef BitPlanesSparseData<M> TrackerData;
  typedef typename TrackerData::MotionModelType MotionModelType;
  typedef typename MotionModelType::Gradient    Gradient;
  typedef typename MotionModelType::Hessian     Hessian;
  typedef typename MotionModelType::Transform   Transform;
  typedef typename MotionModelType::ParameterVector ParameterVector;

  //typedef Eigen::LDLT<Hessian> Solver;

 public:
  inline BitplanesTrackerSparse(AlgorithmParameters p = AlgorithmParameters())
      : _alg_params(p) {}

  inline void setTemplate(const cv::Mat& I, const cv::Rect& roi)
  {
    I.copyTo(_I);
    cv::GaussianBlur(_I(roi), _I(roi), _k_size, _k_sigma);
    _solver.compute( -_data.set(_I, roi, -1) );

    int B = 0.1*std::max(roi.height, roi.width);
    _search_roi = cv::Rect(std::max(0, roi.x-B),
                           std::max(0, roi.y-B),
                           std::min(roi.width+B, _I.cols-1),
                           std::min(roi.height+B,  _I.rows-1));

    //cv::imshow("_I", _I); cv::waitKey();
  }

  Result track(const cv::Mat&, const Transform& T_init = Transform::Identity());

 protected:
  AlgorithmParameters _alg_params;
  TrackerData _data;
  Gradient _gradient;
  Vector_<float> _residuals;
  Solver _solver;
  cv::Rect _search_roi;

  cv::Size _k_size = cv::Size();
  double _k_sigma = 0.85; //1.2;

  inline float linearize(const cv::Mat& I, const Transform& T) {
    return _data.linearize(I, T, _gradient);
  }

  inline float gradientNorm() const {
    return _gradient.template lpNorm<Eigen::Infinity>();
  }

 private:
  cv::Mat _I;
}; // BitplanesTrackerSparse


template <class M, class S> inline Result
BitplanesTrackerSparse<M,S>::track(const cv::Mat& I, const Transform& T_init)
{
  Result ret(T_init);
  Timer timer;

  I.copyTo(_I);
  cv::GaussianBlur(_I(_search_roi), _I(_search_roi), _k_size, _k_sigma);

  auto sum_sq = linearize(_I, ret.T);
  auto g_norm = gradientNorm();
  auto p_tol = _alg_params.parameter_tolerance,
       f_tol = _alg_params.function_tolerance,
       sqrt_eps = std::sqrt(std::numeric_limits<float>::epsilon()),
       tol_opt = 1e-4f * p_tol,
       rel_factor = std::max(g_norm, sqrt_eps);
  auto max_iters = _alg_params.max_iterations;
  auto verbose = _alg_params.verbose;

  bool has_converged = g_norm < tol_opt * rel_factor;

  if(has_converged) {
    ret.status = OptimizerStatus::FirstOrderOptimality;
    if(verbose)
      printf("Initial value is optimal\n");
    ret.final_ssd_error = sum_sq;
    ret.first_order_optimality = g_norm;
    ret.num_iterations = 0;
    ret.time_ms = timer.stop().count();
    return ret;
  }

  auto old_sum_sq = std::numeric_limits<float>::max();
  int it = 0;
  while( !has_converged && it++ < max_iters )
  {
    const ParameterVector dp = _solver.solve(_gradient);

    const float dp_norm = dp.norm();
    const float p_norm = MotionModelType::MatrixToParams(ret.T).norm();

    has_converged = TestConverged(dp_norm, p_norm, p_tol, g_norm, tol_opt,
                                  rel_factor, sum_sq, old_sum_sq, f_tol,
                                  sqrt_eps, it, max_iters, verbose, ret.status);

    old_sum_sq = sum_sq;

    ret.T *= _data.toTransform(dp);

    if(!has_converged) {
      sum_sq = linearize(_I, ret.T);
      g_norm = gradientNorm();
      if(verbose)
        printf("\t%3d/%d F=%g g=%g |dp|=%0.2e\n",
               it, max_iters, sum_sq, g_norm, dp_norm);
    }
  }

  ret.final_ssd_error = sum_sq;
  ret.first_order_optimality = g_norm;
  ret.num_iterations = it;
  ret.time_ms = timer.stop().count();
  if(ret.num_iterations > max_iters) ret.status = OptimizerStatus::MaxIterations;

  return ret;
}

}; // bp

#endif // BITPLANES_CORE_BITPLANES_TRACKER_SPARSE_H
