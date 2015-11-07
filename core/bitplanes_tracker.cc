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

#include "bitplanes/core/bitplanes_tracker.h"
#include "bitplanes/core/internal/normalization.h"
#include "bitplanes/core/internal/optim_common.h"
#include "bitplanes/core/internal/imwarp.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/timer.h"
#include "bitplanes/utils/error.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/LU>

namespace bp {

template <class M>
BitplanesTracker<M>::BitplanesTracker(AlgorithmParameters p)
  : _alg_params(p), _cdata(p.subsampling)
  , _T(Matrix33f::Identity()), _T_inv(Matrix33f::Identity())
  , _interp(cv::INTER_LINEAR) {}

template <class M>
void BitplanesTracker<M>::setTemplate(const cv::Mat& image, const cv::Rect& bbox)
{
  image.copyTo(_I);
  smoothImage(_I, bbox);

  _cdata.getCoordinateNormalization(bbox, _T, _T_inv);
  _bbox = bbox;
  _cdata.set(_I, bbox, _T(0,0), _T_inv(0,2), _T_inv(1,2));

  _solver.compute(-_cdata.hessian());
}

template <class M>
Result BitplanesTracker<M>::track(const cv::Mat& image, const Transform& T_init)
{
  image.copyTo(_I);
  smoothImage(_I, _bbox);

  Result ret(T_init);
  Timer timer;

  auto g_norm = this->linearize(_I, ret.T);
  const auto p_tol = this->_alg_params.parameter_tolerance,
        f_tol = this->_alg_params.function_tolerance,
        sqrt_eps = std::sqrt(std::numeric_limits<float>::epsilon()),
        tol_opt = 1e-4f * f_tol, rel_factor = std::max(sqrt_eps, g_norm);

  const auto max_iters = this->_alg_params.max_iterations;
  const auto verbose = this->_alg_params.verbose;

  if(verbose) {
    printf("\n                                        First-Order         Norm of \n"
           " Iteration  Func-count    Residual       optimality            step\n");
    printf(" %5d       %5d   %13.6g    %12.3g\n", 0, 1, _residuals.squaredNorm(), g_norm);
  }

  if(g_norm < tol_opt*rel_factor) {
    if(verbose)
      printf("initial value is optimal %g < %g\n", g_norm, tol_opt*rel_factor);

    ret.final_ssd_error = _residuals.squaredNorm();
    ret.first_order_optimality = g_norm;
    ret.time_ms = timer.stop().count();
    ret.num_iterations = 1;
    ret.status = OptimizerStatus::FirstOrderOptimality;
    return ret;
  }

  float old_sum_sq = std::numeric_limits<float>::max();
  bool has_converged = false;
  int it = 1;
  while(!has_converged && it++ < max_iters)
  {
    const ParameterVector dp = _solver.solve(_gradient);
    const auto sum_sq = _residuals.squaredNorm();
    {
      const auto dp_norm = dp.norm();
      const auto p_norm = MotionModelType::MatrixToParams(ret.T).norm();

      if(verbose) {
        printf(" %5d       %5d   %13.6g    %12.3g    %12.6g\n",
               it, 1 + it, sum_sq, g_norm, dp_norm);
      }

      has_converged = TestConverged(dp_norm, p_norm, p_tol,
                                    g_norm, tol_opt, rel_factor,
                                    sum_sq, old_sum_sq, f_tol,
                                    sqrt_eps, it, max_iters, verbose,
                                    ret.status);
      old_sum_sq = sum_sq;
    }

    const Transform Td = _T_inv * MotionModelType::ParamsToMatrix(dp) * _T;
    ret.T = Td * ret.T;

    if(!has_converged) {
      g_norm = this->linearize(_I, ret.T);
    }
  }

  ret.time_ms = timer.stop().count();
  ret.num_iterations = it;
  ret.final_ssd_error = old_sum_sq;
  ret.first_order_optimality = g_norm;
  if(ret.status == OptimizerStatus::NotStarted) {
    ret.status = OptimizerStatus::MaxIterations;
    if(verbose) {
      std::cout << "Max iterations reached\n";
    }
  }

  if(verbose) {
    printf("\n\n");
  }

  return ret;
}

template <class M> inline
float BitplanesTracker<M>::linearize(const cv::Mat& I, const Transform& T)
{
  _cdata.warpImage(I, T, _bbox, _Iw, _interp, 0.0f);
  _cdata.computeResiduals(_Iw, _residuals);

  _gradient = _cdata.jacobian().transpose() * _residuals;

  return _gradient.template lpNorm<Eigen::Infinity>();
}

template <class M> inline
void BitplanesTracker<M>::smoothImage(cv::Mat& I, const cv::Rect& /*roi*/)
{
  if(_alg_params.sigma > 0)
    cv::GaussianBlur(I, I, cv::Size(), _alg_params.sigma);
}


template class BitplanesTracker<Homography>;

}; // bp
