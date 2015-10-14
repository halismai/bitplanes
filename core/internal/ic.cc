/*
  This file is part of bitplanes.

  the Free Software Foundation, either version 3 of the License, or
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

#include "bitplanes/core/internal/ic.h"
#include "bitplanes/core/internal/optim_common.h"

#include "bitplanes/core/homography.h"
#include "bitplanes/core/translation.h"
#include "bitplanes/core/affine.h"

#include "bitplanes/utils/timer.h"

#include <limits>
#include <iostream>

#include <opencv2/core/core.hpp>

namespace bp {

template <class M> inline
void InverseCompositionalImpl<M>::
setTemplate(const cv::Mat& I, const cv::Rect& bbox)
{
  Tracker::Impl::setTemplate(I, bbox);

  this->computeChannels(I, bbox, this->_channels);
  this->setChannelData();
  this->allocateInterpMaps(bbox.size());

  _hessian.setZero();
  for(const auto& c : this->_channel_data)
    _hessian.noalias() += (c.jacobian().transpose() * c.jacobian());
}

template <class M> inline
float InverseCompositionalImpl<M>::
linearize(const cv::Mat& I, const Transform& T)
{
  this->computeResiduals(I, T); // warp and compute residuals

  _gradient.setZero();
  for(size_t i = 0; i < this->_channel_data.size(); ++i)
    _gradient.noalias() += this->_channel_data[i].jacobian().transpose() *
        this->_residuals[i];

  return _gradient.template lpNorm<Eigen::Infinity>();
}

template <class M> inline
Result InverseCompositionalImpl<M>::
track(const cv::Mat& image, const Transform& T_init)
{
  Timer timer;
  Result ret(T_init);

  std::cout << ret.T << std::endl;

  auto g_norm = this->linearize(image, ret.T);

  const auto p_tol = this->_alg_params.parameter_tolerance,
        f_tol = this->_alg_params.function_tolerance,
        sqrt_eps = std::sqrt(std::numeric_limits<float>::epsilon()),
        tol_opt = 1e-4f * p_tol, rel_factor = std::max(sqrt_eps, g_norm);

  const auto max_iters = this->_alg_params.max_iterations;
  const auto verbose = this->_alg_params.verbose;

  if(g_norm < tol_opt*rel_factor) {
    if(verbose)
      printf("initial value is optimal %g < %g\n", g_norm, tol_opt*rel_factor);

    ret.final_ssd_error = this->computeSumSquaredErrors(this->_residuals);
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
    const ParameterVector dp = MotionModelType::Solve(_hessian, _gradient);
    const Transform Td = this->_T_inv * MotionModelType::ParamsToMatrix(dp) * this->_T;
    const auto sum_sq = this->computeSumSquaredErrors(this->_residuals);

    g_norm = _gradient.template lpNorm<Eigen::Infinity>();

    {
      const auto dp_norm = dp.norm();
      const auto p_norm = MotionModelType::MatrixToParams(ret.T).norm();

      if(verbose)
        printf("\t%3d/%d F=%g g=%g |dp|=%0.2e\n",
               it, max_iters, sum_sq, g_norm, dp_norm);

      has_converged = TestConverged(dp_norm, p_norm, p_tol,
                                    g_norm, tol_opt, rel_factor,
                                    sum_sq, old_sum_sq, f_tol,
                                    sqrt_eps, it, max_iters, verbose,
                                    ret.status);
      old_sum_sq = sum_sq;
    }

    if(!has_converged) {
      ret.T = ret.T * Td;
      this->linearize(image, ret.T);
    }
  }


  ret.time_ms = timer.stop().count();
  ret.num_iterations = it;
  ret.final_ssd_error = old_sum_sq;
  if(ret.status == OptimizerStatus::NotStarted)
    ret.status = OptimizerStatus::MaxIterations;

  return ret;
}

template class InverseCompositionalImpl<Homography>;
template class InverseCompositionalImpl<Translation>;
template class InverseCompositionalImpl<Affine>;

}; // bp

