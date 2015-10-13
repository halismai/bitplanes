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

#include "bitplanes/core/tracker.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/affine.h"
#include "bitplanes/core/translation.h"
#include "bitplanes/core/internal/mc_extractor.h"
#include "bitplanes/core/internal/channel_data.h"
#include "bitplanes/core/internal/SmallVector.h"

#include "bitplanes/utils/utils.h"
#include "bitplanes/utils/error.h"
#include "bitplanes/utils/timer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if defined(BITPLANES_WITH_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <iostream>
#include <limits>

namespace bp {

static inline bool
TestConverged(float dp_norm, float p_norm, float x_tol, float g_norm,
              float tol_opt, float rel_factor, float new_f, float old_f,
              float f_tol, float sqrt_eps, int it, int max_iters, bool verbose,
              OptimizerStatus& status)
{
  if(it > max_iters)
  {
    if(verbose)
      std::cout << "MaxIterations reached\n";

    return true;
  }

  if(g_norm < tol_opt * rel_factor)
  {
    if(verbose)
      std::cout << "First order optimality reached\n";

    status = OptimizerStatus::FirstOrderOptimality;
    return true;
  }

  if(dp_norm < x_tol)
  {
    if(verbose)
      std::cout << "Small abs step\n";

    status = OptimizerStatus::SmallAbsParameters;
    return true;
  }

  if(dp_norm < x_tol * (sqrt_eps * p_norm))
  {
    if(verbose)
      std::cout << "Small change in parameters\n";

    status = OptimizerStatus::SmallParameterUpdate;
    return true;
  }

  if(fabs(old_f - new_f) < f_tol * old_f)
  {
    if(verbose)
      std::cout << "Small relative reduction in error\n";

    status = OptimizerStatus::SmallRelativeReduction;
    return true;
  }

  return false;
}


struct Tracker::Impl
{
  typedef Tracker::Impl Self;

  inline Impl(MotionType m, AlgorithmParameters p)
      :  _alg_params(p), _motion_type(m),
      _mc(MultiChannelExtractor::Create(p.multi_channel_function)),
      _T(Matrix33f::Identity()), _T_inv(Matrix33f::Identity())
  {
    _mc->setSigma(p.sigma);
  }

  inline virtual ~Impl() {}

  virtual inline void setTemplate(const cv::Mat& image, const cv::Rect& box)
  {
    THROW_ERROR_IF(
        box.y < 1 || box.y + box.height >= image.rows ||
        box.x < 1 || box.x + box.width  >= image.cols,
        "Bounding box is outside of image boundaries");

    _bbox = box;
    _points.resize(box.area());
    Vector2f c(0.0f, 0.0f);

    for(int y = box.y, i = 0; y < box.y + box.height; ++y)
      for(int x = box.x; x < box.x + box.width; ++x, ++i)
      {
        _points[i] = Eigen::Vector3f(x, y, 1.0f);
        c += _points[i].head<2>();
      }

    c /= _points.size(); // center of mass
    float m = 0.0f;
    for(const auto& pt : _points) {
      m += (pt.head<2>() - c).norm();
    }
    m /= _points.size();

    float s = sqrt(2.0f) / std::max(m, 1e-6f);

    _T << s, 0, -s*c[0],
          0, s, -s*c[1],
          0, 0, 1;

    _T_inv << 1.0f/s, 0, c[0],
              0, 1.0f/s, c[1],
              0, 0, 1;
  }

  inline float computeSumSquaredErrors(const ResidualsVector& r) const
  {
    float ret = 0.0f;
    for(const auto& e : r)
      ret += e.squaredNorm();
    return ret;
  }

  virtual Result track(const cv::Mat& image, const Transform& T_init) = 0;

  inline void resizeChannelData(size_t n)
  {
    this->_channel_data.resize(n, this->_motion_type);
  }

  inline void computeChannels(const cv::Mat& src, const cv::Rect& bbox)
  {
    this->_mc->operator()(src, bbox, _channels);
  }

  inline void setChannelData()
  {
    const auto s = _T(0,0), c1 = _T_inv(0,2), c2 = _T_inv(1,2);
    this->resizeChannelData(_channels.size());
    size_t i = 0;

#if defined(BITPLANES_WITH_TBB)
    tbb::parallel_for( tbb::blocked_range<size_t>(0, _channels.size()),
                      [=](const tbb::blocked_range<size_t>& r)
                      {
                        for(size_t j = r.begin(); j != r.end(); ++j)
                          _channel_data[j].set(_channels[j], _points, s, c1, c2);
                      });
    i = _channels.size();
#endif

    for(;  i < _channel_data.size(); ++i)
      _channel_data[i].set(_channels[i], _points, s, c1, c2);
  }

  inline void allocateInterpMaps(const cv::Size& size)
  {
    _interp_maps[0].create(size, CV_32F);
    _interp_maps[1].create(size, CV_32F);
  }

  inline void computeResiduals(const cv::Mat& I, const Transform& T)
  {
    const int x_off = _bbox.x, y_off = _bbox.y;

    {
      const auto stride = _interp_maps[0].cols;
      float* x_map = _interp_maps[0].ptr<float>();
      float* y_map = _interp_maps[1].ptr<float>();

      for(const auto& p : _points)
      {
        Vector3f pw = T * p;
        pw *= (1.0f / pw[2]);

        int y = p.y() - y_off,
            x = p.x() - x_off;

        int ii = y*stride + x;

        x_map[ii] = pw.x();
        y_map[ii] = pw.y();
      }
    }

    cv::remap(I, _Iw, _interp_maps[0], _interp_maps[1], _interp, _border, _border_val);

    cv::Rect cw_rect(0, 0, _Iw.cols, _Iw.rows);
    _mc->operator()(_Iw, cw_rect, _channels_warped);

    const auto num_channels = _channels_warped.size();
    assert( num_channels == _channel_data.size() && "Wrong num_channels" );

    _residuals.resize(num_channels);

    size_t i = 0;
#if defined(BITPLANES_WITH_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_channels),
                      [=](const tbb::blocked_range<size_t>& r)
                      {
                        for(size_t j = r.begin(); j != r.end(); ++j)
                          _channel_data[i].computeResiduals(_channels_warped[i],
                                                            _residuals[i]);
                      });
    i = num_channels;
#endif

    for( ; i < num_channels; ++i)
      _channel_data[i].computeResiduals(_channels_warped[i], _residuals[i]);

  }

  static UniquePointer<Self> Create(MotionType m, AlgorithmParameters p);

  AlgorithmParameters _alg_params;
  MotionType _motion_type;
  UniquePointer<MultiChannelExtractor> _mc;
  PointVector _points;
  cv::Rect _bbox;
  cv::Mat _interp_maps[2];
  cv::Mat _Iw;

  Matrix33f _T, _T_inv;

  /** holds the channels (opencv images) */
  typename MultiChannelExtractor::ChannelsVector _channels;
  typename MultiChannelExtractor::ChannelsVector _channels_warped;

  ResidualsVector _residuals;

  /** holds the channel data */
  llvm::SmallVector<ChannelData, 16> _channel_data;

  int _interp = cv::INTER_LINEAR;
  int _border = cv::BORDER_CONSTANT;
  cv::Scalar _border_val = cv::Scalar(0.0);
};

template <class Motion>
struct InverseCompositionalImpl : public Tracker::Impl
{
  typedef MotionModel<Motion> MotionModelType;
  typedef typename MotionModelType::Transform Transform;
  typedef typename MotionModelType::Hessian Hessian;
  typedef typename MotionModelType::Gradient Gradient;
  typedef typename MotionModelType::JacobianMatrix JacobianMatrix;
  typedef typename MotionModelType::ParameterVector ParameterVector;

  inline InverseCompositionalImpl(MotionType m, AlgorithmParameters p)
      : Tracker::Impl(m, p) {}

  virtual ~InverseCompositionalImpl() {}

  inline void setTemplate(const cv::Mat& src, const cv::Rect& bbox)
  {
    Tracker::Impl::setTemplate(src, bbox);

    _T_init.setIdentity();
    this->computeChannels(src, bbox);
    this->setChannelData();
    this->allocateInterpMaps(bbox.size());

    // pre-compute the Hessian for IC
    _hessian.setZero();
    for(const auto& c : this->_channel_data)
      _hessian.noalias() += (c.jacobian().transpose() * c.jacobian());
  }

  inline float linearize(const cv::Mat& I, const Tracker::Transform& T)
  {
    this->computeResiduals(I, T);

    _gradient.setZero();
    for(size_t i = 0; i < this->_residuals.size(); ++i)
    {
      const auto& J = _channel_data[i].jacobian();
      _gradient.noalias() += J.transpose() * this->_residuals[i];
    }

    return _gradient.template lpNorm<Eigen::Infinity>();
  }


  inline Result track(const cv::Mat& image, const Tracker::Transform& T_init)
  {
    Timer timer;
    Result ret(T_init);

    auto g_norm = this->linearize(image, ret.T);
    const auto sqrt_eps = std::sqrt(std::numeric_limits<float>::epsilon());
    const float tol_opt = 1e-4 * _alg_params.parameter_tolerance;
    const float rel_factor = std::max(g_norm, sqrt_eps);

    if(g_norm < tol_opt * rel_factor)
    {
      if(_alg_params.verbose)
        std::cout << "Initial value is optimal" << std::endl;

      ret.final_ssd_error = computeSumSquaredErrors(_residuals);
      ret.first_order_optimality = g_norm;
      ret.time_ms = timer.stop().count();
      ret.status = OptimizerStatus::FirstOrderOptimality;
      return ret;
    }

    auto old_sum_sq = std::numeric_limits<float>::max();
    auto has_converged = false;

    const auto max_iters = _alg_params.max_iterations;
    const auto verbose = _alg_params.verbose;
    const auto p_tol = _alg_params.parameter_tolerance;
    const auto f_tol = _alg_params.function_tolerance;

    int it = 1;
    while( !has_converged && it++ < max_iters )
    {
      const auto dp = MotionModelType::Solve(_hessian, _gradient);
      const auto Td = this->_T_inv * MotionModelType::ParamsToMatrix(dp) * this->_T;
      const auto sum_sq = computeSumSquaredErrors(_residuals);
      const auto dp_norm = dp.norm();
      const auto p_norm = MotionModelType::MatrixToParams(ret.T).norm();
      g_norm = _gradient.template lpNorm<Eigen::Infinity>();

      if(verbose)
        printf("\t%3d/%d F=%g g=%g |dp|=%0.2e\n", it, max_iters, sum_sq, g_norm, dp_norm);

      has_converged = TestConverged(dp_norm, p_norm, p_tol, g_norm, tol_opt, rel_factor,
                                  sum_sq, old_sum_sq, f_tol,
                                  sqrt_eps, it, max_iters, verbose, ret.status);
      old_sum_sq = sum_sq;
      if(!has_converged) {
        ret.T = ret.T * Td;
        linearize(image, ret.T);
      }
    }

    ret.time_ms = timer.stop().count();
    ret.num_iterations = it;
    if(ret.status == OptimizerStatus::NotStarted)
      ret.status = OptimizerStatus::MaxIterations;

    return ret;
  }

  Transform _T_init;
  Hessian _hessian;
  Gradient _gradient;
};

Tracker::Tracker(MotionType m, AlgorithmParameters p)
    : _impl(Tracker::Impl::Create(m, p)) {}

Tracker::~Tracker() {}

void Tracker::setTemplate(const cv::Mat& src, const cv::Rect& bbox)
{
  _impl->setTemplate(src, bbox);
}

Result Tracker::track(const cv::Mat& I, const Transform& T_init)
{
  return _impl->track(I, T_init);
}

/*
Result Tracker::track(const cv::Mat I)
{
  return _impl->track(I);
}*/

UniquePointer<Tracker::Impl>
Tracker::Impl::Create(MotionType m, AlgorithmParameters p)
{
  switch(p.linearizer)
  {
    case AlgorithmParameters::LinearizerType::InverseCompositional:
      {
        switch(m)
        {
          case MotionType::Homography:
            return UniquePointer<Tracker::Impl>(
                new InverseCompositionalImpl<Homography>(m, p));
            break;

          case MotionType::Affine:
            return UniquePointer<Tracker::Impl>(
                new InverseCompositionalImpl<Affine>(m, p));
            break;

          case MotionType::Translation:
            return UniquePointer<Tracker::Impl>(
                new InverseCompositionalImpl<Translation>(m, p));
            break;
        }
      } break;

    default:
      THROW_ERROR("not implemented");
  }

  return UniquePointer<Tracker::Impl>(new InverseCompositionalImpl<Homography>(m, p));
}

} // bp

