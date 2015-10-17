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

#include "bitplanes/core/config.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/affine.h"
#include "bitplanes/core/translation.h"

#include "bitplanes/core/internal/tracker_impl.h"
#include "bitplanes/core/internal/imwarp.h"

#include "bitplanes/utils/utils.h"
#include "bitplanes/utils/error.h"
#include "bitplanes/utils/timer.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#if defined(BITPLANES_WITH_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <iostream>
#include <limits>

#include <Eigen/LU>

namespace bp {

static inline
void HartlyNormalization(const cv::Rect& box, typename Tracker::Transform& T,
                         typename Tracker::Transform& T_inv)
{
  Vector2f c(0.0f, 0.0f);
  for(int y = box.y; y < box.y + box.height; ++y)
    for(int x = box.x; x < box.x + box.width; ++x)
      c += Vector2f(x, y);
  c /= static_cast<float>( box.area() );

  float m = 0.0f;
  for(int y = box.y; y < box.y + box.height; ++y)
    for(int x = box.x; x < box.x + box.width; ++x)
      m += (Vector2f(x,y) - c).norm();

  float s = sqrt(2.0f) / std::max(m, 1e-6f);

  T << s, 0, -s*c[0],
       0, s, -s*c[1],
       0, 0, 1;

  T_inv << 1.0f/s, 0, c[0],
           0, 1.0f/s, c[1],
           0, 0, 1;
}

Tracker::Impl::Impl(MotionType motion_model, AlgorithmParameters p)
    : _alg_params(p), _motion_type(motion_model)
      , _mc(MultiChannelExtractor::Create(p.multi_channel_function))
      , _T(Matrix33f::Identity()), _T_inv(Matrix33f::Identity())
      , _interp(cv::INTER_LINEAR), _border(cv::BORDER_CONSTANT)
{
  _mc->setSigma(p.sigma);
}

Tracker::Impl::~Impl() {}

void Tracker::Impl::setTemplate(const cv::Mat& /*image*/, const cv::Rect& box)
{
  if(_motion_type == MotionType::Homography)
    HartlyNormalization(box, _T, _T_inv);
  else
  {
    _T.setIdentity();
    _T_inv.setIdentity();
  }

  _bbox = box;
}

void Tracker::Impl::resizeChannelData(size_t n)
{
  switch(_motion_type)
  {
    case MotionType::Homography:
      _channel_data_homography.resize(n);
      break;

    case MotionType::Affine:
      _channel_data_affine.resize(n);
      break;

    case MotionType::Translation:
      _channel_data_translation.resize(n);
      break;
  }
}

void Tracker::Impl::setChannelData()
{
  THROW_ERROR_IF( _channels.empty(), "no channels" );

  const auto s = _T(0,0), c1 = _T_inv(0,2), c2 = _T_inv(1,2);
  resizeChannelData(_channels.size());

  auto Op = [=](size_t ind)
  {
    switch(_motion_type)
    {
      case MotionType::Homography:
        _channel_data_homography[ind].set(_channels[ind], _bbox, s, c1, c2);
        break;

      case MotionType::Affine:
        _channel_data_affine[ind].set(_channels[ind], _bbox, s, c1, c2);
        break;

      case MotionType::Translation:
        _channel_data_translation[ind].set(_channels[ind], _bbox, s, c1, c2);
    }
  }; // Op

  size_t i = 0;
#if defined(BITPLANES_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range<size_t>(0, _channels.size()),
                    [=](const tbb::blocked_range<size_t>& range) {
                      for(size_t j = range.begin(); j != range.end(); ++j)
                        Op(j);
                    });
  i = _channel_data.size();
#endif

  for( ; i < _channels.size(); ++i)
    Op(i);
}

void Tracker::Impl::computeResiduals(const cv::Mat& I, const Transform& T)
{
  switch(_motion_type)
  {
    case MotionType::Homography:
      {
        imwarp<Homography>(I, _Iw, T, _bbox, _interp_maps[0], _interp_maps[1], _interp);
      } break;

    case MotionType::Affine:
      {
        imwarp<Affine>(I, _Iw, T, _bbox, _interp_maps[0], _interp_maps[1], _interp);
      } break;

    case MotionType::Translation:
      {
        imwarp<Translation>(I, _Iw, T, _bbox, _interp_maps[0], _interp_maps[1], _interp);
      } break;
  }

  computeChannels(_Iw, cv::Rect(0,0,_Iw.cols,_Iw.rows), _channels_warped);

  const auto N = _channels_warped.size();
  _residuals.resize(N);

  auto Op = [=](size_t ind)
  {
    switch(_motion_type)
    {
      case MotionType::Homography:
        _channel_data_homography[ind].computeResiduals(_channels_warped[ind],
                                                       _residuals[ind]);
        break;

      case MotionType::Affine:
        _channel_data_affine[ind].computeResiduals(_channels_warped[ind],
                                                   _residuals[ind]);
        break;

      case MotionType::Translation:
        _channel_data_translation[ind].computeResiduals(_channels_warped[ind],
                                                        _residuals[ind]);
        break;
    }
  }; // Op


  size_t i = 0;
#if defined(BITPLANES_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range<size_t>(0, N)
                    [=](const tbb::blocked_range<size_t>& range)
                    {
                      for(size_t j = range.begin(); j != range.end(); ++j)
                        Op(j);
                    });
  i = N;
#endif

  for( ; i < N; ++i)
    Op(i);
}

template <class CDataVector, class GradientType> static inline
void ComputeGradient(const CDataVector& cdata, const ResidualsVector& residuals,
                     GradientType& g)
{
  assert( cdata.size() == residuals.size() && "incorrect data size" );

  g.setZero();
  for(size_t i = 0; i < cdata.size(); ++i)
    g.noalias() += cdata[i].jacobian().transpose() * residuals[i];
}

template <> void Tracker::Impl::
computeGradient<Homography>(const ResidualsVector& residuals,
                            typename MotionModel<Homography>::Gradient& g) const
{
  ComputeGradient(_channel_data_homography, residuals, g);
}

template <> void Tracker::Impl::
computeGradient<Affine>(const ResidualsVector& residuals,
                            typename MotionModel<Affine>::Gradient& g) const
{
  ComputeGradient(_channel_data_affine, residuals, g);
}

template <> void Tracker::Impl::
computeGradient<Translation>(const ResidualsVector& residuals,
                            typename MotionModel<Translation>::Gradient& g) const
{
  ComputeGradient(_channel_data_translation, residuals, g);
}

template <class CDataVector, class HessianType> static inline
void ComputeHessian(const CDataVector& cdata, HessianType& H)
{
  H.setZero();
  for(const auto& c : cdata)
    H.noalias() += c.jacobian().transpose() * c.jacobian();
}

template <> void Tracker::Impl::
computeHessian<Homography>(typename MotionModel<Homography>::Hessian& H) const
{
  ComputeHessian(_channel_data_homography, H);
}

template <> void Tracker::Impl::
computeHessian<Affine>(typename MotionModel<Affine>::Hessian& H) const
{
  ComputeHessian(_channel_data_affine, H);
}

template <> void Tracker::Impl::
computeHessian<Translation>(typename MotionModel<Translation>::Hessian& H) const
{
  ComputeHessian(_channel_data_translation, H);
}

} // bp



