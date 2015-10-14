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

#if defined(BITPLANES_WITH_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <iostream>
#include <limits>

#include <Eigen/LU>

namespace bp {

Tracker::Impl::Impl(MotionType motion_model, AlgorithmParameters p)
    : _alg_params(p), _motion_type(motion_model)
      , _mc(MultiChannelExtractor::Create(p.multi_channel_function))
      , _T(Matrix33f::Identity()), _T_inv(Matrix33f::Identity())
      , _interp(cv::INTER_LINEAR), _border(cv::BORDER_CONSTANT) {}

Tracker::Impl::~Impl() {}


void Tracker::Impl::setTemplate(const cv::Mat& image, const cv::Rect& box)
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

  c /= (float) _points.size(); // center of mass
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

void Tracker::Impl::setChannelData()
{
  THROW_ERROR_IF( _channels.empty(), "no channels" );

  const auto s = _T(0,0), c1 = _T_inv(0,2), c2 = _T_inv(1,2);
  resizeChannelData(_channels.size());

  auto Op = [=](size_t ind)
  {
    _channel_data[ind].set(_channels[ind], _points, s, c1, c2);
  }; // Op

  size_t i = 0;
#if defined(BITPLANES_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range<size_t>(0, _channel_data.size()),
                    [=](const tbb::blocked_range<size_t>& range)
                    {
                      for(size_t j = range.begin(); j != range.end(); ++j)
                        Op(j);
                    });
  i = _channel_data.size();
#endif

  for( ; i < _channel_data.size(); ++i)
    Op(i);
}

void Tracker::Impl::computeResiduals(const cv::Mat& I, const Transform& T)
{
  imwarp(I, _Iw, T, _points, _bbox, _interp_maps[0], _interp_maps[1],
         _motion_type == MotionType::Homography, _interp, _border, _border_val[0]);

  computeChannels(_Iw, cv::Rect(0,0,_Iw.cols,_Iw.rows), _channels_warped);

  const auto N = _channels_warped.size();
  assert( N == _channel_data.size() && "Wrong number of channels" );

  _residuals.resize(N);

  auto Op = [=](size_t ind)
  {
    _channel_data[ind].computeResiduals(_channels_warped[ind], _residuals[ind]);
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

} // bp



