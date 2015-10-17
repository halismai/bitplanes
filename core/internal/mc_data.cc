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

#include "bitplanes/core/internal/mc_data.h"
#include "bitplanes/core/internal/imwarp.h"
#include "bitplanes/core/internal/channel_extractor.h"
#include "bitplanes/core/homography.h"

#include <opencv2/core.hpp>

namespace bp {

static inline
void HartlyNormalization(const cv::Rect& box, Matrix33f& T, Matrix33f& T_inv)
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


template <class M>
MultiChannelData<M>::MultiChannelData()
  : _T(Matrix33f::Identity()), _T_inv(Matrix33f::Identity()) {}

template <class M>
void MultiChannelData<M>::setTemplate(const cv::Mat& image, const cv::Rect& box)
{
  HartlyNormalization(box, _T, _T_inv);
  _cdata.resize(1);
  _cdata[0].set(image, box, _T(0,0), _T_inv(0,2), _T_inv(1,2));
}

template <class M>
void MultiChannelData<M>::computeResiduals(const cv::Mat& image, const Matrix33f& T,
                                           Vector_<float>& residuals)
{
  assert( !_cdata.empty() && "no data" );

  cv::Mat Iw;
  imwarp<M>(image, Iw, T, _cdata[0].bbox());
  _cdata[0].computeResiduals(Iw, residuals);
}

template class MultiChannelData<Homography>;

}


