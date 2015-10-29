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

#include "bitplanes/core/internal/bitplanes_channel_data_packed2.h"
#include "bitplanes/core/internal/ct.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/debug.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core.hpp>
#include <iostream>

namespace bp {

template <class M> static inline typename
EigenStdVector<typename M::WarpJacobian>::type
ComputeWarpJacobian(const cv::Rect& roi, float s, float c1, float c2)
{
  const int n_valid = (roi.height-2)*(roi.width-2);

  typename EigenStdVector<typename M::WarpJacobian>::type ret(n_valid);
  for(int y = 0, i = 0; y < roi.height-2; ++y)
    for(int x = 0; x < roi.width-2; ++x, ++i)
      ret[i] = M::ComputeWarpJacobian(x + 0.0f, y + 0.0f, s, c1, c2);

  return ret;
}

template <class M> void
BitPlanesChannelDataPacked2<M>::set(const cv::Mat& src, const cv::Rect& roi,
                             float s, float c1, float c2)
{
  THROW_ERROR_IF(roi.x < 1 || roi.x > src.cols - 1 ||
                 roi.y < 1 || roi.y > src.rows - 1,
                 "template bounding box is outside image");

  // template data is computed at the interior of the bounding box
  // this is an offset of 1 pixel
  int n_valid = (roi.width-2) * (roi.height-2);

  // stores the census signature (packed)
  _pixels.resize(n_valid);

  // 8 channels per pixel
  _jacobian.resize(n_valid*8, M::DOF);

  cv::Mat C;
  simd::census(src, roi, C);
  int stride = C.cols;

  /**
   * compute the channel x- and y-gradient at col=x for bit=b
   */
  auto G = [=](const uint8_t* p, int x, int b)
  {
    float ix1 = (p[x+1] & (1<<b)) >> b,
          ix2 = (p[x-1] & (1<<b)) >> b,
          iy1 = (p[x+stride] & (1<<b)) >> b,
          iy2 = (p[x-stride] & (1<<b)) >> b;

    return Eigen::Matrix<float,1,2>(0.5f*(ix1-ix2),0.5f*(iy1-iy2));
  }; //

  typename M::WarpJacobian Jw;
  auto* pixels_ptr = _pixels.data();
  int i = 0;
  for(int y = 1; y < C.rows - 1; ++y)
  {
    const auto* srow = C.ptr<const uint8_t>(y);
    for(int x = 1; x < C.cols - 1; ++x)
    {
      Jw = M::ComputeWarpJacobian(x+roi.x, y+roi.y, s, c1, c2);
      *pixels_ptr++ = srow[x];
      _jacobian.row(i++) = G(srow, x, 0) * Jw;
      _jacobian.row(i++) = G(srow, x, 1) * Jw;
      _jacobian.row(i++) = G(srow, x, 2) * Jw;
      _jacobian.row(i++) = G(srow, x, 3) * Jw;
      _jacobian.row(i++) = G(srow, x, 4) * Jw;
      _jacobian.row(i++) = G(srow, x, 5) * Jw;
      _jacobian.row(i++) = G(srow, x, 6) * Jw;
      _jacobian.row(i++) = G(srow, x, 7) * Jw;
    }
  }

  _hessian = _jacobian.transpose() * _jacobian;
}


template <class M>
void BitPlanesChannelDataPacked2<M>::computeResiduals(const cv::Mat& Iw,
                                                     Pixels& residuals) const
{
  simd::census_residual_packed(Iw, _pixels, residuals);
}

template class BitPlanesChannelDataPacked2<Homography>;

} // bp

