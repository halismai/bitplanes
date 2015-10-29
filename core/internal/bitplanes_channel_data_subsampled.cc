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

#include "bitplanes/core/internal/bitplanes_channel_data_subsampled.h"
#include "bitplanes/core/internal/ct.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core.hpp>

#include <type_traits>

namespace bp {


static inline int GetNumValid(const cv::Rect& roi, int s)
{
  int ret = 0;
  for(int y = 1; y < roi.height-1; y += s)
    for(int x = 1; x < roi.width-1; x += s)
      ++ret;

  return ret;
}

template <class M>
void BitPlanesChannelDataSubSampled<M>::
set(const cv::Mat& src, const cv::Rect& roi, float s, float c1, float c2)
{
  THROW_ERROR_IF(roi.x < 1 || roi.x > src.cols - 1 ||
                 roi.y < 1 || roi.y > src.rows - 1,
                 "template bounding box is outside image");

  THROW_ERROR_IF( s <= 0, "scale cannot be negative or 0" );

  printf("s: %d\n", _sub_sampling);
  auto n_valid = GetNumValid(roi, _sub_sampling);
  printf("n_valid %d\n", n_valid);
  _pixels.resize(n_valid);
  _jacobian.resize(8*n_valid, M::DOF);

  cv::Mat C;
  simd::census(src, roi, C);
  int stride = C.cols;

  /**
   * compute the channel x- and y-gradient at col:=x for bit:=b
   */
  auto G = [=](const uint8_t* p, int x, int b)
  {
    float ix1 = (p[x+1] & (1<<b)) >> b, ix2 = (p[x-1] & (1<<b)) >> b,
          iy1 = (p[x+stride] & (1<<b)) >> b, iy2 = (p[x-stride] & (1<<b)) >> b;
    return Eigen::Matrix<float,1,2>(0.5f*(ix1-ix2), 0.5f*(iy1-iy2));
  }; //

  typename M::WarpJacobian Jw;
  auto* pixels_ptr = _pixels.data();
  int i = 0;
  for(int y = 1; y < C.rows - 1; y += _sub_sampling)
  {
    const auto* srow = C.ptr<const uint8_t>(y);
    for(int x = 1; x < C.cols - 1; x += _sub_sampling)
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
  _roi_stride = roi.width;
}

template <class M>
void BitPlanesChannelDataSubSampled<M>::
computeResiduals(const cv::Mat& Iw, Residuals& residuals) const
{
  simd::census_residual_packed(Iw, _pixels, residuals, _sub_sampling, _roi_stride);
}

template <class M>
void BitPlanesChannelDataSubSampled<M>::
getCoordinateNormalization(const cv::Rect& /*roi*/, Transform& T, Transform& T_inv) const
{
  T.setIdentity();
  T_inv.setIdentity();
}

template <>
void BitPlanesChannelDataSubSampled<Homography>::
getCoordinateNormalization(const cv::Rect& roi, Transform& T, Transform& T_inv) const
{
  Vector2f c(0,0);

  int n_valid = 0;
  for(int y = 1; y < roi.height-1; y += _sub_sampling)
    for(int x = 1; x < roi.width-1; x += _sub_sampling, ++n_valid)
      c += Vector2f(x + roi.x, y + roi.y);
  c /= n_valid;

  float m = 0.0f;
  for(int y = 1; y < roi.height-1; y += _sub_sampling)
    for(int x = 1; x < roi.width-1; x += _sub_sampling)
      m += (Vector2f(x+roi.x, y + roi.y) - c).norm();
  m /= n_valid;

  float s = sqrt(2.0f) / std::max(m, 1e-6f);

  T << s, 0, -s*c[0],
       0, s, -s*c[1],
       0, 0, 1;

  T_inv << 1.0f/s, 0, c[0],
           0, 1.0f/s, c[1],
           0, 0, 1;
}

template class BitPlanesChannelDataSubSampled<Homography>;

}
