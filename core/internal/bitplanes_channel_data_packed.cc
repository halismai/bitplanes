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

#include "bitplanes/core/internal/bitplanes_channel_data_packed.h"
#include "bitplanes/core/internal/ct.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/debug.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core.hpp>

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
BitPlanesChannelDataPacked<M>::set(const cv::Mat& src, const cv::Rect& roi,
                             float s, float c1, float c2)
{
  THROW_ERROR_IF(roi.x < 1 || roi.x > src.cols - 1 ||
                 roi.y < 1 || roi.y > src.rows - 1,
                 "template bounding box is outside image");

  const auto Jw_tmp = ComputeWarpJacobian<M>(roi, s, c1, c2);
  const int n_valid = Jw_tmp.size();


  printf("n_valid: %zu %d\n", n_valid,
         (roi.width-2) * (roi.height-2));
  _pixels.resize(n_valid);
  auto* pixel_ptr = _pixels.data();

  _jacobian.resize(n_valid * 8, M::DOF);

  cv::Mat C;
  simd::census(src, roi, C);

  _roi_stride = roi.width - 2;

  auto GetBit = [](uint8_t v, int bit)
  {
    return static_cast<float>( (v & (1<<bit)) >> bit );
  };

  for(int y = 1; y < C.rows - 1; ++y)
  {
    const uint8_t* srow = C.ptr<const uint8_t>(y);
    for(int x = 1; x < C.cols - 1; ++x)
    {
      //int ii = (y-1)*_roi_stride + x - 1;
      //const auto& Jw = Jw_tmp[ii];
      *pixel_ptr++ = srow[x];
      for(int b = 0; b < 8; ++b)
      {
#if 0
        float Ix =
            static_cast<float>( (srow[x+1] & (1 << b)) /*>> b*/ ) -
            static_cast<float>( (srow[x-1] & (1 << b)) /*>> b*/ ) ;
        float Iy =
            static_cast<float>(srow[x + C.cols] & (1 << b) /*>> b*/) -
            static_cast<float>(srow[x - C.cols] & (1 << b) /*>> b*/);
        //float w = sqrt( fabs(Ix) + fabs(Iy) );
        float w = 1.0f / (float) ( 1 << b );
#endif

        float w = 0.5f;
        float Ix = w * GetBit(srow[x+1], b) - GetBit(srow[x-1], b);
        float Iy = w * GetBit(srow[x+C.cols], b) - GetBit(srow[x-C.cols], b);
        int jj  = 8*((y-1)*_roi_stride + x - 1) + b;
        //_jacobian.row(jj) = ImageGradient(Ix, Iy) * Jw;
        Eigen::Matrix<float,8,1> J;

        float u = x - 1;
        float v = y - 1;
        J <<
            Ix/s,
            Iy/s,
            Iy*(c1 - u) - Ix*(c2 - v),
            Ix*u - Iy*c2 - Ix*c1 + Iy*v,
            Iy*(c2 - v) - Ix*(c1 - u),
            -Ix*(c2 - v),
            -s*(c1 - u)*(Ix*c1 + Iy*c2 - Ix*u - Iy*v),
            -s*(c2 - v)*(Ix*c1 + Iy*c2 - Ix*v - Iy*v);
        _jacobian.row(jj) = J;
      }
    }
  }

  _hessian = _jacobian.transpose() * _jacobian; // bottleneck
}


template <class M>
void BitPlanesChannelDataPacked<M>::computeResiduals(const cv::Mat& Iw,
                                                     Pixels& residuals) const
{
  simd::census_residual_packed(Iw, _pixels, residuals);
}

template class BitPlanesChannelDataPacked<Homography>;

} // bp

