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

#include "bitplanes/core/internal/bitplanes_channel_data_fast.h"
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
BitPlanesChannelDataFast<M>::set(const cv::Mat& src, const cv::Rect& roi,
                             float s, float c1, float c2)
{
  THROW_ERROR_IF(roi.x < 1 || roi.x > src.cols - 1 ||
                 roi.y < 1 || roi.y > src.rows - 1,
                 "template bounding box is outside image");

  const auto Jw_tmp = ComputeWarpJacobian<M>(roi, s, c1, c2);
  const int n_valid = Jw_tmp.size();

  _pixels.resize(n_valid);
  auto* pixel_ptr = _pixels.data();

  _jacobian.resize(n_valid * 8, M::DOF);

  cv::Mat C;
  simd::census(src, roi, C);

  _roi_stride = roi.width - 2;

  typedef Eigen::Matrix<float,1,2> ImageGradient;
  for(int y = 1; y < C.rows - 1; ++y)
  {
    const uint8_t* srow = C.ptr<const uint8_t>(y);
    for(int x = 1; x < C.cols - 1; ++x)
    {
      int ii = (y-1)*_roi_stride + x - 1;
      const auto& Jw = Jw_tmp[ii];
      *pixel_ptr++ = srow[x];
      for(int b = 0; b < 8; ++b)
      {
        //pixel_ptr[8*ii + b] = (srow[x] & (1<<b)) >> b;

        float Ix =
            static_cast<float>( (srow[x+1] & (1 << b)) /*>> b*/ ) -
            static_cast<float>( (srow[x-1] & (1 << b)) /*>> b*/ ) ;

        float Iy =
            static_cast<float>(srow[x + C.cols] & (1 << b) /*>> b*/) -
            static_cast<float>(srow[x - C.cols] & (1 << b) /*>> b*/);

        //float w = sqrt( fabs(Ix) + fabs(Iy) );
        float w = 1.0f / (float) ( 1 << b );

        int jj = 8*((y-1)*_roi_stride + x - 1) + b;
        _jacobian.row(jj) = w * 0.5f * ImageGradient(Ix, Iy) * Jw;
      }
    }
  }

  _hessian = _jacobian.transpose() * _jacobian; // bottleneck
}


template <class M>
void BitPlanesChannelDataFast<M>::computeResiduals(const cv::Mat& Iw, Pixels& residuals) const
{
  simd::census_residual_packed(Iw, _pixels, residuals);
}

template <class M> void
BitPlanesChannelDataFast<M>::
linearize(const cv::Mat& Iw, Gradient& g) const
{
  g.setZero();

  const auto* c0_ptr = _pixels.data();
  const int src_stride = Iw.cols;

  Eigen::Matrix<float,8,1> R;
  for(int y = 1, i=0; y < Iw.rows - 1; ++y)
  {
    const uint8_t* srow = Iw.ptr<const uint8_t>(y);

    int x = 1;

    for( ; x < Iw.cols - 1; ++x, ++i)
    {
      const uint8_t* p = srow + x;
      const uint8_t c = *c0_ptr++;

      R <<
           (*(p - src_stride - 1) >= *p) - (c & (1<<0) >> 0),
           (*(p - src_stride    ) >= *p) - (c & (1<<1) >> 1),
           (*(p - src_stride + 1) >= *p) - (c & (1<<2) >> 2),
           (*(p              - 1) >= *p) - (c & (1<<3) >> 3),
           (*(p              + 1) >= *p) - (c & (1<<4) >> 4),
           (*(p + src_stride - 1) >= *p) - (c & (1<<5) >> 5),
           (*(p + src_stride    ) >= *p) - (c & (1<<6) >> 6),
           (*(p + src_stride + 1) >= *p) - (c & (1<<7) >> 7);

      g.noalias() += _jacobian.template block<8,8>(8*i, 0).transpose() * R;
#if 0
      g.noalias() +=
          _jacobian.template block<8,8>(8*i, 0).transpose() *
          (Eigen::Matrix<float,8,1>() <<
           (*(p - src_stride - 1) >= *p) - (c & (1<<0) >> 0),
           (*(p - src_stride    ) >= *p) - (c & (1<<1) >> 1),
           (*(p - src_stride + 1) >= *p) - (c & (1<<2) >> 2),
           (*(p              - 1) >= *p) - (c & (1<<3) >> 3),
           (*(p              + 1) >= *p) - (c & (1<<4) >> 4),
           (*(p + src_stride - 1) >= *p) - (c & (1<<5) >> 5),
           (*(p + src_stride    ) >= *p) - (c & (1<<6) >> 6),
           (*(p + src_stride + 1) >= *p) - (c & (1<<7) >> 7)
           ).finished();

#endif

#if 0
      g.noalias() += _jacobian.row(8*i + 0).transpose() *
          (float) ((*(p - src_stride - 1) >= *p) - (c & (1<<0) >> 0));

      g.noalias() += _jacobian.row(8*i + 1).transpose() *
          (float) ((*(p - src_stride    ) >= *p) - (c & (1<<1) >> 1));

      g.noalias() += _jacobian.row(8*i + 2).transpose() *
          (float) ((*(p - src_stride + 1) >= *p) - (c & (1<<2) >> 2));

      g.noalias() += _jacobian.row(8*i + 3).transpose() *
          (float) ((*(p - 1) >= *p) - (c & (1<<3) >> 3));

      g.noalias() += _jacobian.row(8*i + 4).transpose() *
          (float) ((*(p + 1) >= *p) - (c & (1<<4) >> 4));

      g.noalias() += _jacobian.row(8*i + 5).transpose() *
          (float) ((*(p + src_stride - 1) >= *p) - (c & (1<<5) >> 5));

      g.noalias() += _jacobian.row(8*i + 6).transpose() *
          (float) ((*(p + src_stride    ) >= *p) - (c & (1<<6) >> 6));

      g.noalias() += _jacobian.row(8*i + 7).transpose() *
          (float) ((*(p + src_stride + 1) >= *p) - (c & (1<<7) >> 7));
#endif

    }
  }
}

template class BitPlanesChannelDataFast<Homography>;


}; // bp
