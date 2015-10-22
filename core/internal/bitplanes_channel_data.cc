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

#include "bitplanes/core/internal/bitplanes_channel_data.h"
#include "bitplanes/core/internal/intrin.h"
#include "bitplanes/core/internal/v128.h"
#include "bitplanes/core/internal/census.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/debug.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <array>
#include <iostream>
#include <fstream>

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
BitPlanesChannelData<M>::set(const cv::Mat& src, const cv::Rect& roi,
                             float s, float c1, float c2)
{
  THROW_ERROR_IF(roi.x < 1 || roi.x > src.cols - 1 ||
                 roi.y < 1 || roi.y > src.rows - 1,
                 "template bounding box is outside image");

  const auto Jw_tmp = ComputeWarpJacobian<M>(roi, s, c1, c2);
  const int n_valid = Jw_tmp.size();

  _pixels.resize(n_valid * 8);
  auto* pixel_ptr = _pixels.data();

  _jacobian.resize(n_valid * 8, M::DOF);

  cv::Mat C;
  simd::CensusTransform2(src, roi, C);

  _roi_stride = roi.width - 2;

  typedef Eigen::Matrix<float,1,2> ImageGradient;

//#pragma omp parallel for
  for(int y = 1; y < C.rows - 1; ++y)
  {
    const uint8_t* srow = C.ptr<const uint8_t>(y);
    for(int x = 1; x < C.cols - 1; ++x)
    {
      int ii = (y-1)*_roi_stride + x - 1;
      const auto& Jw = Jw_tmp[ii];
      for(int b = 0; b < 8; ++b)
      {
        pixel_ptr[8*ii + b] = (srow[x] & (1<<b)) >> b;

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

static const __m128i K0x01 = _mm_set1_epi8(0x01);
static const __m128i K0x02 = _mm_set1_epi8(0x02);
static const __m128i K0x04 = _mm_set1_epi8(0x04);
static const __m128i K0x08 = _mm_set1_epi8(0x08);
static const __m128i K0x10 = _mm_set1_epi8(0x10);
static const __m128i K0x20 = _mm_set1_epi8(0x20);
static const __m128i K0x40 = _mm_set1_epi8(0x40);
static const __m128i K0x80 = _mm_set1_epi8(0x80);


template <class M>
void BitPlanesChannelData<M>::computeResiduals(const cv::Mat& Iw, Pixels& residuals) const
{
  residuals.resize(_pixels.size());
  const auto* pixels_ptr = _pixels.data();
  float* residuals_ptr = residuals.data();

#if 0
  cv::Mat C;
  simd::CensusTransform2(Iw, cv::Rect(1,1,Iw.cols-1,Iw.rows-1), C);

  for(int y = 0; y < C.rows - 1; ++y)
  {
    const uint8_t* srow = C.ptr<const uint8_t>(y);
    for(int x = 0; x < C.cols - 1; ++x)
    {
      int ii = y*_roi_stride + x;
      for(int b = 0; b < 8; ++b)
      {
        int jj = 8*ii + b;
        residuals_ptr[jj] = ((srow[x] & (1<<b)) >> b) - pixels_ptr[jj];
      }
    }
  }
#else
  int src_stride = Iw.cols;
  cv::Rect roi(1, 1, Iw.cols-1, Iw.rows-1);
  //v128 buf[8];
  ALIGNED(16) uint8_t obuf[8][16];
  for(int y = 0; y < roi.height - 1; ++y)
  {
    const uint8_t* srow = Iw.ptr<const uint8_t>(y + roi.y);

    int x = 0;
    for( ; x <= roi.width - 16 - 1; x += 16)
    {
      const uint8_t* p = srow + x + roi.x;
      const v128 c(p);

#if 0
      _mm_store_si128((__m128i*) obuf[0],
                      ((v128(p - src_stride - 1) >= c) & K0x01) >> 0);

      _mm_store_si128((__m128i*) obuf[1],
                      ((v128(p - src_stride    ) >= c) & K0x02) >> 1);

      _mm_store_si128((__m128i*) obuf[2],
                      ((v128(p - src_stride + 1) >= c) & K0x04) >> 2);

      _mm_store_si128((__m128i*) obuf[3],
                      ((v128(p              - 1) >= c) & K0x08) >> 3);

      _mm_store_si128((__m128i*) obuf[4],
                      ((v128(p              + 1) >= c) & K0x10) >> 4);

      _mm_store_si128((__m128i*) obuf[5],
                      ((v128(p + src_stride - 1) >= c) & K0x20) >> 5);

      _mm_store_si128((__m128i*) obuf[6],
                      ((v128(p + src_stride    ) >= c) & K0x40) >> 6);

      _mm_store_si128((__m128i*) obuf[7],
                      ((v128(p + src_stride + 1) >= c) & K0x80) >> 7);

#endif

      for(int j = 0; j < 16; ++j)
      {
        *residuals_ptr++ = obuf[0][j] - *pixels_ptr++;
        *residuals_ptr++ = obuf[1][j] - *pixels_ptr++;
        *residuals_ptr++ = obuf[2][j] - *pixels_ptr++;
        *residuals_ptr++ = obuf[3][j] - *pixels_ptr++;
        *residuals_ptr++ = obuf[4][j] - *pixels_ptr++;
        *residuals_ptr++ = obuf[5][j] - *pixels_ptr++;
        *residuals_ptr++ = obuf[6][j] - *pixels_ptr++;
        *residuals_ptr++ = obuf[7][j] - *pixels_ptr++;
      }
    }

    for( ; x < roi.width - 1; ++x)
    {
      const uint8_t* p = srow + x + roi.x;
      const uint8_t c = *p;

      *residuals_ptr++ = ((*(p - src_stride - 1) >= c)) - *pixels_ptr++;
      *residuals_ptr++ = ((*(p - src_stride    ) >= c)) - *pixels_ptr++;
      *residuals_ptr++ = ((*(p - src_stride + 1) >= c)) - *pixels_ptr++;
      *residuals_ptr++ = ((*(p              - 1) >= c)) - *pixels_ptr++;
      *residuals_ptr++ = ((*(p              + 1) >= c)) - *pixels_ptr++;
      *residuals_ptr++ = ((*(p + src_stride - 1) >= c)) - *pixels_ptr++;
      *residuals_ptr++ = ((*(p + src_stride    ) >= c)) - *pixels_ptr++;
      *residuals_ptr++ = ((*(p + src_stride + 1) >= c)) - *pixels_ptr++;
    }
  }
#endif

}

template class BitPlanesChannelData<Homography>;

} // bp

