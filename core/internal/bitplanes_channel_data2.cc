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

#include "bitplanes/core/internal/bitplanes_channel_data2.h"
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

static const __m128i K0x01 = _mm_set1_epi8(0x01);
static const __m128i K0x02 = _mm_set1_epi8(0x02);
static const __m128i K0x04 = _mm_set1_epi8(0x04);
static const __m128i K0x08 = _mm_set1_epi8(0x08);
static const __m128i K0x10 = _mm_set1_epi8(0x10);
static const __m128i K0x20 = _mm_set1_epi8(0x20);
static const __m128i K0x40 = _mm_set1_epi8(0x40);
static const __m128i K0x80 = _mm_set1_epi8(0x80);

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

static inline uint8_t CensusSignature(const uint8_t* p, int s)
{
  return
  ((*(p - s - 1) >= *p) << 0) |
  ((*(p - s    ) >= *p) << 1) |
  ((*(p - s + 1) >= *p) << 2) |
  ((*(p     - 1) >= *p) << 3) |
  ((*(p     + 1) >= *p) << 4) |
  ((*(p + s - 1) >= *p) << 5) |
  ((*(p + s    ) >= *p) << 6) |
  ((*(p + s + 1) >= *p) << 7) ;
}

static inline v128 CensusSignatureSIMD(const uint8_t* p, int s)
{
  const v128 c(p);
 return
     ((v128(p - s - 1) >= c) & K0x01) |
     ((v128(p - s    ) >= c) & K0x02) |
     ((v128(p - s + 1) >= c) & K0x04) |
     ((v128(p     - 1) >= c) & K0x08) |
     ((v128(p     + 1) >= c) & K0x10) |
     ((v128(p + s - 1) >= c) & K0x20) |
     ((v128(p + s    ) >= c) & K0x40) |
     ((v128(p + s + 1) >= c) & K0x80) ;
}

static inline void CensusSignature(const uint8_t* p, int s, uint8_t* dst)
{
  _mm_storeu_si128((__m128i*) dst, CensusSignatureSIMD(p, s));
}

static inline uint8_t CensusBit(uint8_t c, int b)
{
  return ((c & (1 << b)) >> b);
}

template <class M> void
BitPlanesChannelData2<M>::set(const cv::Mat& src, const cv::Rect& roi,
                              float s, float c1, float c2)
{
  THROW_ERROR_IF(roi.x < 1 || roi.x > src.cols - 1 ||
                 roi.y < 1 || roi.y > src.rows - 1,
                 "template bounding box is outside image");

  cv::Mat tmp;
  src(roi).copyTo(tmp);
  _pixels.create(roi.size(), CV_8UC1);

  for(int y = 0; y < roi.height; ++y)
  {
    const auto* srow = src.ptr<uint8_t>(y + roi.y);
    auto* drow = _pixels.ptr<uint8_t>(y);

    int x = 0;
    for(x = 0; x <= roi.width - 16; x += 16)
    {
      CensusSignature(srow + x + roi.x, src.cols, drow + x);
    }

    for( ; x < roi.width; ++x)
      drow[x] = CensusSignature(srow + x + roi.x, src.cols);
  }
}

template <class M>
void BitPlanesChannelData2<M>::computeResiduals(const cv::Mat& Iw,
                                                Pixels& residuals) const
{
  THROW_ERROR_IF(Iw.rows != _pixels.rows || Iw.cols != _pixels.cols, "bad size");

  residuals.resize( 8 * ((Iw.rows-1) * (Iw.cols-1) ) );
  auto* r_ptr = residuals.data();
  ALIGNED(16) uint8_t obuf[8][16];
  int s = Iw.cols;
  for(int y = 1; y < Iw.rows-1; ++y)
  {
    const uint8_t* Iw_row = Iw.ptr<uint8_t>(y);
    const uint8_t* I0_row = _pixels.ptr<uint8_t>(y);

    int x;
    for(x=1; x <= Iw.cols-16-1; x += 16, r_ptr += (8*16))
    {
      const auto iw = CensusSignatureSIMD(Iw_row + x, Iw.cols);
      const auto i0 = v128(I0_row + x);
      (((iw & K0x01) >> 0) - ((i0 & K0x01) >> 0)).storeFloat(r_ptr + 0);
      (((iw & K0x02) >> 1) - ((i0 & K0x02) >> 1)).storeFloat(r_ptr + 16);
      (((iw & K0x04) >> 2) - ((i0 & K0x04) >> 2)).storeFloat(r_ptr + 32);
      (((iw & K0x08) >> 3) - ((i0 & K0x08) >> 3)).storeFloat(r_ptr + 48);
      (((iw & K0x10) >> 4) - ((i0 & K0x10) >> 4)).storeFloat(r_ptr + 64);
      (((iw & K0x20) >> 5) - ((i0 & K0x20) >> 5)).storeFloat(r_ptr + 80);
      (((iw & K0x40) >> 6) - ((i0 & K0x40) >> 6)).storeFloat(r_ptr + 96);
      (((iw & K0x80) >> 7) - ((i0 & K0x80) >> 7)).storeFloat(r_ptr + 112);

      /*
      const auto* p = Iw_row + x;

      const v128 c = v128(p);
      (((v128(p - s - 1) >= c) & K0x01) >> 0).store(obuf[0]);
      (((v128(p - s    ) >= c) & K0x02) >> 1).store(obuf[1]);
      (((v128(p - s + 1) >= c) & K0x04) >> 2).store(obuf[2]);
      (((v128(p     - 1) >= c) & K0x08) >> 3).store(obuf[3]);
      (((v128(p     + 1) >= c) & K0x10) >> 4).store(obuf[4]);
      (((v128(p + s - 1) >= c) & K0x20) >> 5).store(obuf[5]);
      (((v128(p + s    ) >= c) & K0x40) >> 6).store(obuf[6]);
      (((v128(p + s + 1) >= c) & K0x80) >> 7).store(obuf[7]);

      for(int b = 0, ii = 0; b < 8; ++b)
      {
        for(int j = 0; j < 16; ++j, ++ii)
        {
        }
      }
      */
    }

    for( ; x < Iw.cols - 1; ++x)
    {
      *r_ptr++ = 0.0f;
    }
  }

  for(int y = 1; y < Iw.rows-1; ++y)
  {
    const uint8_t* srow = Iw.ptr<uint8_t>(y);
    for(int x = 1; x < Iw.cols-1; ++x)
    {
      auto c = CensusSignature(srow + x, Iw.cols);
      int error = c - _pixels.at<uint8_t>(y, x);
      if(error != 0)
      {
        printf("(%d,%d), %d\n", y, x, error);
        THROW_ERROR("bad");
      }
    }
  }
}

template class BitPlanesChannelData2<Homography>;

} // bp

