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
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <array>
#include <iostream>

namespace bp {

#if 0
static const __m128i K0x01 = _mm_set1_epi8(0x01);
static const __m128i K0x02 = _mm_set1_epi8(0x02);
static const __m128i K0x04 = _mm_set1_epi8(0x04);
static const __m128i K0x08 = _mm_set1_epi8(0x08);
static const __m128i K0x10 = _mm_set1_epi8(0x10);
static const __m128i K0x20 = _mm_set1_epi8(0x20);
static const __m128i K0x40 = _mm_set1_epi8(0x40);
static const __m128i K0x80 = _mm_set1_epi8(0x80);
#endif

#define C_OP >=
static FORCE_INLINE
void CensusChannelOp(const uint8_t* src, int offset, uint8_t* dst)
{
  // we aligned dst properly
  _mm_storeu_si128((__m128i*) dst, v128(src + offset) C_OP v128(src) );
}

static FORCE_INLINE
void CensusChannelOp(const uint8_t* src, int offset, uint8_t* dst, float* dst_f)
{
  const v128 c = v128(src + offset) C_OP v128(src);
  _mm_storeu_si128((__m128i*) dst, c);

  c.storeFloat(dst_f);
}

typedef Eigen::Matrix<float, 1, 2> ImageGradient;
typedef typename EigenStdVector<ImageGradient>::type ImageGradientVector;

static FORCE_INLINE
void ExtractChannel(const cv::Mat& src, const cv::Rect& roi, const __m128i& mask,
                    int offset, ImageGradientVector& g, float* bits)
{
  cv::Mat dst;
  dst.create(roi.size(), CV_8UC1);

  for(int y = 0; y < roi.height; ++y)
  {
    const uint8_t* srow = src.ptr<const uint8_t>(y + roi.y);
    uint8_t* drow = dst.ptr<uint8_t>(y);
    int x = 0;
    for( ; x <= roi.width - 16; x +=16, bits += 16)
    {
      const v128 c = (v128(srow + roi.x + x + offset) C_OP v128(srow + x + roi.x)) & mask;
      _mm_store_si128((__m128i*) (drow + x), c);
      c.storeFloat(bits);
    }

    for( ; x < roi.width; ++x)
    {
      drow[x] = *(srow + x + roi.x + offset) C_OP *(srow + x + roi.x);
      *bits++ = static_cast<float>( drow[x] );
    }
  }

  g.resize(roi.area());

  int i = 0;
  for(int y = 0; y < roi.height; ++y, ++i)
    g[i].setZero();

  auto Ix = [=](int x, int y)
  {
    return (float) dst.at<uint8_t>(y, x+1) - (float) dst.at<uint8_t>(y, x-1);
  }; //

  auto Iy = [=](int x, int y)
  {
    return (float) dst.at<uint8_t>(y+1, x) - (float) dst.at<uint8_t>(y-1, x);
  };

  const int d_stride = dst.cols;

  for(int y = 1; y < roi.height - 1; ++y)
  {
    g[i].setZero();

    const uint8_t* drow = dst.ptr<uint8_t>(y);
    for(int x = 1; x < roi.width - 1; ++x, ++i)
    {
      g[i][0] = 0.5f * ((float) drow[x+1] - (float) drow[x-1]);
      g[i][1] = 0.5f * ((float) drow[x+d_stride] - (float) drow[x-d_stride]);

      std::cout << g[i] << std::endl;
    }

    g[i].setZero();
  }

}

static inline
std::array<int,8> GetCensusOffsets(int stride)
{
  return

  {
    -stride - 1, -stride, -stride + 1,
            - 1,                  + 1,
     stride - 1,  stride, stride + 1
  };
}

template <class M> void
BitPlanesChannelData<M>::set(const cv::Mat& src, const cv::Rect& roi,
                             float s, float c1, float c2)
{
  THROW_ERROR_IF(roi.x < 1 || roi.x > src.cols - 1 ||
                 roi.y < 1 || roi.y > src.rows - 1,
                 "template bounding box is outside image");

  const int n_valid = roi.area();
  typename EigenStdVector<WarpJacobian>::type Jw_tmp(n_valid);
  for(int y = 0, i = 0; y < roi.height; ++y)
    for(int x = 0; x < roi.width; ++x, ++i)
      Jw_tmp[i] = M::ComputeWarpJacobian(x, y, s, c1, c2);


  _pixels.resize(n_valid * 8);
  float* pixel_ptr = _pixels.data();

  _jacobian.resize(n_valid * 8, M::DOF);

  const __m128i masks[8] = {
    _mm_set1_epi8(0x01),
    _mm_set1_epi8(0x02),
    _mm_set1_epi8(0x04),
    _mm_set1_epi8(0x08),
    _mm_set1_epi8(0x10),
    _mm_set1_epi8(0x20),
    _mm_set1_epi8(0x40),
    _mm_set1_epi8(0x80)
  };

  const auto offsets = GetCensusOffsets(src.cols);
  for(size_t i = 0; i < offsets.size(); ++i)
  {
    ImageGradientVector g;
    ExtractChannel(src, roi, masks[i], offsets[i], g, pixel_ptr + i*n_valid);

    for(size_t j = 0; j < g.size(); ++j)
    {
      _jacobian.row(i*n_valid + j) = g[j] * Jw_tmp[j];
    }
  }

}

#undef C_OP


template class BitPlanesChannelData<Homography>;
} // bp
