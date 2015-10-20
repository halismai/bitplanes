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

#include "bitplanes/core/internal/intrin.h"
#include "bitplanes/core/internal/gmag.h"

#include <opencv2/core/core.hpp>

namespace bp {
namespace simd {

#if BITPLANES_HAVE_SSE2
static FORCE_INLINE __m128i absdiff(__m128i a, __m128i b)
{
  return _mm_sub_epi8(_mm_max_epu8(a, b), _mm_min_epu8(a, b));
}

static FORCE_INLINE __m128i gmap_op(const uint8_t* src, int stride)
{
  __m128i s10 = _mm_loadu_si128((const __m128i*) (src - 1)),
          s12 = _mm_loadu_si128((const __m128i*) (src + 1)),
          s01 = _mm_loadu_si128((const __m128i*) (src - stride)),
          s21 = _mm_loadu_si128((const __m128i*) (src + stride)),
          dx = absdiff(s10, s12),
          dy = absdiff(s01, s21);
  return _mm_adds_epu8(dx, dy);
}

void gradientAbsMag(const cv::Mat& src, cv::Mat& dst)
{
  assert( src.type() == CV_8UC1 );
  dst.create(src.size(), CV_8UC1);

  const uint8_t* src_ptr = src.data;
  uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.data);

  memset(dst_ptr, 0, dst.cols);
  src_ptr += src.cols;
  dst_ptr += dst.cols;

  const int W = src.cols & ~15;

  for(int r = 2; r < src.rows; ++r, src_ptr += src.cols, dst_ptr += dst.cols)
  {
    dst_ptr[0] = 0;
    for(int c = 0; c < W; c += 16)
      _mm_storeu_si128((__m128i*) (dst_ptr + c), gmap_op(src_ptr + c, src.cols));

    if(src.cols-1 != W)
      _mm_storeu_si128((__m128i*) (dst_ptr + src.cols - 16),
                       gmap_op(src_ptr + src.cols - 16, src.cols));

    dst_ptr[dst.cols-1] = 0;
  }

  memset(dst_ptr, 0, dst.cols);
}

#else
#warning "SSE is not enabled. Falling back to normal code"
static FORCE_INLINE int absdiff(int a, int b)
{
  int d = a - b;
  int m = d >> 8;
  return (d & ~m) | (-d & m);
}

static FORCE_INLINE int min_u8(int a, int b)
{
  int d = a - b;
  int m = ~(d >> 8);
  return a - ( d & m );
}

void gradientAbsMag(const cv::Mat& src, cv::Mat& dst)
{
  assert( src.type() == CV_8UC1 );
  dst.create(src.size(), CV_8UC1);

  const uint8_t* src_ptr = src.data;
  uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.data);

  memset(dst_ptr, 0, dst.cols);
  src_ptr += src.cols;
  dst_ptr += dst.cols;

  for(int r = 2; r < src.rows; ++r, src_ptr += src.cols, dst_ptr += dst.cols)
  {
    dst_ptr[0] = 0;
    for(int c = 1; c < src.cols-1; ++c)
    {
      int dx = absdiff(src_ptr[c - src.cols], src_ptr[c + src.cols]),
          dy = absdiff(src_ptr[c - 1], src_ptr[c + 1]);
      dst_ptr[c] = min_u8(dx + dy, 0xff);
    }
    dst_ptr[dst.cols-1] = 0;
  }

  memset(dst_ptr, 0, dst.cols);
}
#endif

} // simd
} // bp
