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

#include "bitplanes/core/internal/census.h"
#include "bitplanes/core/internal/v128.h"

#include "bitplanes/utils/error.h"

#include <opencv2/core/core.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace bp {

namespace simd {

static const __m128i K0x01 = _mm_set1_epi8(0x01);
static const __m128i K0x02 = _mm_set1_epi8(0x02);
static const __m128i K0x04 = _mm_set1_epi8(0x04);
static const __m128i K0x08 = _mm_set1_epi8(0x08);
static const __m128i K0x10 = _mm_set1_epi8(0x10);
static const __m128i K0x20 = _mm_set1_epi8(0x20);
static const __m128i K0x40 = _mm_set1_epi8(0x40);
static const __m128i K0x80 = _mm_set1_epi8(0x80);


#define C_OP >=
static inline void census_op(const uint8_t* src, int stride, uint8_t* dst)
{
  const v128 c(src);
  _mm_storeu_si128((__m128i*) dst,
                   ((v128(src - stride - 1) C_OP c) & K0x01) |
                   ((v128(src - stride    ) C_OP c) & K0x02) |
                   ((v128(src - stride + 1) C_OP c) & K0x04) |
                   ((v128(src          - 1) C_OP c) & K0x08) |
                   ((v128(src          + 1) C_OP c) & K0x10) |
                   ((v128(src + stride - 1) C_OP c) & K0x20) |
                   ((v128(src + stride    ) C_OP c) & K0x40) |
                   ((v128(src + stride + 1) C_OP c) & K0x80));

}


void CensusTransform(const cv::Mat& src, cv::Mat& dst)
{
  assert( src.type() == CV_8UC1 && src.isContinuous() );
  dst.create(src.size(), src.type());

  const int W = ((src.cols-2) & ~15) + 1;
  const uint8_t* src_ptr = src.ptr<const uint8_t>();
  uint8_t* dst_ptr = dst.ptr<uint8_t>();

  memset(dst_ptr, 0, src.cols);
  src_ptr += src.cols;
  dst_ptr += dst.cols;

  for(int r = 2; r < src.rows; ++r, src_ptr += src.cols, dst_ptr += dst.cols)
  {
    *(dst_ptr + 0) = 0;
    for(int c = 0; c < W; c += 16)
      census_op(src_ptr + c, src.cols, dst_ptr + c);

    if(W != src.cols-1)
      census_op(src_ptr + src.cols - 1 - 16, src.cols, dst_ptr + dst.cols - 1 - 16);

    *( dst_ptr + dst.cols - 1 ) = 0;
  }

  memset(dst_ptr, 0, dst.cols);
}


FORCE_INLINE void census_op_channel(const uint8_t* src, int offset, int /*stride*/,
                                    uint8_t* dst)
{
  _mm_storeu_si128((__m128i*) dst, v128( src + offset ) C_OP v128(src));
}


struct CensusTransformChannel8u
{
  int operator()(const uint8_t* src, uint8_t* dst, int offset, int width) const
  {
    int c = 0;
    for( ; c <= width - 32; c += 32)
    {
      census_op_channel(src + c, offset, width, dst + c);
      census_op_channel(src + c + 16, offset, width, dst + c + 16);
    }

    return c;
  }
}; // CensusTransformChannel8u

template <class Op>
class censusInvoker : public cv::ParallelLoopBody
{
 public:
  censusInvoker(const cv::Mat& src, int offset, cv::Mat& dst, int B)
      : cv::ParallelLoopBody(), _src(src), _offset(offset), _dst(dst), _B(B)
  {}

  virtual void operator()(const cv::Range& range) const
  {
    Op op;
    for(int y = range.start; y < range.end; ++y)
    {
      int x = op(_src.ptr<const uint8_t>(y), _dst.ptr<uint8_t>(y), _offset, _src.cols);

      for( ; x < _src.cols; ++x )
        ;
    }
  }

 private:
  const cv::Mat& _src;
  int _offset;
  cv::Mat& _dst;
  int _B;
}; // censusInvoker


void CensusTransformChannel(const cv::Mat& src, int offset, cv::Mat& dst, int B)
{
  assert( src.type() == CV_8UC1 && src.isContinuous() );
  dst.create(src.size(), src.type());

#if 0

  cv::Range range(1, src.rows-1);
  censusInvoker<CensusTransformChannel8u> invoker(src, offset, dst, B);
  cv::parallel_for_(range, invoker);

#else
  const uint8_t* src_ptr = src.ptr<const uint8_t>();
  uint8_t* dst_ptr = dst.ptr<uint8_t>();
  memset(dst_ptr, 0, src.cols);

  src_ptr += (B * src.cols);
  dst_ptr += (B * dst.cols);

  for(int r = B+1; r < src.rows; ++r, src_ptr += src.cols, dst_ptr += dst.cols)
  {
    *(dst_ptr + 0) = 0;

    int c = 0;
    for( ; c <= src.cols - 32; c += 32) {
      census_op_channel(src_ptr + c, offset, src.cols, dst_ptr + c);
      census_op_channel(src_ptr + c + 16, offset, src.cols, dst_ptr + c + 16);
    }

    for( ; c < src.cols; ++c)
      census_op_channel(src_ptr + c, offset, src.cols, dst_ptr + c);

    *( dst_ptr + dst.cols - 1 ) = 0;
  }

  memset(dst_ptr, 0, dst.cols);
#endif
}

void CensusTransform(const cv::Mat& src, const cv::Rect& roi, cv::Mat& dst)
{
  dst.create( cv::Size(roi.width+2, roi.height+2), CV_8UC1 );
  THROW_ERROR_IF( dst.empty(), "failed to allocate" );

  const int src_stride = src.cols, dst_stride = dst.cols;
  const uint8_t* src_ptr = src.ptr<uint8_t>();// + roi.y*src_stride;
  uint8_t* dst_ptr = dst.ptr<uint8_t>();

  memset(dst_ptr, 0, dst_stride);

  for(int y = 0; y < roi.height; ++y)
  {
    uint8_t* d_row = dst_ptr + (y + 1)*dst_stride;
    const uint8_t* s_row = src_ptr + (y + roi.y)*src_stride;

    int x = 0;
    d_row[x] = 0;
    const int w = roi.width & ~15;
    for( ; x < w; x += 16)
    {
      census_op(s_row + x + roi.x + 16, src_stride, d_row + x + 1 + 16);
      //census_op(s_row + x + roi.x + 1*0xf, src_stride, d_row + x + 1 + 1*0xf);
      //census_op(s_row + x + roi.x + 2*0xf, src_stride, d_row + x + 1 + 2*0xf);
      //census_op(s_row + x + roi.x + 3*0xf, src_stride, d_row + x + 1 + 3*0xf);
    }

    for( ; x < roi.width; ++x)
    {
      const int xs = x + roi.x;
      const uint8_t c = s_row[xs];
      d_row[x+1] =
          ((*(s_row + xs - src_stride - 1) >= c) << 0) |
          ((*(s_row + xs - src_stride    ) >= c) << 1) |
          ((*(s_row + xs - src_stride + 1) >= c) << 2) |
          ((*(s_row + xs              - 1) >= c) << 3) |
          ((*(s_row + xs              + 1) >= c) << 4) |
          ((*(s_row + xs + src_stride - 1) >= c) << 5) |
          ((*(s_row + xs + src_stride    ) >= c) << 6) |
          ((*(s_row + xs + src_stride + 1) >= c) << 7) ;
    }

    d_row[dst_stride-1] = 0;
  }

  memset(dst.ptr<uint8_t>(dst.rows-1), 0, dst_stride);
}


}; // simd

#undef C_OP

static inline int
get_border_from_offset(const int* offsets, int N)
{
  int max_val = std::numeric_limits<int>::min();
  for(int i = 0; i < N; ++i) {
    int o = std::abs(offsets[i]);
    if(o > max_val)
      max_val = o;
  }

  return 1 + max_val;
}

template <typename T> static inline
int Heaviside(T x)
{
  return static_cast<int>( x <= 0.0 );
}

template<typename T>
struct identity { using type = T; };

template<typename T>
using try_make_signed = typename std::conditional<
std::is_integral<T>::value, std::make_signed<T>, identity<T> >::type;


template <typename T> static inline
void CensusTransformGeneric(const T* src_ptr, const cv::Size& src_size,
                            cv::Mat& dst, const int* x_off, int x_len,
                            const int* y_off, int y_len)
{
  dst.create(src_size, CV_32FC1);
  auto* dst_ptr = dst.ptr<float>();
  auto stride = dst.cols;

  int y_border = get_border_from_offset(y_off, y_len),
      x_border = get_border_from_offset(x_off, x_len);

  std::fill_n(dst_ptr, y_border*stride, 0.0f);

  typedef typename try_make_signed<T>::type SignedType;

  for(int y = y_border; y < dst.rows - y_border - 1; ++y)
  {
    for(int x = 0; x < x_border; ++x)
      dst_ptr[y*stride + x] = 0.0f;


    for(int x = x_border; x < dst.cols - x_border - 1; ++x)
    {
      const auto p0 = static_cast<SignedType>( src_ptr[y*stride + x] );

      // sample around the pixel using the provided sampling pattern (offsets)
      // we'll store the result as a float weighted by the sampling location
      // float is used to accomodate large sampling locations with radii larger
      // than 2 or 3 pixels
      float ct = 0.0f;
      for(int i = 0, ii=0; i < y_len; ++i)
      {
        int ys = y + y_off[i];
        for(int j = 0; j < x_len; ++j, ++ii)
        {
          int xs = x + x_off[j];
          const auto p = static_cast<SignedType>( src_ptr[ys*stride + xs] );

          ct += static_cast<float>( Heaviside(p0 - p) * (1 << ii) );
        }
      }

      dst_ptr[y*stride + x] = ct;
    }
  }
}



void CensusTransform(const cv::Mat& src, cv::Mat& dst,
                     const int* x_off, int x_len,
                     const int* y_off, int y_len)
{
  assert( src.isContinuous() );

  switch(src.type())
  {
    case cv::DataType<uint8_t>::type:
      CensusTransformGeneric(src.ptr<const uint8_t>(), src.size(),
                             dst, x_off, x_len, y_off, y_len);
      break;

    case cv::DataType<uint16_t>::type:
      CensusTransformGeneric(src.ptr<const uint16_t>(), src.size(),
                             dst, x_off, x_len, y_off, y_len);
      break;

    case cv::DataType<float>::type:
      CensusTransformGeneric(src.ptr<const float>(), src.size(),
                             dst, x_off, x_len, y_off, y_len);
      break;

    case cv::DataType<double>::type:
      CensusTransformGeneric(src.ptr<const double>(), src.size(),
                             dst, x_off, x_len, y_off, y_len);
      break;

    default:
      THROW_ERROR("unsupported image type");
  }
}


void CensusTransform(const cv::Mat& src, const cv::Rect& roi, cv::Mat& dst)
{
  THROW_ERROR_IF(src.type() != CV_8UC1, "image must be uint8_t" );

  dst.create( cv::Size(roi.width+2, roi.height+2), CV_8UC1 );
  uint8_t* dst_ptr = dst.ptr<uint8_t>();
  const uint8_t* src_ptr = src.ptr<uint8_t>();
  const int src_stride = src.cols, dst_stride = dst.cols;

  memset(dst_ptr, 0, dst_stride);

  for(int y = 0; y < roi.height; ++y)
  {
    uint8_t* d_row = dst_ptr + (y + 1)*dst_stride;
    const uint8_t* s_row = src_ptr + (y + roi.y)*src_stride;

    int x = 0;
    d_row[x] = 0;
    for(  ; x < roi.width; ++x)
    {
      const int xs = x + roi.x;
      const uint8_t c = s_row[xs];
      d_row[x+1] =
          ((*(s_row + xs - src_stride - 1) >= c) << 0) |
          ((*(s_row + xs - src_stride    ) >= c) << 1) |
          ((*(s_row + xs - src_stride + 1) >= c) << 2) |
          ((*(s_row + xs              - 1) >= c) << 3) |
          ((*(s_row + xs              + 1) >= c) << 4) |
          ((*(s_row + xs + src_stride - 1) >= c) << 5) |
          ((*(s_row + xs + src_stride    ) >= c) << 6) |
          ((*(s_row + xs + src_stride + 1) >= c) << 7) ;
    }

    d_row[dst_stride-1] = 0;
  }

  memset(dst.ptr<uint8_t>(dst.rows-1), 0, dst_stride);
}

} // bp

