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

#include "bitplanes/core/internal/ct.h"
#include "bitplanes/core/config.h"
#include "bitplanes/utils/utils.h"
#include "bitplanes/utils/error.h"
#include <opencv2/core.hpp>


#define HAVE_SSE2 BITPLANES_HAVE_SSE2
#define HAVE_NEON BITPLANES_HAVE_ARM

#if HAVE_SSE2
#include <xmmintrin.h>
#endif

#if HAVE_NEON
#include <arm_neon.h>
#endif

#include <cstddef>
#include <iostream>

#if BITPLANES_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

namespace bp {
namespace simd {

void census(const cv::Mat& src, const cv::Rect& roi, cv::Mat& dst)
{
  THROW_ERROR_IF( src.type() != CV_8UC1, "src image must be CV_8UC1" );
  THROW_ERROR_IF( !src.isContinuous() , "src image must be continuous");
  THROW_ERROR_IF( roi.x < 1 || roi.x > src.cols - 2 ||
                  roi.y < 1 || roi.y > src.rows - 2,
                  "ROI must inside the image boundaries with 1 px offset at least");

  dst.create(roi.size(), src.type());
  THROW_ERROR_IF( dst.empty(), "Failed to allocate memory" );

  int src_stride = src.cols;

  for(int y = 0; y < roi.height; ++y)
  {
    const uint8_t* srow = src.ptr<const uint8_t>(y + roi.y);
    uint8_t* drow = dst.ptr<uint8_t>(y);

    int x = 0;

#if HAVE_SSE2
#endif

#if HAVE_NEON
#endif

    for( ; x < roi.width; ++x)
    {
      const uint8_t* p = srow + x + roi.x;
      drow[x] =
          ((*(p - src_stride - 1) >= *p) << 0) |
          ((*(p - src_stride    ) >= *p) << 1) |
          ((*(p - src_stride + 1) >= *p) << 2) |
          ((*(p              - 1) >= *p) << 3) |
          ((*(p              + 1) >= *p) << 4) |
          ((*(p + src_stride - 1) >= *p) << 5) |
          ((*(p + src_stride    ) >= *p) << 6) |
          ((*(p + src_stride + 1) >= *p) << 7) ;
    }
  }
}

void census_residual(const cv::Mat& Iw, const Vector_<uint8_t>& c0,
                     Vector_<float>& residuals)
{
  THROW_ERROR_IF( Iw.type() != CV_8UC1, "image must CV_8UC1" );
  THROW_ERROR_IF( !Iw.isContinuous(), "image must continuous");

  const auto* c0_ptr = c0.data();
  const int src_stride = Iw.cols;

  residuals.resize(c0.size());
  auto* r_ptr = residuals.data();

  for(int x = 0; x < Iw.cols; ++x)
    *r_ptr++ = 0.0f;

  for(int y = 1; y < Iw.rows - 1; ++y)
  {
    const uint8_t* srow = Iw.ptr<const uint8_t>(y);

    int x = 1;

#if HAVE_SSE2
#endif

#if HAVE_NEON
#endif

    *r_ptr++ = 0.0;
    for( ; x < Iw.cols - 1; ++x)
    {
      const uint8_t* p = srow + x;
      *r_ptr++ = (*(p - src_stride - 1) >= *p) - *c0_ptr++;
      *r_ptr++ = (*(p - src_stride    ) >= *p) - *c0_ptr++;
      *r_ptr++ = (*(p - src_stride + 1) >= *p) - *c0_ptr++;
      *r_ptr++ = (*(p              - 1) >= *p) - *c0_ptr++;
      *r_ptr++ = (*(p              + 1) >= *p) - *c0_ptr++;
      *r_ptr++ = (*(p + src_stride - 1) >= *p) - *c0_ptr++;
      *r_ptr++ = (*(p + src_stride    ) >= *p) - *c0_ptr++;
      *r_ptr++ = (*(p + src_stride + 1) >= *p) - *c0_ptr++;
    }
    *r_ptr++ = 0;
  }

  printf("DIFF: %p %p\n", r_ptr, residuals.data() + residuals.size());
  exit(0);
}

#if BITPLANES_WITH_TBB

class CensusResidualsPacked
{
 public:
  /**
   * \param image the   input/warped image
   * \param c0          pointer to the reference census transform
   * \param residuals   output residuals
   * \param roi_stride  ROI stride
   */
  CensusResidualsPacked(const cv::Mat& image, const uint8_t* c0,
                        float* residuals, int roi_stride)
      : _image(image), _c0(c0), _residuals(residuals), _roi_stride(roi_stride) {}

  void operator()(const tbb::blocked_range<int>& range) const
  {
    int src_stride = _image.cols;
    int s = range.grainsize();
    for(int y = range.begin(); y != range.end(); ++y)
    {
      const uint8_t* srow = _image.ptr<const uint8_t>(y);
      for(int x = 1; x < _image.cols - 1; x += s)
      {
        int i = ((y-1)/s) * _roi_stride + ((x-1)/s);
        const uint8_t* p = srow + x;
        const uint8_t c = _c0[i];
        float* r_ptr = _residuals + 8*i;
        *r_ptr++ = (*(p - src_stride - 1) >= *p) - ((c & (1<<0)) >> 0);
        *r_ptr++ = (*(p - src_stride    ) >= *p) - ((c & (1<<1)) >> 1);
        *r_ptr++ = (*(p - src_stride + 1) >= *p) - ((c & (1<<2)) >> 2);
        *r_ptr++ = (*(p              - 1) >= *p) - ((c & (1<<3)) >> 3);
        *r_ptr++ = (*(p              + 1) >= *p) - ((c & (1<<4)) >> 4);
        *r_ptr++ = (*(p + src_stride - 1) >= *p) - ((c & (1<<5)) >> 5);
        *r_ptr++ = (*(p + src_stride    ) >= *p) - ((c & (1<<6)) >> 6);
        *r_ptr++ = (*(p + src_stride + 1) >= *p) - ((c & (1<<7)) >> 7);
      }
    }
  }

 protected:
  const cv::Mat& _image;
  const uint8_t* _c0;
  float* _residuals;
  int _sub_sampling;
  int _roi_stride;
}; // CensusResidualsPacked

#endif

void census_residual_packed(const cv::Mat& Iw, const Vector_<uint8_t>& c0,
                            Vector_<float>& residuals, int s, int roi_stride)
{
  THROW_ERROR_IF( Iw.type() != CV_8UC1, "image must CV_8UC1" );
  THROW_ERROR_IF( !Iw.isContinuous(), "image must continuous");

  residuals.resize(8 * c0.size());

  const auto* c0_ptr = c0.data();
  auto* r_ptr = residuals.data();

#if BITPLANES_WITH_TBB
  THROW_ERROR_IF(roi_stride <= 0, "roi_stride is invalid");
  tbb::parallel_for(tbb::blocked_range<int>(1, Iw.rows-1, s),
                    CensusResidualsPacked(Iw, c0_ptr, r_ptr, roi_stride));
#else
  UNUSED(roi_stride);

  //int8_t buf[8];
  const int src_stride = Iw.cols;
  for(int y = 1; y < Iw.rows - 1; y += s)
  {
    const uint8_t* srow = Iw.ptr<const uint8_t>(y);

#if HAVE_SSE2
#endif

#if HAVE_NEON
#endif

#pragma omp simd
    for(int x = 1; x < Iw.cols - 1; x += s)
    {
      const uint8_t* p = srow + x;
      const uint8_t c = *c0_ptr++;
      *r_ptr++ = (*(p - src_stride - 1) >= *p) - ((c & (1<<0)) >> 0);
      *r_ptr++ = (*(p - src_stride    ) >= *p) - ((c & (1<<1)) >> 1);
      *r_ptr++ = (*(p - src_stride + 1) >= *p) - ((c & (1<<2)) >> 2);
      *r_ptr++ = (*(p              - 1) >= *p) - ((c & (1<<3)) >> 3);
      *r_ptr++ = (*(p              + 1) >= *p) - ((c & (1<<4)) >> 4);
      *r_ptr++ = (*(p + src_stride - 1) >= *p) - ((c & (1<<5)) >> 5);
      *r_ptr++ = (*(p + src_stride    ) >= *p) - ((c & (1<<6)) >> 6);
      *r_ptr++ = (*(p + src_stride + 1) >= *p) - ((c & (1<<7)) >> 7);
    }
  }
#endif // BITPLANES_WITH_TBB
}

}; // simd
}; // bp

