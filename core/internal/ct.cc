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

  for(int y = 1; y < Iw.rows - 1; ++y)
  {
    const uint8_t* srow = Iw.ptr<const uint8_t>(y);

    int x = 1;

#if HAVE_SSE2
#endif

#if HAVE_NEON
#endif

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
  }
}

}; // simd
}; // bp

