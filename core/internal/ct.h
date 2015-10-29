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

#ifndef BITPLANES_CORE_INTERNAL_CT_H
#define BITPLANES_CORE_INTERNAL_CT_H

#include "bitplanes/core/types.h"
#include "bitplanes/core/internal/cvfwd.h"

namespace bp {
namespace simd {

/**
 * \param src the source image
 * \param roi region of interset in the source image
 * \param dst destination image
 *
 * The roi must be inside the image with at least 1 pixel off the border
 */
void census(const cv::Mat& src, const cv::Rect& roi, cv::Mat& dst);

/**
 * \parma Iw warped image
 * \param c0 census bits from the reference image
 * \param residuals output residuals (bits)
 *
 * an offset of 1 pixels is taken off from Iw when computing the residuals. The
 * vector c0 should be computed accordingly
 */
void census_residual(const cv::Mat& Iw, const Vector_<uint8_t>& c0,
                     Vector_<float>& residuals);

void census_residual_packed(const cv::Mat& Iw, const Vector_<uint8_t>& c0,
                            Vector_<float>& residuals, int s = 1);



}; // simd
}; // bp

#endif // BITPLANES_CORE_INTERNAL_CT_H
