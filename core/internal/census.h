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

#ifndef BITPLANES_CORE_INTERNAL_CENSUS_H
#define BITPLANES_CORE_INTERNAL_CENSUS_H

#include "bitplanes/core/internal/cvfwd.h"

namespace bp {

/**
 * Not optimized
 */
void CensusTransform(const cv::Mat&, cv::Mat&);

void CensusTransform(const cv::Mat&, const cv::Rect&, cv::Mat&);

/**
 * most generic form of census, patch radius and sampling locations are
 * determined by x_off and y_off
 */
void CensusTransform(const cv::Mat& src, cv::Mat& dst,
                     const int* x_off, int x_off_len,
                     const int* y_off, int y_off_len);


void CensusTransformChannel(const cv::Mat&, const cv::Rect&, int off, cv::Mat&);


namespace simd {
/**
 */
void CensusTransform(const cv::Mat&, cv::Mat&);

void CensusTransform(const cv::Mat&, const cv::Rect&, cv::Mat&);

/**
 */
void CensusTransformChannel(const cv::Mat&, int addr, cv::Mat&, int off);

}; // simd

}; // bp

#endif // BITPLANES_CORE_INTERNAL_CENSUS_H
