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

#ifndef BITPLANES_CORE_INTERNAL_IMWARP_H
#define BITPLANES_CORE_INTERNAL_IMWARP_H

#include "bitplanes/core/internal/cvfwd.h"
#include "bitplanes/core/types.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace bp {

/**
 * \param src the source image
 * \param dst destination
 * \param T   transform
 * \param point vector of point coordinates
 * \param xmap, ymap storage to be used for interpolation. Must be allocated to
 *                   the right size
 * \param is_projective to indicate if the transform is projective/homography
 */
void imwarp(const cv::Mat& src, cv::Mat& dst, const Matrix33f& T,
            const PointVector& points, const cv::Rect& box,
            cv::Mat& xmap, cv::Mat& ymap, bool is_projective = true,
            int interp = cv::INTER_LINEAR, int border = cv::BORDER_CONSTANT,
            float border_val = 0.0f);

template <class M>
void imwarp(const cv::Mat& src, cv::Mat& dst, const Matrix33f& T,
            const cv::Rect& bbox, cv::Mat& xmap, cv::Mat& ymap,
            int interp = cv::INTER_LINEAR);

template <class M>
void imwarp(const cv::Mat& src, cv::Mat& dst, const Matrix33f& T,
            const cv::Rect& bbox);

namespace simd {

/**
 * Warps the image. NOTE: all pointers must be pre-allocated to hold at least
 * size N the function does NOT allocate anything
 *
 * NOTE: requires SSE enabled
 *
 *
 * \param I pointer to the image data to be warped
 * \param w width of the image (and its stride)
 * \param h height of the image
 * \param X image points to be warped. Each point has 4 elements
 * \param I_ref intensities corresponding to the points X at the ref frame
 * \param residuals differences between I_ref and warped
 * \param valid  1 if the pixel warps to a valid location within the image
 * \param N  number of elements in X, I_ref, residuals and valid
 * \param I_warped if not null we'll store the value of warped intensities
 *
 * \return number of valid points
 *
 *
 * NOTE: H is a pointer to 4x4 rigid-body transform matrix in col major order
 *
 */
int imwarp(const uint8_t* I, int w, int h, const float* H, const float* X,
           const float* I_ref, float* residuals, uint8_t* valid, int N,
           float* I_warped = nullptr);


/**
 * same as above, but H is 3x3 matrix and X is 3xN set of points
 */
int imwarp3(const uint8_t* I, int w, int h, const float* H, const float* X,
           const float* I_ref, float* residuals, uint8_t* valid, int N,
           float* I_warped = nullptr);


}; // simd

}; // bp

#endif // BITPLANES_CORE_INTERNAL_IMWARP_H
