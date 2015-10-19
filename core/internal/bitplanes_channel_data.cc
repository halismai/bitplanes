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
#include "bitplanes/core/internal/census.h"
#include "bitplanes/core/homography.h"
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
      ret[i] = M::ComputeWarpJacobian(x + 1.0f, y + 1.0f, s, c1, c2);

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
  float* pixel_ptr = _pixels.data();

  _jacobian.resize(n_valid * 8, M::DOF);

  cv::Mat C;
  simd::CensusTransform2(src, roi, C);

  _roi_stride = roi.width - 2;

  typedef Eigen::Matrix<float,1,2> ImageGradient;

#pragma omp parallel for
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
            static_cast<float>( (srow[x+1] & (1 << b)) >> b ) -
            static_cast<float>( (srow[x-1] & (1 << b)) >> b ) ;

        float Iy =
            static_cast<float>(srow[x + C.cols] & (1 << b) >> b) -
            static_cast<float>(srow[x - C.cols] & (1 << b) >> b);

        int jj = 8*((y-1)*_roi_stride + x - 1) + b;
        _jacobian.row(jj) = 0.5f * ImageGradient(Ix, Iy) * Jw;
      }
    }
  }

  _hessian = _jacobian.transpose() * _jacobian; // bottleneck
}


template <class M>
void BitPlanesChannelData<M>::computeResiduals(const cv::Mat& Iw, Pixels& residuals) const
{
  cv::Mat C;
  simd::CensusTransform2(Iw, cv::Rect(1,1,Iw.cols-1,Iw.rows-1), C);

  residuals.resize(_pixels.size());
  const float* pixels_ptr = _pixels.data();
  float* residuals_ptr = residuals.data();

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

}

template class BitPlanesChannelData<Homography>;

} // bp

