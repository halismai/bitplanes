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

#include "bitplanes/core/internal/channel_data_dense.h"
#include "bitplanes/core/config.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/translation.h"
#include "bitplanes/core/affine.h"
#include "bitplanes/utils/error.h"

#include <cassert>
#include <iostream>

namespace bp {

namespace {
template <typename T, class M, class PixelsType, class JacobianType> static inline
void SetData(const cv::Mat& image, const cv::Rect& bbox, float s, float c1, float c2,
             PixelsType& pixels, JacobianType& jacobian)
{
  typedef MotionModel<M> Motion;
  constexpr int DOF = Motion::DOF;

  const int stride = image.cols;

  THROW_ERROR_IF( bbox.x < 1 || bbox.y < 1, "bad box" );
  //const auto n = cv::Rect(bbox.x+1,bbox.y+1,bbox.width-1,bbox.height-1).area();
  const auto n = bbox.area();
  if(pixels.size() != n)
    pixels.resize(n);

  if(jacobian.rows() != n)
    jacobian.resize(n, DOF);

  const T* I = image.ptr<const T>();

  typename Motion::Jacobian J;
  const int y_s = bbox.y, x_s = bbox.x;
  for(int y = y_s + 0, i=0; y < bbox.height + y_s - 0; ++y)
  {
    for(int x = x_s + 0; x < bbox.width + x_s - 0; ++x, ++i)
    {
      const int ii = y*stride + x;
      pixels[i] = I[ii];
      float Ix = 0.5f * ( (float) I[ii+1] - (float) I[ii-1] );
      float Iy = 0.5f * ( (float) I[ii+stride] - (float) I[ii-stride] );
      Motion::ComputeJacobian(J, x-x_s, y-y_s, Ix, Iy, s, c1, c2);
      jacobian.row(i) = J;
    }
  }
}

template <typename T, class Pixels> static inline
void ComputeResiduals(const cv::Mat& I, const cv::Rect& bbox, const Pixels& p0,
                      Pixels& residuals)
{
  const auto n = p0.size();
  if(n != residuals.size())
    residuals.resize(n);


  const T* ptr = I.ptr<const T>();
  const int y_s = bbox.y, x_s = bbox.x;
  for(int y = y_s + 1, i = 0; y < bbox.height + y_s - 1; ++y)
  {
    const auto* p = ptr + y*I.cols;
    for(int x = x_s + 1; x < bbox.width + x_s - 1; ++x, ++i)
      residuals[i] = static_cast<float>( *(p + x) ) - p0[i];
  }
}

static inline void CheckImageType(const cv::Mat& image)
{
  THROW_ERROR_IF(image.type() != cv::DataType<float>::type &&
                 image.type() != cv::DataType<uint8_t>::type,
                 "Incorrect image type, must be float or uint8_t");

  THROW_ERROR_IF(image.channels() > 1, "image must be grayscale");
}

} // namespace

template <class M>
void ChannelDataDense<M>::set(const cv::Mat& image, const cv::Rect& bbox,
                              float s, float c1, float c2)
{
  CheckImageType(image);

  switch( image.type() )
  {
    case cv::DataType<float>::type:
      SetData<float, M>(image, bbox, s, c1, c2, _pixels, _jacobian);
      break;

    case cv::DataType<uint8_t>::type:
      SetData<uint8_t, M>(image, bbox, s, c1, c2, _pixels, _jacobian);
      break;
  }

  _bbox = bbox;
}

template <class M>
void ChannelDataDense<M>::computeResiduals(const cv::Mat& Iw, Pixels& residuals) const
{
  CheckImageType(Iw);

  switch(Iw.type())
  {
    case cv::DataType<float>::type:
      ComputeResiduals<float>(Iw, _bbox, _pixels, residuals);
      break;
    case cv::DataType<uint8_t>::type:
      ComputeResiduals<uint8_t>(Iw, _bbox, _pixels, residuals);
      break;
  }

}

template class ChannelDataDense<Homography>;
template class ChannelDataDense<Affine>;
template class ChannelDataDense<Translation>;

} // bp

