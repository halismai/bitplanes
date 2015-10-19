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

#include "bitplanes/core/internal/bitplanes_channel_data_packed.h"
#include "bitplanes/core/internal/census.h"
#include "bitplanes/core/internal/imwarp.h"

#include "bitplanes/core/homography.h"
#include "bitplanes/core/affine.h"
#include "bitplanes/core/translation.h"

#include <opencv2/imgproc.hpp>

namespace bp {

template <class M> void
BitPlanesChannelDataPacked<M>::set(const cv::Mat& I, const cv::Rect& bbox,
                                   float s, float c1, float c2)
{
  _bbox = bbox;

  const int npts = bbox.area() * 8;
  _jacobian.resize( npts, M::DOF );
  _pixels.resize( npts );

  I.copyTo(_I0);
  cv::GaussianBlur(_I0(bbox), _I0(bbox), cv::Size(3,3), 1.2);
  simd::CensusTransform(_I0, bbox, _C0);

  typename EigenStdVector<typename M::WarpJacobian>::type Jw_tmp(bbox.area());
  for(int y = 0, i = 0; y < bbox.height; ++y)
    for(int x = 0; x < bbox.width; ++x, ++i)
      Jw_tmp[i] = M::ComputeWarpJacobian(x, y, s, c1, c2);


  const uint8_t* c_ptr = _C0.ptr<const uint8_t>();
  const int stride = _C0.cols;

  typedef Eigen::Matrix<float,1,2> ImageGradient;

  auto get_bit = [](uint8_t v, int b) { return (v & (1 << b)) >> b; }; // get_bit

  auto G = [=](int ii, int b)
  {
    float ix = get_bit( c_ptr[ii + 1], b ) - get_bit( c_ptr[ii - 1], b ),
          iy = get_bit( c_ptr[ii + stride], b ) - get_bit( c_ptr[ii - stride], b );
    return ImageGradient(0.5f*ix, 0.5f*iy);
  }; // G

  _hessian.setZero();
  for(int y = 1, i=0, j=0; y < _C0.rows - 1; ++y)
    for(int x = 1; x < _C0.cols - 1; ++x, ++j, i += 8)
    {
      const int ii = y*stride + x;
      const auto& Jw = Jw_tmp[j];

      _jacobian.row(i+0) = G(ii, 0) * Jw;
      _jacobian.row(i+1) = G(ii, 1) * Jw;
      _jacobian.row(i+2) = G(ii, 2) * Jw;
      _jacobian.row(i+3) = G(ii, 3) * Jw;
      _jacobian.row(i+4) = G(ii, 4) * Jw;
      _jacobian.row(i+5) = G(ii, 5) * Jw;
      _jacobian.row(i+6) = G(ii, 6) * Jw;
      _jacobian.row(i+7) = G(ii, 7) * Jw;

      /*_hessian.noalias() += _jacobian.template block<8,8>(i,0).transpose() *
          _jacobian.template block<8,8>(i,0);*/

      _pixels[i+0] = get_bit(ii, 0);
      _pixels[i+1] = get_bit(ii, 1);
      _pixels[i+2] = get_bit(ii, 2);
      _pixels[i+3] = get_bit(ii, 3);
      _pixels[i+4] = get_bit(ii, 4);
      _pixels[i+5] = get_bit(ii, 5);
      _pixels[i+6] = get_bit(ii, 6);
      _pixels[i+7] = get_bit(ii, 7);
    }

  _hessian = _jacobian.transpose() * _jacobian;
}

template <class M> void
BitPlanesChannelDataPacked<M>::computeResiduals(const cv::Mat& Iw, Pixels& residuals)
{
  if(residuals.size() != _pixels.size())
    residuals.resize(_pixels.size());

  auto get_bit = [](uint8_t v, int b) { return (v & (1 << b)) >> b; }; // get_bit
  simd::CensusTransform(Iw, _bbox, _C0);

  const uint8_t* c_ptr = _C0.ptr<const uint8_t>();

  for(int y = 1, i = 0; y < _C0.rows - 1; ++y)
    for(int x = 1; x < _C0.cols - 1; ++x, i+=8)
    {
      const int ii = y*_C0.cols + x;
      const uint8_t c = c_ptr[ii];
      residuals[i+0] = (float) get_bit(c,0) - _pixels[i+0];
      residuals[i+1] = (float) get_bit(c,1) - _pixels[i+1];
      residuals[i+2] = (float) get_bit(c,2) - _pixels[i+2];
      residuals[i+3] = (float) get_bit(c,3) - _pixels[i+3];
      residuals[i+4] = (float) get_bit(c,4) - _pixels[i+4];
      residuals[i+5] = (float) get_bit(c,5) - _pixels[i+5];
      residuals[i+6] = (float) get_bit(c,6) - _pixels[i+6];
      residuals[i+7] = (float) get_bit(c,7) - _pixels[i+7];
    }
}

template class BitPlanesChannelDataPacked<Homography>;
template class BitPlanesChannelDataPacked<Affine>;
template class BitPlanesChannelDataPacked<Translation>;

}; // bp
