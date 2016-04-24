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

#ifndef BITPLANES_INTERNAL_CHANNEL_DATA_SUBSAMPLED_H
#define BITPLANES_INTERNAL_CHANNEL_DATA_SUBSAMPLED_H

#include "bitplanes/core/internal/bitplanes_channel_data_base.h"
#include "bitplanes/core/motion_model.h"

#include <opencv2/imgproc.hpp>

namespace bp {

template <class> class BitPlanesChannelDataSubSampled;

template <class M>
class BitPlanesChannelDataSubSampled :
    public BitPlanesChannelData<BitPlanesChannelDataSubSampled<M>>
{
 public:
  typedef BitPlanesChannelDataSubSampled<M> Self;
  typedef BitPlanesChannelData<Self> Base;
  typedef typename Base::MotionModelType MotionModelType;
  typedef typename Base::Pixels Pixels;
  typedef typename Base::Residuals Residuals;
  typedef typename Base::WarpJacobian WarpJacobian;
  typedef typename Base::JacobianMatrix JacobianMatrix;
  typedef typename Base::Hessian Hessian;
  typedef typename Base::Transform Transform;
  typedef typename Base::Gradient Gradient;

 public:
  /**
   * \param s subsampling/decimation factor. A value of 1 means no decimation, a
   * value of 2 means decimate by half, and so on
   */
  inline BitPlanesChannelDataSubSampled(size_t s = 1)
      : Base(), _sub_sampling(s) {}

  void set(const cv::Mat&, const cv::Rect& roi, float s = 1,
           float c1 = 0, float c2 = 0);

  void computeResiduals(const cv::Mat& Iw, Residuals& residuals) const;

  float doLinearize(const cv::Mat& Iw, Gradient&) const;

  void warpImage(const cv::Mat& src, const Transform& T, const cv::Rect& roi,
                 cv::Mat& dst, int interp = cv::INTER_LINEAR, float border = 0.0f);

  inline const Pixels& pixels() const { return _pixels; }
  inline const Hessian& hessian() const { return _hessian; }
  inline const JacobianMatrix& jacobian() const { return _jacobian; }

  void getCoordinateNormalization(const cv::Rect&, Transform&, Transform&) const;

 protected:
  JacobianMatrix _jacobian;
  Pixels _pixels;
  Hessian _hessian;
  int _sub_sampling;
  int _roi_stride;
}; // BitPlanesChannelDataSubSampled

}; // bp

#endif // BITPLANES_INTERNAL_CHANNEL_DATA_SUBSAMPLED_H
