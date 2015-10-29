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

#ifndef BITPLANES_INTERNAL_CHANNEL_DATA_BASE_H
#define BITPLANES_INTERNAL_CHANNEL_DATA_BASE_H

#include "bitplanes/core/debug.h"
#include "bitplanes/core/types.h"
#include "bitplanes/core/internal/cvfwd.h"

namespace bp {

template <class> struct channel_data_traits;

template <class DerivedType>
class BitPlanesChannelData
{
 public:
  typedef DerivedType Derived;
  typedef typename channel_data_traits<Derived>::MotionModelType MotionModelType;
  typedef typename channel_data_traits<Derived>::Pixels   Pixels;
  typedef typename channel_data_traits<Derived>::Residuals Residuals;
  typedef typename MotionModelType::WarpJacobian    WarpJacobian;
  typedef typename MotionModelType::JacobianMatrix  JacobianMatrix;
  typedef typename MotionModelType::Transform       Transform;
  typedef typename MotionModelType::Hessian         Hessian;

 public:
  /**
   * sets the template data
   *
   * \param image the input image
   * \param roi   region of interest specifying the template location
   * \param args  normalization to be applied when computing jacobians
   */
  template <class ... Args> inline
  void set(const cv::Mat& image, const cv::Rect& roi, Args&...args)
  {
    printf("calling set\n");
    return derived()->set(image, roi, args...);
  }

  /**
   * computes the residuals
   *
   * \param warped_image the warped image
   * \param residuals  output residuals
   */
  inline void computeResiduals(const cv::Mat& warped_image, Residuals& residuals) const
  {
    return derived()->computeResiduals(warped_image, residuals);
  }

  inline const Pixels& pixels() const { return derived()->pixels(); }
  inline const Hessian& hessian() const { return derived()->hessian(); }
  inline const JacobianMatrix& jacobian() const { return derived()->jacobian(); }

  inline void getCoordinateNormalization(const cv::Rect& roi,
                                         Transform& T, Transform& T_inv) const
  {
    return derived()->getCoordinateNormalization(roi, T, T_inv);
  }

 private:
  inline const Derived* derived() const { return static_cast<const Derived*>(this); }
  inline       Derived* derived()       { return static_cast<Derived*>(this); }
}; // BitPlanesChannelData


template <class> class BitPlanesChannelDataSubSampled;
template<> template <class M>
struct channel_data_traits< BitPlanesChannelDataSubSampled<M> >
{
  typedef M MotionModelType;
  typedef Vector_<uint8_t> Pixels;
  typedef Vector_<float> Residuals;
}; // channel_data_traits

}; // bp

#endif // BITPLANES_INTERNAL_CHANNEL_DATA_BASE_H
