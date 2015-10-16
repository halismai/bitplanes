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

#ifndef BITPLANES_CORE_INTERNAL_TRACKER_IMPL_H
#define BITPLANES_CORE_INTERNAL_TRACKER_IMPL_H

#include "bitplanes/core/tracker.h"
#include "bitplanes/core/internal/mc_extractor.h"
#include "bitplanes/core/internal/channel_data.h"
#include "bitplanes/core/internal/SmallVector.h"

#include <opencv2/core/core.hpp>

namespace bp {


struct Tracker::Impl
{
  typedef Tracker::Impl Self;
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

  Impl(MotionType, AlgorithmParameters);
  virtual ~Impl();


  static UniquePointer<Self> Create(MotionType, AlgorithmParameters);

  /**
   * Track the template, pure virtual
   */
  virtual Result track(const cv::Mat&, const Transform&) = 0;

  /**
   * sets the template data
   */
  virtual void setTemplate(const cv::Mat&, const cv::Rect&);

  /**
   * \return the sum of squared residuals
   */
  inline float computeSumSquaredErrors(const ResidualsVector& residuals) const
  {
    float ret = 0.0f;
    for(const auto& r : residuals)
      ret += r.squaredNorm();
    return ret;
  }

  /**
   */
  inline void resizeChannelData(size_t n)
  {
    _channel_data.resize(n, _motion_type);
  }

  /**
   */
  void computeChannels(const cv::Mat& src, const cv::Rect& bbox, ChannelsVector& C)
  {
    _mc->operator()(src, bbox, C);
  }

  /**
   */
  void setChannelData();

  /**
   */
  inline void allocateInterpMaps(const cv::Size& s)
  {
    _interp_maps[0].create(s, CV_32F);
    _interp_maps[1].create(s, CV_32F);
  }

  /**
   * compute the residuals given the input image and Transform
   */
  void computeResiduals(const cv::Mat&, const Transform&);


  AlgorithmParameters _alg_params; //< algorithm parameters
  MotionType _motion_type;         //< type of motion to estimate
  UniquePointer<MultiChannelExtractor> _mc; // multi-channel extractor function

  ChannelsVector _channels; //< template channels
  ChannelsVector _channels_warped; //< channels computed at the warped image

  PointVector _points; //< template points

  cv::Rect _bbox; //< template bounding box at the reference image

  cv::Mat _interp_maps[2]; //< x,y interpolation maps
  cv::Mat _Iw; //< warped input image buffer

  ResidualsVector _residuals; //< buffer to hold residuals

  /**
   * ChannelData, which containts jacobias and intensities
   */
  llvm::SmallVector<ChannelData, 16> _channel_data;

  /** hartly's normalization */
  Matrix33f _T, _T_inv;

  // interpolation constants
  int _interp; //< cv::INTER_LINEAR
  int _border; //< cv::BORDER_CONSTANT
  cv::Scalar _border_val = cv::Scalar(0.0);

  //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // Tracker::Impl

}; // bp

#endif // BITPLANES_CORE_INTERNAL_TRACKER_IMPL_H
