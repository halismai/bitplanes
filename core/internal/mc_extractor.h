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

#ifndef BITPLANES_CORE_INTERNAL_MC_EXTRACTOR_H
#define BITPLANES_CORE_INTERNAL_MC_EXTRACTOR_H

#include "bitplanes/core/algorithm_parameters.h"

#include "bitplanes/core/internal/cvfwd.h"
#include "bitplanes/core/internal/SmallVector.h"

#include <array>

namespace bp {

/**
 * base class for multi-channel extraction types
 */
class MultiChannelExtractor
{
 public:
  typedef llvm::SmallVector<cv::Mat,16> ChannelsVector;

 public:
  /**
   * \param sigma std. deviation of Gaussian to pre-smooth the image before
   * computing the channels.
   *
   * If sigma < 0.0, no smoothing is applied
   */
  MultiChannelExtractor(double sigma = -1.0);

  virtual ~MultiChannelExtractor();

 public:
  /**
   * Extract the channels at the specified bounding box
   */
  virtual void operator()(const cv::Mat& src, const cv::Rect&, ChannelsVector& dst) = 0;

  void setSigma(double s);
  double getSigma() const;

 public:
  /**
   * \return pointer from the MultiChannelExtractorType enum
   */
  static MultiChannelExtractor* Create(AlgorithmParameters::MultiChannelExtractorType);

 protected:
  virtual void smoothImage(const cv::Mat& src, cv::Mat& dst);

 protected:
  double _sigma = -1.0;
}; // MultiChannelExtractor


/**
 * Trivial mult-channel. This is only a single channel composed of raw data
 * Use the class for classic LK
 */
class IntensityGrayChannel : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

 public:
  inline IntensityGrayChannel() {}
  inline virtual ~IntensityGrayChannel() {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);
}; // IntensityGrayChannel

/**
 * Return intensity values as well as image gradients
 * 3-channels
 */
class IntensityAndGradient : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

 public:
  inline IntensityAndGradient() {}
  inline virtual ~IntensityAndGradient() {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);
}; // IntensityAndGradient


/**
 *
 */
class GradientAbsMag : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

 public:
  inline GradientAbsMag() {}
  inline virtual ~GradientAbsMag() {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);
}; // GradientAbsMag


/**
 * Single channel (Census)
 */
class CensusChannel : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

 public:

 public:
  CensusChannel();
  inline virtual ~CensusChannel() {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);

  inline void setSampling(std::array<int,8> xs, std::array<int,8> ys)
  {
    _x_samples = xs;
    _y_samples = ys;
  }

 protected:
  /** specifies sampling locations */
  std::array<int,8> _x_samples;
  std::array<int,8> _y_samples;
}; // CensusChannel


/**
 * Optimized implementation with fixed sampling locations
 */
class CensusChannelFast : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

  inline CensusChannelFast() {}
  inline virtual ~CensusChannelFast() {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);
}; // CensusChannelFast


/**
 * Generic any radius is supported
 */
class BitPlanes : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

 public:
  BitPlanes(int radius = 1) : _radius(radius) {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);

 protected:
  int _radius;
}; // BitPlanes

/**
 * Fast implementation, there will be 8 channels. The offset argument is how
 * far to test pixels away from the center. A value of 1 tests immediate
 * neighbors (original census/lbp)
 */
class BitPlanesFast : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

 public:
  inline BitPlanesFast(float s = -1.0f, int offset = 1)
      : MultiChannelExtractor(s), _offset(offset) {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);

 protected:
  int _offset;

}; //  BitPlanesFast


/**
 * Implementation of
 *
 * A. Crivellaro, and V. Leptit, ``Robust 3D Tracking with Descriptor Fields'',
 * CVPR, 2014
 *
 */
class DescriptorFields1 : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

 public:
  inline DescriptorFields1() {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);
}; // DescriptorFields

/**
 * Implementation of
 *
 * A. Crivellaro, and V. Leptit, ``Robust 3D Tracking with Descriptor Fields'',
 * CVPR, 2014
 *
 */
class DescriptorFields2 : public MultiChannelExtractor
{
 public:
  typedef typename MultiChannelExtractor::ChannelsVector ChannelsVector;

 public:
  inline DescriptorFields2() {}

  void operator()(const cv::Mat&, const cv::Rect&, ChannelsVector&);
}; // DescriptorFields


}; // bp

#endif // BITPLANES_CORE_INTERNAL_MC_EXTRACTOR_H

