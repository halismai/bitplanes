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

#include "bitplanes/core/internal/mc_extractor.h"
#include "bitplanes/core/internal/imsmooth.h"
#include "bitplanes/core/internal/gmag.h"
#include "bitplanes/core/debug.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace bp {

MultiChannelExtractor::MultiChannelExtractor(double sigma)
    : _sigma(sigma) {}

MultiChannelExtractor::~MultiChannelExtractor() {}

void MultiChannelExtractor::setSigma(double s) { _sigma = s; }

double MultiChannelExtractor::getSigma() const { return _sigma; }

void MultiChannelExtractor::smoothImage(const cv::Mat& src, cv::Mat& dst)
{
  imsmooth(src, dst, _sigma);
}

MultiChannelExtractor*
MultiChannelExtractor::Create(AlgorithmParameters::MultiChannelExtractorType m)
{
  MultiChannelExtractor* ret(nullptr);
  switch(m)
  {
    case AlgorithmParameters::MultiChannelExtractorType::IntensityGrayChannel:
      Info("IntensityGrayChannel\n");
      ret = new IntensityGrayChannel;
      break;

    case AlgorithmParameters::MultiChannelExtractorType::GradientAbsMag:
      Info("GradientAbsMag\n");
      ret = new GradientAbsMag;
      break;

    case AlgorithmParameters::MultiChannelExtractorType::IntensityAndGradient:
      Info("IntensityAndGradient\n");
      ret = new IntensityAndGradient;
      break;

    case AlgorithmParameters::MultiChannelExtractorType::CensusChannel:
      Info("CensusChannelFast\n");
      ret = new CensusChannelFast;
      break;

    case AlgorithmParameters::MultiChannelExtractorType::DescriptorFields1:
      Info("DescriptorFields\n");
      ret = new DescriptorFields1;
      break;

    case AlgorithmParameters::MultiChannelExtractorType::DescriptorFields2:
      Info("DescriptorFields\n");
      ret = new DescriptorFields2;
      break;

    case AlgorithmParameters::MultiChannelExtractorType::BitPlanes:
      Info("BitPlanesFast\n");
      ret = new BitPlanesFast;
      break;
  }

  return ret;
}

void IntensityGrayChannel::operator()(const cv::Mat& src, const cv::Rect& bbox,
                                      ChannelsVector& channels)
{
  channels.resize(1);

  MultiChannelExtractor::smoothImage(src(bbox), channels[0]);
  channels[0].convertTo(channels[0], CV_32FC1);
}

inline void ImageGradient32f(const cv::Mat& src, cv::Mat& Ix, cv::Mat& Iy)
{
  cv::Scharr(src, Ix, CV_32FC1, 1, 0, 1.0/16.0);
  cv::Scharr(src, Iy, CV_32FC1, 0, 1, 1.0/16.0);
}

void IntensityAndGradient::operator()(const cv::Mat& src, const cv::Rect& bbox,
                                      ChannelsVector& channels)
{
  channels.resize(3);
  MultiChannelExtractor::smoothImage(src(bbox), channels[0]);

  channels[0].convertTo(channels[0], CV_32FC1);
  channels[0] *= (1.0f / 255.f);

  ImageGradient32f(channels[0], channels[1], channels[2]);
}

void GradientAbsMag::operator()(const cv::Mat& src, const cv::Rect& bbox,
                                ChannelsVector& channels)
{
  assert( src.type() == CV_8UC1 );

  cv::Mat tmp;
  MultiChannelExtractor::smoothImage(src(bbox), tmp);

  channels.resize(1);
  simd::gradientAbsMag(tmp, channels[0]);
}



} // bp
