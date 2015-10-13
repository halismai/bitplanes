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
#include "bitplanes/core/internal/census.h"
#include "bitplanes/core/internal/gmag.h"
#include "bitplanes/core/internal/imsmooth.h"
#include "bitplanes/core/debug.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if defined(BITPLANES_WITH_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

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

void CensusChannelFast::operator()(const cv::Mat& src, const cv::Rect& bbox,
                                   ChannelsVector& channels)
{
  cv::Mat tmp;
  MultiChannelExtractor::smoothImage(src(bbox), tmp);

  channels.resize(1);
  simd::CensusTransform(tmp, channels[0]);
}

static inline std::array<int,8> CensusDefaultXSampling()
{
  return {
    {
      -1, 0, 1,
      -1,    1,
      -1, 0, 1
    }
  };
}

static inline std::array<int,8> CensusDefaultYSampling()
{
  return {
    {
      -1, -1, -1,
       0,      0,
       1,  1,  1
    }
  };
}


CensusChannel::CensusChannel()
  : _x_samples(CensusDefaultXSampling())
    , _y_samples(CensusDefaultYSampling()) {}

void CensusChannel::operator()(const cv::Mat& src, const cv::Rect& bbox,
                               ChannelsVector& channels)
{
  cv::Mat tmp;
  MultiChannelExtractor::smoothImage(src(bbox), tmp);

  channels.resize(1);
  CensusTransform(tmp, channels[0], _x_samples.data(), _x_samples.size(),
                  _y_samples.data(), _y_samples.size());
}

namespace {

/**
 * return the size of the patch from the given radius
 */
static inline int SizeFromRadius(int r)
{
  return (2*r + 1) * (2*r + 1);
}

/**
 * generates offsets for pixel comparison. This is put in its own function to
 * avoid any bugs with different pixel order comparisons
 */
static llvm::SmallVector<std::pair<int,int>, 32>
MakeNeighborhoodOffsetsForCensus(int radius)
{
  llvm::SmallVector<std::pair<int,int>, 32> ret(SizeFromRadius(radius));

  for(int r_i = -radius, i=0; r_i <= radius; ++r_i)
    for(int c_i = -radius; c_i <= radius; ++c_i)
    {
      if(0 == c_i && 0 == r_i)
        continue; // skip the center pixel

      ret[i++] = std::make_pair(r_i, c_i);
    }

  return ret;
}

template <typename T> inline float Heaviside(T x)
{
  return static_cast<float>( x <= T(0.0) );
}

template <typename T> static inline
void ExtractCensus(const cv::Mat& src, const cv::Rect& box,
                   const std::pair<int,int>& offset, cv::Mat& ret)
{
  ret.create(box.size(), cv::DataType<float>::type);
  const auto y_off = offset.first;
  const auto x_off = offset.second;

  for(int y = box.y; y < box.y + box.height; ++y)
  {
    for(int x = box.x; x < box.x + box.width; ++x)
    {
      int I0 = static_cast<int>( src.at<T>(y, x) ),
          I1 = static_cast<int>( src.at<T>(y+y_off, x+x_off) );

      ret.at<float>(y,x) = Heaviside( I1 - I0 );
    }
  }
}

} // namespace

void BitPlanes::operator()(const cv::Mat& src, const cv::Rect& bbox,
                           ChannelsVector& channels)
{
  channels.resize(SizeFromRadius(_radius));
  const auto offsets = MakeNeighborhoodOffsetsForCensus(_radius);

  assert( src.type() == cv::DataType<float>::type ||
          src.type() == cv::DataType<uint8_t>::type );

  switch( src.type() )
  {
    case cv::DataType<float>::type:
      {
        for(size_t i = 0; i < channels.size(); ++i)
          ExtractCensus<float>(src, bbox, offsets[i], channels[i]);
      }
      break;

    case cv::DataType<uint8_t>::type:
      {
        for(size_t i = 0; i < channels.size(); ++i)
          ExtractCensus<uint8_t>(src, bbox, offsets[i], channels[i]);
      } break;

    default:
      ;
  }
}


void BitPlanesFast::operator()(const cv::Mat& src, const cv::Rect& bbox,
                               ChannelsVector& channels)
{
  cv::Mat tmp;
  imsmooth(src(bbox), tmp, this->_sigma); // Gaussian is the best here

  channels.resize(8);

  const int stride = tmp.cols;
  const int offsets[8] = {
    - _offset - stride,
              - stride,
    + _offset - stride,
    + _offset         ,
    + _offset + stride,
    +           stride,
    - _offset + stride,
    - _offset };

#if defined(BITPLANES_WITH_TBB)
  tbb::parallel_for(
      tbb::blocked_range<int>(0,8),
      [&](const tbb::blocked_range<int>& r) {
        for(int i = r.begin(); i != r.end(); ++i)
        simd::CensusTransformChannel(tmp, offsets[i], channels[i], _offset);
      });
#else
  for(int i = 0; i < 8; ++i) {
    simd::CensusTransformChannel(tmp, offsets[i], channels[i], _offset);
  }
#endif
}

static inline
void MakeDescriptorFieldsChannel(const cv::Mat& G, cv::Mat& pos, cv::Mat& neg,
                                 float sigma)
{
  pos.create(G.size(), CV_32FC1);
  neg.create(G.size(), CV_32FC1);

  assert( G.type() == CV_32FC1 );
  auto* G_ptr = G.ptr<const float>();
  auto* pos_ptr = pos.ptr<float>();
  auto* neg_ptr = neg.ptr<float>();

  int N = G.rows * G.cols;
  for(int i = 0; i < N; ++i) {
    pos_ptr[i] = G_ptr[i] >= 0.0f ? G_ptr[i] : 0.0f;
    neg_ptr[i] = G_ptr[i] < 0.0f ? G_ptr[i] : 0.0f;
  }

  imsmooth(pos, sigma);
  imsmooth(neg, sigma);
}

void DescriptorFields1::operator()(const cv::Mat& I, const cv::Rect& bbox,
                                  ChannelsVector& channels)
{
  //
  // does not make sense to do this without the smoothing
  //
  auto sigma = MultiChannelExtractor::_sigma;
  assert( sigma > 0.0 );


  cv::Mat Ix, Iy;
  ImageGradient32f(I(bbox), Ix, Iy);

  channels.resize(4);

  MakeDescriptorFieldsChannel(Ix, channels[0], channels[1], sigma);
  MakeDescriptorFieldsChannel(Iy, channels[2], channels[3], sigma);
}

void DescriptorFields2::operator()(const cv::Mat& I, const cv::Rect& bbox,
                                  ChannelsVector& channels)
{
  //
  // does not make sense to do this without the smoothing
  //
  auto sigma = MultiChannelExtractor::_sigma;
  assert( sigma > 0.0 );


  cv::Mat Ix, Iy, Ixx, Iyy, Ixy;
  ImageGradient32f(I(bbox), Ix, Iy);
  ImageGradient32f(Ix, Ixx, Ixy);
  ImageGradient32f(Iy, Iyy, Ixy);

  channels.resize(10);

  MakeDescriptorFieldsChannel(Ix,  channels[0], channels[1], sigma);
  MakeDescriptorFieldsChannel(Iy,  channels[2], channels[3], sigma);
  MakeDescriptorFieldsChannel(Ixx, channels[4], channels[5], sigma);
  MakeDescriptorFieldsChannel(Iyy, channels[6], channels[7], sigma);
  MakeDescriptorFieldsChannel(Ixy, channels[8], channels[9], sigma);
}

} // bp

