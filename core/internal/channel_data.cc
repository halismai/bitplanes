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

#include "bitplanes/core/internal/channel_data.h"
#include "bitplanes/core/config.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/translation.h"
#include "bitplanes/core/affine.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core/core.hpp>

#if defined(BITPLANES_WITH_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif // BITPLANES_WITH_TBB

#include <cassert>

namespace bp {

ChannelData::ChannelData(MotionType motion) : _motion_type(motion) {}

typedef Eigen::Matrix<float,1,2> ImageGradient;
typedef typename EigenStdVector<ImageGradient>::type ImageGradientVector;

template <typename T> static inline
void GetValidData(const cv::Mat& image, const PointVector& points,
                  ImageGradientVector& gradients, std::vector<float>& pixels,
                  std::vector<size_t>& inds)
{
  assert( image.isContinuous() && "image must be continous" );

  const T* I_ptr = image.ptr<const T>();
  const int stride = image.cols;

  gradients.clear();
  pixels.clear();
  inds.clear();

  const auto N = points.size();
  gradients.resize(N);
  pixels.resize(N);

  std::vector<uint8_t> tmp_inds(N, 0);

  int y_off = points[0].y(),
      x_off = points[0].x();

#define USE_SCHARR 0

#if USE_SCHARR
  cv::Mat Gx, Gy;
  cv::Scharr(image, Gx, CV_32F, 1, 0, 1.0 / 16.0);
  cv::Scharr(image, Gy, CV_32F, 0, 1, 1.0 / 16.0);
#endif


  size_t i = 0;
  for(i = 0; i < N; ++i)
  {
    int y = static_cast<int>( points[i].y() ) - y_off,
        x = static_cast<int>( points[i].x() ) - x_off;

    if(x > 0 && x < image.cols - 1 && y > 0 && y < image.rows - 1)
    {
      int ii = y*stride + x;

#if USE_SCHARR
      float Ix = Gx.at<float>(y,x),
            Iy = Gy.at<float>(y,x);
#else
      float Ix = (float) I_ptr[ii+1] - (float) I_ptr[ii-1],
            Iy = (float) I_ptr[ii+stride] - (float) I_ptr[ii-stride];
#endif

      Eigen::Matrix<float,1,2> G(Ix, Iy);
      if( true || G.array().abs().sum() > -1.0f )
      {
        tmp_inds[i] = 1;
        pixels[i] = I_ptr[ii];
        gradients[i] = 0.5f * G;
      }
    }
  }

  if(i < N) {
    pixels.erase(pixels.begin(), pixels.begin() + i);
    gradients.erase(gradients.begin(), gradients.begin() + i);
  }

  inds.clear();
  inds.resize(N);
  for(size_t j = 0, jj = 0; j < tmp_inds.size(); ++j)
    if(tmp_inds[j]) inds[jj++] = j;
}


template <class M> static inline void
ComputeJacobian(const ImageGradientVector& gradients, const PointVector& points,
                const std::vector<size_t>& inds, typename ChannelData::Matrix& jacobians,
                float s, float c1, float c2)
{
  typedef MotionModel<M> Motion;

  constexpr int DOF = Motion::DOF;
  const size_t npts = inds.size();

  Eigen::Matrix<float, DOF, Eigen::Dynamic> tmp_jacobian(DOF, npts);
  for(size_t i = 0; i < npts; ++i)
  {
    float Ix = gradients[i][0],
          Iy = gradients[i][1];

    size_t ii = inds[i];
    float x = points[ii].x(),
          y = points[ii].y();

    //tmp_jacobian.col(i) = Motion::ComputeJacobian(x, y, Ix, Iy, s, c1, c2);

    Motion::ComputeJacobian(tmp_jacobian.col(i), x, y, Ix, Iy, s, c1, c2);
  }

  jacobians = tmp_jacobian.transpose();
}

void ChannelData::set(const cv::Mat& image, const PointVector& points,
                      float s, float c1, float c2)
{
  //
  // the first point indicates the top left corner
  //

  assert( !points.empty() );
  assert( points[0].x() >= 0 && points[0].x() < image.cols &&
         points[0].y() >= 0 && points[0].y() < image.rows  );

  ImageGradientVector gradients;
  std::vector<float> tmp_pixels;

  //
  // find valid points based on the magnitude of the gradient
  //
  switch( image.type() )
  {
    case cv::DataType<uint8_t>::type:
      GetValidData<uint8_t>(image, points, gradients, tmp_pixels, _inds);
      break;

    case cv::DataType<float>::type:
      GetValidData<float>(image, points, gradients, tmp_pixels, _inds);
      break;

    default:
      throw std::runtime_error("invalid image/channel data type");
  }

  //
  // copy the intensities
  //
  _pixels.resize(tmp_pixels.size());
  memcpy(_pixels.data(), tmp_pixels.data(), sizeof(float) * tmp_pixels.size());

  //
  // compute the jacobians
  //

  switch(_motion_type)
  {
    case MotionType::Homography:
      ComputeJacobian<Homography>(gradients, points, _inds, _jacobian, s, c1, c2);
      break;
    case MotionType::Translation:
      ComputeJacobian<Translation>(gradients, points, _inds, _jacobian, s, c1, c2);
      break;
    case MotionType::Affine:
      ComputeJacobian<Affine>(gradients, points, _inds, _jacobian, s, c1, c2);
      break;
  }
}


template <typename T> inline
void ComputeResiduals(const T* ptr, const std::vector<size_t>& indices,
                      const Eigen::VectorXf& pixels, Eigen::VectorXf& E)
{
  assert( indices.size() == (size_t) pixels.size() );

  E.resize( indices.size() );
  for(size_t i = 0; i < indices.size(); ++i)
    E[i] = static_cast<float>(ptr[indices[i]]) - pixels[i];
}

template<> inline
void ComputeResiduals(const uint8_t* ptr, const std::vector<size_t>& indices,
                      const Eigen::VectorXf& pixels, Eigen::VectorXf& E)
{
  assert( indices.size() == (size_t) pixels.size() );

  if(E.size() != (int) indices.size())
    E.resize( indices.size() );

  float* e = (float*) __builtin_assume_aligned(E.data(), 16);
  const float* p = (float*) __builtin_assume_aligned(pixels.data(), 16);
  const size_t* inds = (size_t*) __builtin_assume_aligned(indices.data(), 16);

  for(size_t i = 0; i < indices.size(); ++i)
    e[i] = static_cast<float>(ptr[ inds[i] ]) - p[i];
}

void ChannelData::computeResiduals(const cv::Mat& Cw, Eigen::VectorXf& E) const
{
  assert( Cw.type() == cv::DataType<uint8_t>::type ||
          Cw.type() == cv::DataType<float>::type );

  switch( Cw.type() )
  {
    case cv::DataType<uint8_t>::type:
      ComputeResiduals(Cw.ptr<const uint8_t>(), indices(), pixels(), E);
      break;

    case cv::DataType<float>::type:
      ComputeResiduals(Cw.ptr<const float>(), indices(), pixels(), E);
      break;

    default:
      THROW_ERROR("Unsuppored image type");
  }
}


} // bp
