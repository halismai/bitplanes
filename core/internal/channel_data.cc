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

#if BITPLANES_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif // BITPLANES_WITH_TBB

#include <cassert>

namespace bp {

ChannelData::ChannelData(MotionType motion) : _motion_type(motion) {}

typedef Eigen::Matrix<float,1,2> ImageGradient;
typedef typename EigenStdVector<ImageGradient>::type ImageGradientVector;

/**
 * \param image the input image
 * \param point vector of template points
 * \param pixels intensity values
 * \param jacobian jacobian of the warp
 */
template <typename T, class M> static inline
void GetValidData(const cv::Mat& image, const PointVector& points,
                  typename ChannelData::Vector& pixels,
                  typename ChannelData::Matrix& jacobian,
                  float s, float c1, float c2)
{

  const T* I_ptr = image.ptr<const T>();
  const int stride = image.cols;
  const auto n = (int) points.size();

  THROW_ERROR_IF(0 == n, "no points");

  typedef MotionModel<M> Motion;
  constexpr int DOF = Motion::DOF;

  if(pixels.size() != n) pixels.resize(n, 1);
  if(jacobian.rows() != n || jacobian.cols() != DOF) jacobian.resize(n, DOF);

  // first point is an offset
  int y_off = points[0].y(),
      x_off = points[0].x();

  ImageGradientVector G(n);
  for(int i = 0; i < n; ++i)
  {
    int y = static_cast<int>(points[i].y()) - y_off,
        x = static_cast<int>(points[i].x()) - x_off;

    if(x > 0 && x < image.cols - 1 && y > 0 && y < image.rows - 1)
    {
      int ii = y*stride + x;
      float Ix = 0.5f * ((float) I_ptr[ii+1] - (float) I_ptr[ii-1]),
            Iy = 0.5f * ((float) I_ptr[ii+stride] - (float) I_ptr[ii-stride]);

      G[i] = ImageGradient(Ix, Iy);
      pixels[i] = I_ptr[ii];

    } else
    {
      pixels[i] = 0.0f;
      G[i].setZero();
    }
  }

  for(size_t i = 0; i < G.size(); ++i)
  {
    typename Motion::Jacobian J;
    Motion::ComputeJacobian(J, points[i].x(), points[i].y(), G[i].x(), G[i].y(),
                            s, c1, c2);
    jacobian.row(i) = J;
  }
}

template <typename T> static inline
void GetValidData(const cv::Mat& image, const PointVector& points,
                  typename ChannelData::Vector& pixels,
                  typename ChannelData::Matrix& jacobian,
                  MotionType motion_type, float s, float c1, float c2)
{
  switch(motion_type)
  {
    case MotionType::Homography:
      {
        GetValidData<T,Homography>(image, points, pixels, jacobian, s, c1, c2);
      } break;

    case MotionType::Affine:
      {
        GetValidData<T,Affine>(image, points, pixels, jacobian, s, c1, c2);
      } break;

    case MotionType::Translation:
      {
        GetValidData<T,Translation>(image, points, pixels, jacobian, s, c1, c2);
      } break;
  }
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
      GetValidData<uint8_t>(image, points, _pixels, _jacobian, _motion_type,
                            s, c1, c2);
      break;

    case cv::DataType<float>::type:
      GetValidData<float>(image, points, _pixels, _jacobian, _motion_type,
                          s, c1, c2);
      break;

    default:
      throw std::runtime_error("invalid image/channel data type");
  }

}

template <typename T> inline
void ComputeResiduals(const T* ptr, const typename ChannelData::Vector& pixels,
                      Eigen::VectorXf& E)
{
  const auto n = pixels.size();
  if(E.size() != n)
    E.resize(n);

#pragma omp simd
  for(int i = 0; i < n; ++i)
    E[i] = static_cast<float>(ptr[i]) - pixels[i];
}

void ChannelData::computeResiduals(const cv::Mat& Cw, Eigen::VectorXf& E) const
{
  assert( Cw.type() == cv::DataType<uint8_t>::type ||
          Cw.type() == cv::DataType<float>::type );

  switch( Cw.type() )
  {
    case cv::DataType<uint8_t>::type:
      ComputeResiduals(Cw.ptr<const uint8_t>(), pixels(), E);
      break;

    case cv::DataType<float>::type:
      ComputeResiduals(Cw.ptr<const float>(), pixels(), E);
      break;

    default:
      THROW_ERROR("Unsuppored image type");
  }
}

} // bp

