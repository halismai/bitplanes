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

#include "bitplanes/core/internal/bitplanes_channel_data_subsampled.h"
#include "bitplanes/core/internal/ct.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/debug.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core.hpp>

#include <type_traits>

namespace bp {


static inline int GetNumValid(const cv::Rect& roi, int s)
{
  int ret = 0;
  for(int y = 1; y < roi.height-1; y += s)
    for(int x = 1; x < roi.width-1; x += s)
      ++ret;

  return ret;
}

template <class M>
void BitPlanesChannelDataSubSampled<M>::
set(const cv::Mat& src, const cv::Rect& roi, float s, float c1, float c2)
{
  THROW_ERROR_IF(roi.x < 1 || roi.x > src.cols - 1 ||
                 roi.y < 1 || roi.y > src.rows - 1,
                 "template bounding box is outside image");

  THROW_ERROR_IF( s <= 0, "scale cannot be negative or 0" );

  auto n_valid = GetNumValid(roi, _sub_sampling);
  _pixels.resize(n_valid);
  _jacobian.resize(8*n_valid, M::DOF);

  cv::Mat C;
  simd::census(src, roi, C);
  int stride = C.cols;

  /**
   * compute the channel x- and y-gradient at col:=x for bit:=b
   */
  auto G = [=](const uint8_t* p, int x, int b)
  {
    float ix1 = (p[x+1] & (1<<b)) >> b, ix2 = (p[x-1] & (1<<b)) >> b,
          iy1 = (p[x+stride] & (1<<b)) >> b, iy2 = (p[x-stride] & (1<<b)) >> b;
    return Eigen::Matrix<float,1,2>(0.5f*(ix1-ix2), 0.5f*(iy1-iy2));
  }; //

  typename M::WarpJacobian Jw;
  auto* pixels_ptr = _pixels.data();
  int i = 0;
  for(int y = 1; y < C.rows - 1; y += _sub_sampling)
  {
    const auto* srow = C.ptr<const uint8_t>(y);
    for(int x = 1; x < C.cols - 1; x += _sub_sampling, i+=8)
    {
      Jw = M::ComputeWarpJacobian(x+roi.x, y+roi.y, s, c1, c2);
      *pixels_ptr++ = srow[x];
      _jacobian.row(i+0) = G(srow, x, 0) * Jw;
      _jacobian.row(i+1) = G(srow, x, 1) * Jw;
      _jacobian.row(i+2) = G(srow, x, 2) * Jw;
      _jacobian.row(i+3) = G(srow, x, 3) * Jw;
      _jacobian.row(i+4) = G(srow, x, 4) * Jw;
      _jacobian.row(i+5) = G(srow, x, 5) * Jw;
      _jacobian.row(i+6) = G(srow, x, 6) * Jw;
      _jacobian.row(i+7) = G(srow, x, 7) * Jw;
    }
  }

  _hessian = _jacobian.transpose() * _jacobian;
  _roi_stride = roi.width;
}

template <class M>
void BitPlanesChannelDataSubSampled<M>::
computeResiduals(const cv::Mat& Iw, Residuals& residuals) const
{
  //simd::census_residual_packed(Iw, _pixels, residuals, _sub_sampling, _roi_stride);
  typedef int8_t CType;
  cv::AutoBuffer<CType> buf(8*_pixels.size());
  CType* r_ptr = buf;
  const uint8_t* c0_ptr = _pixels.data();

  const int src_stride = Iw.cols;
  for(int y = 1; y < Iw.rows - 1; y += _sub_sampling)
  {
    const uint8_t* srow = Iw.ptr<const uint8_t>(y);

#pragma omp simd
    for(int x = 1; x < Iw.cols - 1; x += _sub_sampling)
    {
      const uint8_t* p = srow + x;
      const uint8_t c = *c0_ptr++;
      *r_ptr++ = (*(p - src_stride - 1) >= *p) - ((c & (1<<0)) >> 0);
      *r_ptr++ = (*(p - src_stride    ) >= *p) - ((c & (1<<1)) >> 1);
      *r_ptr++ = (*(p - src_stride + 1) >= *p) - ((c & (1<<2)) >> 2);
      *r_ptr++ = (*(p              - 1) >= *p) - ((c & (1<<3)) >> 3);
      *r_ptr++ = (*(p              + 1) >= *p) - ((c & (1<<4)) >> 4);
      *r_ptr++ = (*(p + src_stride - 1) >= *p) - ((c & (1<<5)) >> 5);
      *r_ptr++ = (*(p + src_stride    ) >= *p) - ((c & (1<<6)) >> 6);
      *r_ptr++ = (*(p + src_stride + 1) >= *p) - ((c & (1<<7)) >> 7);
    }
  }

  using namespace Eigen;
  residuals=Map<Vector_<CType>, Aligned>(buf,_pixels.size()*8,1).template cast<float>();
}

template <class Derived> static inline
Eigen::Matrix<typename Derived::PlainObject::Scalar,
    Derived::PlainObject::RowsAtCompileTime, Derived::PlainObject::ColsAtCompileTime>
normHomog(const Eigen::MatrixBase<Derived>& x)
{
  static_assert(Derived::PlainObject::RowsAtCompileTime != Eigen::Dynamic &&
                Derived::PlainObject::ColsAtCompileTime == 1,
                "normHomog: input must be a vector of known dimension");

  return x * (1.0f / x[Derived::PlainObject::RowsAtCompileTime-1]);
}


template <class M>
void BitPlanesChannelDataSubSampled<M>::
warpImage(const cv::Mat& src, const Transform& T, const cv::Rect& roi,
          cv::Mat& dst, int interp, float border)
{
#if 1
  cv::Mat xmap(roi.size(), CV_32FC1);
  cv::Mat ymap(roi.size(), CV_32FC1);

  THROW_ERROR_IF( xmap.empty() || ymap.empty(), "Failed to allocate interp maps" );

  using namespace Eigen;

  const int x_s = roi.x, y_s = roi.y;
  for(int y = 0; y < roi.height; ++y)
  {
    auto* xm_ptr = xmap.ptr<float>(y);
    auto* ym_ptr = ymap.ptr<float>(y);

    int yy = y + y_s;

    for(int x = 0; x < roi.width; ++x)
    {
      const Vector3f pw = normHomog(T*Vector3f(x + x_s, yy, 1.0f));
      xm_ptr[x] = pw[0];
      ym_ptr[x] = pw[1];
    }
  }

  cv::remap(src, dst, xmap, ymap, interp, cv::BORDER_CONSTANT, cv::Scalar(border));
#else

  cv::Mat map1(roi.size(), CV_16SC2);
  cv::Mat map2(roi.size(), CV_16UC1);

  THROW_ERROR_IF( map1.empty() || map2.empty(), "Failed to allocate interp maps" );

  using namespace Eigen;

  const int x_s = roi.x, y_s = roi.y;
  for(int y = 0; y < roi.height; ++y)
  {
    auto* dst1 = map1.ptr<short>(y);
    auto* dst2 = map2.ptr<ushort>(y);

    for(int x = 0; x < roi.width; ++x)
    {
      const Vector3f pw = normHomog(T*Vector3f(x + x_s, y + y_s, 1.0f)) * cv::INTER_TAB_SIZE;
      int ix = cv::saturate_cast<int>(pw.x());
      int iy = cv::saturate_cast<int>(pw.y());
      dst1[2*x + 0] = cv::saturate_cast<short>( ix >> cv::INTER_BITS );
      dst1[2*x + 1] = cv::saturate_cast<short>( iy >> cv::INTER_BITS );
      dst2[x] = (ushort) ((iy & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE +
                          (ix & (cv::INTER_TAB_SIZE-1)));
    }
  }

  cv::remap(src, dst, map1, map2, interp, cv::BORDER_CONSTANT, cv::Scalar(border));

#endif
}

template <class M>
void BitPlanesChannelDataSubSampled<M>::
getCoordinateNormalization(const cv::Rect& /*roi*/, Transform& T, Transform& T_inv) const
{
  T.setIdentity();
  T_inv.setIdentity();
}

template <>
void BitPlanesChannelDataSubSampled<Homography>::
getCoordinateNormalization(const cv::Rect& roi, Transform& T, Transform& T_inv) const
{
  Vector2f c(0,0);

  int n_valid = 0;
  for(int y = 1; y < roi.height-1; y += _sub_sampling)
    for(int x = 1; x < roi.width-1; x += _sub_sampling, ++n_valid)
      c += Vector2f(x + roi.x, y + roi.y);
  c /= n_valid;

  float m = 0.0f;
  for(int y = 1; y < roi.height-1; y += _sub_sampling)
    for(int x = 1; x < roi.width-1; x += _sub_sampling)
      m += (Vector2f(x+roi.x, y + roi.y) - c).norm();
  m /= n_valid;

  float s = sqrt(2.0f) / std::max(m, 1e-6f);

  T << s, 0, -s*c[0],
       0, s, -s*c[1],
       0, 0, 1;

  T_inv << 1.0f/s, 0, c[0],
           0, 1.0f/s, c[1],
           0, 0, 1;
}

template class BitPlanesChannelDataSubSampled<Homography>;

}

