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

#include "bitplanes/core/internal/bitplanes_sparse_data.h"
#include "bitplanes/core/internal/census_signature.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

namespace bp {

static inline void NormalizePoints(const std::vector<cv::KeyPoint>& pts,
                                   float& s, float& c1, float& c2)
{
  Vector2f c(0.0, 0.0f);
  for(const auto& p : pts)
    c += Vector2f(p.pt.x, p.pt.y);
  c /= pts.size();

  float m = 0.0;
  for(const auto& p : pts)
    m += (Vector2f(p.pt.x,p.pt.y) - c).norm();
  m /= pts.size();

  s = sqrt(2.0f) / std::max(m, 1e-6f);
  c1 = c[0];
  c2 = c[1];
}

template <class M> auto
BitPlanesSparseData<M>::
set(const cv::Mat& I, const cv::Rect& roi, float s, float c1, float c2) -> Hessian
{
  THROW_ERROR_IF(!I.isContinuous(), "isContinuous()");

  std::vector<cv::KeyPoint> pts;
  cv::FAST(I(roi), pts, 1, true);

  if(s <= 0.0f)
    NormalizePoints(pts, s, c1, c2);

  _T << s, 0, -s*c1, 0, s, -s*c2, 0, 0, 1;
  _T_inv << 1/s, 0, c1, 0, 1/s, c2, 0, 0, 1;

  _roi = roi;

  typedef typename MotionModelType::WarpJacobian WarpJacobian;

  const size_t n = pts.size();
  _points.resize(n);
  _jacobian.resize(8*n, MotionModelType::DOF);
  _pixels.resize(8*n);

  //Hessian ret;
  //ret.setZero();
  for(size_t i = 0; i < n; ++i)
  {
    int y = pts[i].pt.y, x = pts[i].pt.x;
    _points[i][0] = x;
    _points[i][1] = y;
    _points[i][2] = 1.0f;

    const WarpJacobian Jw = MotionModelType::ComputeWarpJacobian(x, y, s, c1, c2);

    int ys = y + roi.y, xs = x + roi.x;
    uint8_t cc  = CensusSignature(I.ptr<uint8_t>(ys) + xs, I.cols);
    uint8_t cx0 = CensusSignature(I.ptr<uint8_t>(ys) + xs - 1, I.cols);
    uint8_t cx1 = CensusSignature(I.ptr<uint8_t>(ys) + xs + 1, I.cols);
    uint8_t cy0 = CensusSignature(I.ptr<uint8_t>(ys - 1) + xs, I.cols);
    uint8_t cy1 = CensusSignature(I.ptr<uint8_t>(ys + 1) + xs, I.cols);

    for(int b = 0; b < 8; ++b)
    {
      _pixels[8*i+b] = CensusBit<uint8_t>(cc, b);
      _jacobian.row(8*i+b) = 0.5f * Eigen::Matrix<float,1,2>(
          (CensusBit<float>( cx1, b ) - CensusBit<float>( cx0, b ) ),
          (CensusBit<float>( cy1, b ) - CensusBit<float>( cy0, b ) ) ) * Jw;

      //ret.noalias() += _jacobian.row(8*i+b).transpose() * _jacobian.row(8*i+b);
      //ret.template selfadjointView<Eigen::Upper>().rankUpdate(_jacobian.row(8*i+b), 1.0f);
    }
  }

  //return ret;
  return _jacobian.transpose() * _jacobian;
}


template <class M> void
BitPlanesSparseData<M>::
computeResiduals(const Matrix33f& T, const cv::Mat& I, Vector_<float>& residuals) const
{
  residuals.resize( _pixels.size() );

  cv::Mat patch;
  for(size_t i = 0; i < _points.size(); ++i)
  {
    Eigen::Matrix<float,3,1> p = T * _points[i];
    p *= (1.0f / p[2]);

    float xf = p[0] + _roi.x,
          yf = p[1] + _roi.y;

    cv::getRectSubPix(I, cv::Size(3,3), cv::Point2f(xf, yf), patch);
    uint8_t iw = CensusSignature(patch.ptr<uint8_t>(1) + 1, patch.cols);

    for(int b = 0; b < 8; ++b) {
      residuals[8*i+b] = CensusBit<float>(iw, b) - _pixels[8*i+b];
    }
  }
}

template <class M> float
BitPlanesSparseData<M>::
linearize(const cv::Mat& I, const Matrix33f& T, Gradient& g) const
{
  float ssd = 0.0f;
  g.setZero();
  cv::Mat patch;
  for(size_t i = 0; i < _points.size(); ++i)
  {
    Eigen::Matrix<float,3,1> p = T * _points[i];
    p *= (1.0f / p[2]);

    float xf = p[0] + _roi.x,
          yf = p[1] + _roi.y;

    cv::getRectSubPix(I, cv::Size(3,3), cv::Point2f(xf, yf), patch);
    uint8_t iw = CensusSignature(patch.ptr<uint8_t>(1) + 1, patch.cols);

    for(int b = 0; b < 8; ++b) {
      float err = (CensusBit<float>(iw, b) - _pixels[8*i+b]);
      ssd += err*err;
      g.noalias() += _jacobian.row(8*i+b) * err;
    }
  }

  return ssd;
}

template class BitPlanesSparseData<Homography>;

} // bp

