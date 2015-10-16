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

#include "bitplanes/core/internal/imwarp.h"
#include "bitplanes/core/internal/intrin.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>

namespace bp {

void imwarp(const cv::Mat& src, cv::Mat& dst, const Matrix33f& T,
            const PointVector& points, const cv::Rect& box,
            cv::Mat& xmap, cv::Mat& ymap, bool is_projective,
            int interp, int border, float border_val)
{
#if 1
  //
  // should set this to zero if there are holes in the points
  // for now, we do this densely
  // TODO for sparse or subsampled points
  //
  xmap.create(box.size(), CV_32FC1);
  ymap.create(box.size(), CV_32FC1);

  assert( !xmap.empty() && !ymap.empty() && "failed to allocate" );
  assert( xmap.isContinuous() && ymap.isContinuous() && "maps must be continous");

  const int x_off = box.x, y_off = box.y;
  const int stride = xmap.cols;
  float* x_map = xmap.ptr<float>();
  float* y_map = ymap.ptr<float>();

  std::fill_n(x_map, xmap.rows * xmap.cols, 0.0f);
  std::fill_n(y_map, ymap.rows * ymap.cols, 0.0f);

  for(size_t i = 0; i < points.size(); ++i)
  {
    const Vector3f& p = points[i];
    Vector3f pw = T * p;
    pw *= is_projective ? (1.0f / pw[2]) : 1.0f;

    int y = p.y() - y_off, x = p.x() - x_off, ii = y*stride + x;
    assert( ii >= 0 && ii < xmap.rows*xmap.cols );

    x_map[ii] = pw.x();
    y_map[ii] = pw.y();
  }

  cv::remap(src, dst, xmap, ymap, interp, border, cv::Scalar(border_val));
#else
  cv::Mat M;
  cv::eigen2cv(T, M);
  int flags = interp | cv::WARP_INVERSE_MAP;
  cv::warpPerspective(src(box), dst, M, cv::Size(), flags, border,
                      cv::Scalar(border_val));
#endif
}

#if defined(BITPLANES_HAVE_SSE3)

namespace {

static inline __m128 sumv(const __m128& x)
{
  const auto v = _mm_hadd_ps(x, x);
  return _mm_hadd_ps(v, v);
}

} // namespace

int imwarp(const uint8_t* I, int w, int h, const float* P, const float* X,
           const float* I_ref, float* residuals, uint8_t* valid, int N,
           float* I_warped)
{
  int stride = w;
  int i = 0;

  int rounding_mode = _MM_GET_ROUNDING_MODE();
  if(_MM_ROUND_TOWARD_ZERO != rounding_mode) _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

  int flush_mode = _MM_GET_FLUSH_ZERO_MODE();
  if(_MM_FLUSH_ZERO_ON != flush_mode) _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  const auto c0 = _mm_load_ps( P + 0 ),
        c1 = _mm_load_ps( P + 4 ),
        c2 = _mm_load_ps( P + 8 ),
        c3 = _mm_load_ps( P + 12 );

  const auto LB = _mm_set1_epi32(-1);                // Lower bound
  const auto UB = _mm_set_epi32(h-1, w-1, h-1, w-1); // Upper bound
  const auto ONES = _mm_set1_ps(1.0f);
  const auto HALF = _mm_set1_ps(0.5f);
  const auto n = N & ~3; // we'll do 4 points at a time

  int num_valid = 0;

  for(i = 0; i < n; i += 4)
  {
    alignas(16) int buf[4];
    __m128 i0, i1, i2, i3;
    {
      __m128 x0, x1, xf;
      __m128i mask, xi;

      {
        auto p = _mm_load_ps(X + 4*i + 0),
             x = _mm_mul_ps(c0, _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,0,0,0))),
             y = _mm_mul_ps(c1, _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,1,1,1))),
             z = _mm_mul_ps(c2, _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,2,2)));
        x0 = _mm_add_ps(c3, _mm_add_ps(_mm_add_ps(x, y), z));
      }

      {
        auto p = _mm_load_ps(X + 4*i + 4),
             x = _mm_mul_ps(c0, _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,0,0,0))),
             y = _mm_mul_ps(c1, _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,1,1,1))),
             z = _mm_mul_ps(c2, _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,2,2)));
        x1 = _mm_add_ps(c3, _mm_add_ps(_mm_add_ps(x, y), z));
      }

      auto zzzz = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2,2,2,2));
      xf = _mm_div_ps(_mm_shuffle_ps(x0, x1, _MM_SHUFFLE(1,0,1,0)), zzzz);
      xi = _mm_cvtps_epi32(_mm_add_ps(xf, HALF));
      mask = _mm_and_si128(_mm_cmpgt_epi32(xi, LB), _mm_cmplt_epi32(xi, UB));
      xi   = _mm_and_si128(mask, xi);

      _mm_store_si128((__m128i*) buf, xi);

      valid[i + 0] = (buf[0] && buf[1]);
      valid[i + 1] = (buf[2] && buf[3]);

      xf = _mm_sub_ps(xf, _mm_cvtepi32_ps(xi));
      auto wx = _mm_sub_ps(ONES, xf);

      auto xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(0,0,0,0)),
           yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(1,1,1,1));
      yy = _mm_shuffle_ps(yy, yy, _MM_SHUFFLE(2,0,2,0));

      int u0 = buf[0], v0 = buf[1];
      const auto* ii = I + v0*stride + u0;
      auto I0 = static_cast<float>( *ii ),
           I1 = static_cast<float>( *(ii + 1) ),
           I2 = static_cast<float>( *(ii + stride) ),
           I3 = static_cast<float>( *(ii + stride + 1) );

      i0 = sumv( _mm_mul_ps(_mm_mul_ps(xx, yy), _mm_set_ps(I3, I2, I1, I0)) );

      xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(2,2,2,2));
      yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(3,3,3,3));
      yy = _mm_shuffle_ps(yy, yy, _MM_SHUFFLE(2,0,2,0));

      u0 = buf[2]; v0 = buf[3];
      ii = I + v0*stride + u0;
      I0 = static_cast<float>( *ii );
      I1 = static_cast<float>( *(ii + 1) );
      I2 = static_cast<float>( *(ii + stride) );
      I3 = static_cast<float>( *(ii + stride + 1) );

      i1 = sumv( _mm_mul_ps(_mm_mul_ps(xx, yy), _mm_set_ps(I3, I2, I1, I0)) );
    }

    {
      __m128 x0, x1, xf;
      __m128i mask, xi;

      {
        auto p = _mm_load_ps(X + 4*i + 8),
             x = _mm_mul_ps(c0, _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,0,0,0))),
             y = _mm_mul_ps(c1, _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,1,1,1))),
             z = _mm_mul_ps(c2, _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,2,2)));
        x0 = _mm_add_ps(c3, _mm_add_ps(_mm_add_ps(x, y), z));
      }

      {
        auto p = _mm_load_ps(X + 4*i + 12),
             x = _mm_mul_ps(c0, _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,0,0,0))),
             y = _mm_mul_ps(c1, _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,1,1,1))),
             z = _mm_mul_ps(c2, _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,2,2)));
        x1 = _mm_add_ps(c3, _mm_add_ps(_mm_add_ps(x, y), z));
      }

      auto zzzz = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2,2,2,2));
      xf = _mm_div_ps(_mm_shuffle_ps(x0, x1, _MM_SHUFFLE(1,0,1,0)), zzzz);
      xi = _mm_cvtps_epi32(_mm_add_ps(xf, HALF));
      mask = _mm_and_si128(_mm_cmpgt_epi32(xi, LB), _mm_cmplt_epi32(xi, UB));
      xi   = _mm_and_si128(mask, xi);

      _mm_store_si128((__m128i*) buf, xi);

      valid[i + 2] = (buf[0] && buf[1]);
      valid[i + 3] = (buf[2] && buf[3]);

      xf = _mm_sub_ps(xf, _mm_cvtepi32_ps(xi));
      auto wx = _mm_sub_ps(ONES, xf);

      auto xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(0,0,0,0)),
           yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(1,1,1,1));
      yy = _mm_shuffle_ps(yy, yy, _MM_SHUFFLE(2,0,2,0));

      int u0 = buf[0], v0 = buf[1];
      const auto* ii = I + v0*stride + u0;
      auto I0 = static_cast<float>( *ii ),
           I1 = static_cast<float>( *(ii + 1) ),
           I2 = static_cast<float>( *(ii + stride) ),
           I3 = static_cast<float>( *(ii + stride + 1) );

      i2 = sumv( _mm_mul_ps(_mm_mul_ps(xx, yy), _mm_set_ps(I3, I2, I1, I0)) );

      xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(2,2,2,2));
      yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(3,3,3,3));
      yy = _mm_shuffle_ps(yy, yy, _MM_SHUFFLE(2,0,2,0));

      u0 = buf[2]; v0 = buf[3];
      ii = I + v0*stride + u0;
      I0 = static_cast<float>( *ii );
      I1 = static_cast<float>( *(ii + 1) );
      I2 = static_cast<float>( *(ii + stride) );
      I3 = static_cast<float>( *(ii + stride + 1) );

      i3 = sumv( _mm_mul_ps(_mm_mul_ps(xx, yy), _mm_set_ps(I3, I2, I1, I0)) );
    }

    auto z1 = _mm_shuffle_ps(i0, i1, _MM_SHUFFLE(0,0,0,0)),
         z2 = _mm_shuffle_ps(i2, i3, _MM_SHUFFLE(0,0,0,0)),
         zz = _mm_shuffle_ps(z1, z2, _MM_SHUFFLE(2,0,2,0));

    _mm_store_ps(residuals + i, _mm_sub_ps(_mm_load_ps(I_ref + i), zz));

    if(I_warped)
      _mm_store_ps(I_warped + i, zz);

    num_valid += valid[i + 0] + valid[i + 1] + valid[i + 2] + valid[i + 3];
  }

  typedef Eigen::Map<const Eigen::Vector4f, Eigen::Aligned> Point4Map;
  const auto PP = Eigen::Map<const Eigen::Matrix4f, Eigen::Aligned>(P).block<3,4>(0,0);

  for( ; i < N; ++i)
  {
    Eigen::Vector3f Xw = PP * Point4Map(X + 4*i);
    float z_i = 1.0f / Xw[2];
    float xf = Xw[0] * z_i; // / Xw[2],
    float yf = Xw[1] * z_i; // / Xw[2];
    int xi = static_cast<int>( xf + 0.5f ),
        yi = static_cast<int>( yf + 0.5f );

    valid[i] = (xi >= 0) && (xi < w-1) && (yi >= 0) && (yi < h-1);

    if(valid[i])
    {
      xf -= xi;
      yf -= yi;

      const auto* p0 = I + yi*stride + xi;
      float i0 = static_cast<float>( *p0 ),
            i1 = static_cast<float>( *(p0 + 1) ),
            i2 = static_cast<float>( *(p0 + stride) ),
            i3 = static_cast<float>( *(p0 + stride + 1) ),
            Iw = (1.0f-yf) * ((1.0f-xf)*i0 + xf*i1) +
                       yf  * ((1.0f-xf)*i2 + xf*i3);
      residuals[i] = I_ref[i] - Iw;

      if(I_warped)
        I_warped[i] = Iw;

      num_valid += 1;
    }
  }

  if(_MM_ROUND_TOWARD_ZERO != rounding_mode) _MM_SET_ROUNDING_MODE(rounding_mode);
  if(_MM_FLUSH_ZERO_ON != flush_mode) _MM_SET_FLUSH_ZERO_MODE(flush_mode);

  return num_valid;
}

int imwarp3(const uint8_t* I_ptr, int w, int h, const float* H_ptr, const float* X,
            const float* I_ref, float* residuals, uint8_t* valid, int N, float* I_warped)
{
  int stride = w, max_cols = w - 1, max_rows = h - 1;
  int num_valid = 0;

  auto I = [=](int r, int c) { return *(I_ptr + r*stride + c); };

  const Eigen::Map<const Matrix33f> H(H_ptr);
  for(int i = 0; i < N; ++i)
  {
    Vector3f Xw = H * Eigen::Map<const Vector3f>(X + 3*i);
    Xw *= (1.0f / Xw[2]);

    float xf = Xw[0],
          yf = Xw[1];

    int xi = static_cast<int>(xf + 0.5f),
        yi = static_cast<int>(yf + 0.5f);

    xf -= xi;
    yf -= yi;

    if( xi >= 0 && xi < max_cols && yi >= 0 && yi < max_rows ) {
      valid[i] = 1;
      const float wx = 1.0 - xf;
      float Iw = (1.0 - yf) * ( I(yi,   xi)*wx + I(yi,   xi+1)*xf )
                      +  yf  * ( I(yi+1, xi)*wx + I(yi+1, xi+1)*xf );

      residuals[i] = I_ref[i] - Iw;

      if(I_warped)
        I_warped[i] = Iw;

      num_valid++;
    } else {
      valid[i] = false;
      residuals[i] = 0.0f;
      if(I_warped)
        I_warped[i] = 0.0f;
    }
  }

  return num_valid;
}

#else

int imwarp(const uint8_t* , int, int, const float*, const float*,
           const float*, float*, uint8_t*, int,
           float*)
{
  THROW_ERROR("simd::imwarp requires SSE3");
}

#endif

} // bp


