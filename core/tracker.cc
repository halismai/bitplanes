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

#include "bitplanes/core/tracker.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/affine.h"
#include "bitplanes/core/translation.h"
#include "bitplanes/core/internal/mc_extractor.h"
#include "bitplanes/core/internal/channel_data.h"
#include "bitplanes/core/internal/SmallVector.h"

#include "bitplanes/utils/utils.h"
#include "bitplanes/utils/error.h"

#include <opencv2/core/core.hpp>


namespace bp {


struct Tracker::Impl
{
  typedef Tracker::Impl Self;

  inline Impl(MotionType m, AlgorithmParameters p)
      :  _alg_params(p), _motion_type(m),
      _mc(MultiChannelExtractor::Create(p.multi_channel_function)),
      _T(Matrix33f::Identity()), _T_inv(Matrix33f::Identity())
  {
    _mc->setSigma(p.sigma);
  }

  inline virtual ~Impl() {}

  virtual void setTemplate(const cv::Mat& image, const cv::Rect& box)
  {
    THROW_ERROR_IF(
        box.y < 1 || box.y + box.height >= image.rows ||
        box.x < 1 || box.x + box.width  >= image.cols,
        "Bounding box is outside of image boundaries");

    _bbox = box;
    _points.resize(box.area());
    Vector2f c(0.0f, 0.0f);

    for(int y = box.y, i = 0; y < box.y + box.height; ++y)
      for(int x = box.x; x < box.x + box.width; ++x, ++i)
      {
        _points[i] = Eigen::Vector3f(x, y, 1.0f);
        c += _points[i].head<2>();
      }

    c /= _points.size(); // center of mass
    float m = 0.0f;
    for(const auto& pt : _points) {
      m += (pt.head<2>() - c).norm();
    }
    m /= _points.size();

    float s = sqrt(2.0f) / std::max(m, 1e-6f);

    _T << s, 0, -s*c[0],
          0, s, -s*c[1],
          0, 0, 1;

    _T_inv << 1.0f/s, 0, c[0],
              0, 1.0f/s, c[1],
              0, 0, 1;
  }

  virtual Result track(const cv::Mat&, const Transform&)  = 0;

  inline void resizeChannelData(size_t n)
  {
    this->_channel_data.resize(n, this->_motion_type);
  }

  inline void computeChannels(const cv::Mat& src, const cv::Rect& bbox)
  {
    this->_mc->operator()(src, bbox, _channels);
  }

  inline void setChannelData()
  {
    const auto s = _T(0,0), c1 = _T_inv(0,2), c2 = _T_inv(1,2);
    this->resizeChannelData(_channels.size());
    for(size_t i = 0; i < _channel_data.size(); ++i)
      _channel_data[i].set(_channels[i], _points, s, c1, c2);
  }

  inline void allocateInterpMaps(const cv::Size& size)
  {
    _interp_maps[0].create(size, CV_32F);
    _interp_maps[1].create(size, CV_32F);
  }

  static UniquePointer<Self> Create(MotionType m, AlgorithmParameters p);

  AlgorithmParameters _alg_params;
  MotionType _motion_type;
  UniquePointer<MultiChannelExtractor> _mc;
  PointVector _points;
  cv::Rect _bbox;
  cv::Mat _interp_maps[2];
  cv::Mat _Iw;

  Matrix33f _T, _T_inv;

  /** holds the channels (opencv images) */
  typename MultiChannelExtractor::ChannelsVector _channels;

  /** holds the channel data */
  llvm::SmallVector<ChannelData, 16> _channel_data;
};

template <class Motion>
struct InverseCompositionalImpl : public Tracker::Impl
{
  typedef MotionModel<Motion> MotionModelType;
  typedef typename MotionModelType::Transform Transform;
  typedef typename MotionModelType::Hessian Hessian;
  typedef typename MotionModelType::Gradient Gradient;
  typedef typename MotionModelType::JacobianMatrix JacobianMatrix;
  typedef typename MotionModelType::ParameterVector ParameterVector;

  inline InverseCompositionalImpl(MotionType m, AlgorithmParameters p)
      : Tracker::Impl(m, p) {}

  inline void setTemplate(const cv::Mat& src, const cv::Rect& bbox)
  {
    Tracker::Impl::setTemplate(src, bbox);

    _T_init.setIdentity();
    this->computeChannels(src, bbox);
    this->setChannelData();
    this->allocateInterpMaps(bbox.size());

    // pre-compute the Hessian for IC
    _hessian.setZero();
    for(const auto& c : this->_channel_data)
      _hessian.noalias() += (c.jacobian().transpose() * c.jacobian());
  }


  inline Result track(const cv::Mat& I, const Tracker::Transform& T_init)
  {
    UNUSED(I);
    UNUSED(T_init);

    return Result();
  }

  Transform _T_init;
  Hessian _hessian;
  Gradient _gradient;
};

Tracker::Tracker(MotionType m, AlgorithmParameters p)
    : _impl(Tracker::Impl::Create(m, p)) {}

void Tracker::setTemplate(const cv::Mat& src, const cv::Rect& bbox)
{
  _impl->setTemplate(src, bbox);
}

Result Tracker::track(const cv::Mat& I, const Transform& T_init)
{
  return _impl->track(I, T_init);
}

/*
Result Tracker::track(const cv::Mat I)
{
  return _impl->track(I);
}*/

UniquePointer<Tracker::Impl>
Tracker::Impl::Create(MotionType m, AlgorithmParameters p)
{
  switch(p.linearizer)
  {
    case AlgorithmParameters::LinearizerType::InverseCompositional:
      {
        switch(m)
        {
          case MotionType::Homography:
            return UniquePointer<Tracker::Impl>(
                new InverseCompositionalImpl<Homography>(m, p));
            break;

          case MotionType::Affine:
            return UniquePointer<Tracker::Impl>(
                new InverseCompositionalImpl<Affine>(m, p));
            break;

          case MotionType::Translation:
            return UniquePointer<Tracker::Impl>(
                new InverseCompositionalImpl<Translation>(m, p));
            break;
        }
      } break;

    default:
      THROW_ERROR("not implemented");
  }

  return UniquePointer<Tracker::Impl>(new InverseCompositionalImpl<Homography>(m, p));
}

} // bp

