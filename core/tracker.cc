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
#include "bitplanes/utils/utils.h"

#include <opencv2/core/core.hpp>

namespace bp {


struct Tracker::Impl
{
  typedef Tracker::Impl Self;

  inline Impl(AlgorithmParameters p)
      : _alg_params(p) {}

  inline virtual ~Impl() {}

  virtual void setTemplate(const cv::Mat&, const cv::Rect&) = 0;
  virtual Result track(const cv::Mat&, const Transform&)    = 0;

  static UniquePointer<Self> Create(AlgorithmParameters p);

  AlgorithmParameters _alg_params;

};

struct InverseCompositionalImpl : public Tracker::Impl
{
  inline InverseCompositionalImpl(AlgorithmParameters p) : Tracker::Impl(p) {}

  inline void setTemplate(const cv::Mat& src, const cv::Rect& bbox)
  {
    UNUSED(src);
    UNUSED(bbox);
  }

  inline Result track(const cv::Mat& I, const Tracker::Transform T_init)
  {
    UNUSED(I);
    UNUSED(T_init);

    return Result();
  }

};

Tracker::Tracker(AlgorithmParameters p)
    : _impl(Tracker::Impl::Create(p)) {}

void Tracker::setTemplate(const cv::Mat& src, const cv::Rect& bbox)
{
  _impl->setTemplate(src, bbox);
}

Result Tracker::track(const cv::Mat& I, const Transform& T_init)
{
  return _impl->track(I, T_init);
}

} // bp

