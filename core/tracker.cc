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
#include "bitplanes/core/homography.h"
#include "bitplanes/core/affine.h"
#include "bitplanes/core/translation.h"

#include "bitplanes/core/internal/tracker_impl.h"
#include "bitplanes/core/internal/ic.h"

#include "bitplanes/utils/error.h"

#include <opencv2/core/core.hpp>

namespace bp {

Tracker::Tracker(MotionType m, AlgorithmParameters p)
    : _impl(Tracker::Impl::Create(m, p)) {}

Tracker::~Tracker() {}

void Tracker::setTemplate(const cv::Mat& src, const cv::Rect& bbox)
{
  _impl->setTemplate(src, bbox);
}

Result Tracker::track(const cv::Mat& I, const Transform& T_init)
{
  return _impl->track(I, T_init);
}

UniquePointer<Tracker::Impl>
Tracker::Impl::Create(MotionType m, AlgorithmParameters p)
{
  UniquePointer<Tracker::Impl> ret;

  switch(p.linearizer)
  {
    case AlgorithmParameters::LinearizerType::InverseCompositional:
      {
        switch(m)
        {
          case MotionType::Homography:
            ret.reset(new InverseCompositionalImpl<Homography>(m,p));
            break;

          case MotionType::Affine:
            ret.reset(new InverseCompositionalImpl<Affine>(m, p));
            break;

          case MotionType::Translation:
            ret.reset(new InverseCompositionalImpl<Translation>(m, p));
            break;
        }

      } break;

    case AlgorithmParameters::LinearizerType::ForwardCompositional:
      {
        THROW_ERROR("not implemented");
      } break;
  }

  return ret;
}

} // bp


