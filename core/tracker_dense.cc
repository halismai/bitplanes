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

#include "bitplanes/core/tracker_dense.h"
#include "bitplanes/core/internal/tracker_dense_impl.h"
#include "bitplanes/core/homography.h"

#include <opencv2/core.hpp>

namespace bp {

template <class M>
TrackerDense<M>::TrackerDense(AlgorithmParameters p) : _impl(make_unique<Impl>(p)) {}

template <class M>
TrackerDense<M>::~TrackerDense() {}

template <class M>
void TrackerDense<M>::setTemplate(const cv::Mat& image, const cv::Rect& bbox)
{
  _impl->setTemplate(image, bbox);
}

template <class M>
Result TrackerDense<M>::track(const cv::Mat& image, const Transform& T_init)
{
  return _impl->track(image, T_init);
}

template <class M> Result
TrackerDense<M>::Impl::track(const cv::Mat& /*I*/, const Transform& /*T*/)
{
  return Result();
}

template class TrackerDense<Homography>;

}


