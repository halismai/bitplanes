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

#ifndef BITPLANES_CORE_TRACKER_DENSE_H
#define BITPLANES_CORE_TRACKER_DENSE_H

#include "bitplanes/core/types.h"
#include "bitplanes/core/algorithm_parameters.h"
#include "bitplanes/core/internal/cvfwd.h"
#include "bitplanes/utils/memory.h"

namespace bp {

template <class M>
class TrackerDense
{
 public:
  typedef Matrix33f Transform;

 public:
  TrackerDense(AlgorithmParameters p = AlgorithmParameters());
  ~TrackerDense();

  void setTemplate(const cv::Mat&, const cv::Rect&);
  Result track(const cv::Mat&, const Transform& = Transform::Identity());

 protected:
  class Impl;
  UniquePointer<Impl> _impl;
}; // TrackerDense

}; // bp

#endif // BITPLANES_CORE_TRACKER_DENSE_H
