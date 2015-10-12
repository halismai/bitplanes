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

#ifndef BITPLANES_CORE_TRACKER_H
#define BITPLANES_CORE_TRACKER_H

#include "bitplanes/core/types.h"
#include "bitplanes/core/algorithm_parameters.h"
#include "bitplanes/core/internal/cvfwd.h"
#include "bitplanes/utils/memory.h"

namespace bp {


class InverseCompositionalImpl;

class Tracker
{
 public:
  typedef Matrix33f Transform;
  typedef Vector3f  Point;
  typedef typename EigenStdVector<Point>::type PointVector;

 public:
  Tracker(AlgorithmParameters p = AlgorithmParameters());

  inline ~Tracker() {}

  /**
   * set the template at the bounding box
   */
  void setTemplate(const cv::Mat& src, const cv::Rect& bbox);

  /**
   * Track the template in the give frame with the given initialization
   */
  Result track(const cv::Mat& I, const Transform& T_init = Transform::Identity());

 protected:
  struct Impl;

 protected:
  bp::UniquePointer<Impl> _impl;


  friend class InverseCompositionalImpl;
}; // Tracker

}; // bp

#endif // BITPLANES_CORE_TRACKER_H

