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

#ifndef BITPLANES_CORE_BITPLANES_TRACKER_PYRAMID_H
#define BITPLANES_CORE_BITPLANES_TRACKER_PYRAMID_H

#include <bitplanes/core/bitplanes_tracker.h>
//#include <bitplanes/core/bitplanes_tracker_sparse.h>
#include <vector>
#include <iostream>

namespace bp {

template <class M>
class BitPlanesTrackerPyramid
{
  typedef BitplanesTracker<M> Tracker;

 public:
  typedef typename Tracker::Transform Transform;
  typedef typename Tracker::MotionModelType MotionModelType;

 public:
  /**
   * \param p algorithm parameters
   */
  BitPlanesTrackerPyramid(const AlgorithmParameters& p = AlgorithmParameters())
      : _alg_params(p)
  {
    std::cout << _alg_params << std::endl;
  }

  inline ~BitPlanesTrackerPyramid() {}

  /**
   * sets the template
   *
   * \param I reference image
   * \param bbox template location
   */
  void setTemplate(const cv::Mat&, const cv::Rect& bbox);

  /**
   * Tracks the template
   *
   * \param I input image
   * \param T pose to use for initialization
   */
  Result track(const cv::Mat&, const Transform&);

  /**
   * Tracks the template
   *
   * \param I input image
   *
   * Uses the previously estimated pose for initialization
   */
  inline Result track(const cv::Mat& I) {
    return track(I, _T_init);
  }

 private:
  AlgorithmParameters _alg_params;
  std::vector<Tracker> _pyramid;
  Transform _T_init = Transform::Identity();
}; // BitPlanesTrackerPyramid

}; // bp

#endif // BITPLANES_CORE_BITPLANES_TRACKER_PYRAMID_H

