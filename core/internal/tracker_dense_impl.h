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

#ifndef BITPLANES_CORE_INTERNAL_TRACKER_DENSE_IMPL_H
#define BITPLANES_CORE_INTERNAL_TRACKER_DENSE_IMPL_H

#include "bitplanes/core/tracker_dense.h"
#include "bitplanes/core/internal/mc_data.h"

namespace bp {

template <class M>
class TrackerDense<M>::Impl
{
 public:
  inline Impl(AlgorithmParameters p) : _alg_params(p) {}
  inline ~Impl() {}

  inline void setTemplate(const cv::Mat& image, const cv::Rect& bbox)
  {
    _mc_data.setTemplate(image, bbox);
  }

  Result track(const cv::Mat& image, const Transform&);

 protected:
  AlgorithmParameters _alg_params;
  MultiChannelData<M> _mc_data;
}; // Impl

}; // bp

#endif // BITPLANES_CORE_INTERNAL_TRACKER_DENSE_IMPL_H
