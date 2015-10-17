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

#ifndef BITPLANES_CORE_INTERNAL_MC_DATA_H
#define BITPLANES_CORE_INTERNAL_MC_DATA_H

#include "bitplanes/core/types.h"
#include "bitplanes/core/motion_model.h"
#include "bitplanes/core/algorithm_parameters.h"
#include "bitplanes/core/internal/cvfwd.h"
#include "bitplanes/core/internal/channel_data_dense.h"
#include "bitplanes/core/internal/SmallVector.h"
#include "bitplanes/utils/memory.h"

namespace bp {

class ChannelExtractor;

template <class M>
class MultiChannelData
{
 public:
  typedef MotionModel<M> Motion;
  typedef llvm::SmallVector<ChannelDataDense<M>,16> CDataVector;

 public:
  MultiChannelData();

 public:
  /**
   * Set the data from the image
   */
  void setTemplate(const cv::Mat& src, const cv::Rect& box);

  /**
   * Warp and commpute the residuals
   */
  void computeResiduals(const cv::Mat& image, const Matrix33f& T,
                        Vector_<float>& residuals);

  inline size_t size() const { return _cdata.size(); }
  inline const ChannelDataDense<M>& operator[](int i) const { return _cdata[i]; }

 protected:
  CDataVector _cdata;

  Matrix33f _T, _T_inv;
}; // MultiChannelData
}; // bp

#endif // BITPLANES_CORE_INTERNAL_MC_DATA_H
