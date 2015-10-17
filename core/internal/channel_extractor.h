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

#ifndef BITPLANES_CORE_INTERNAL_CHANNEL_EXTRACTOR_H
#define BITPLANES_CORE_INTERNAL_CHANNEL_EXTRACTOR_H

#include "bitplanes/core/internal/mc_data.h"
#include <opencv2/core.hpp>
#include <vector>

namespace bp {

class ChannelExtractor
{
 public:
  typedef std::vector<cv::Mat> ImageVector;

 public:
  virtual ~ChannelExtractor();

 public:
  /**
   * Extract the channels at the input image within the specified rect
   */
  virtual void operator()(const cv::Mat&, const cv::Rect&, ImageVector&);
};

}; // bp

#endif // BITPLANES_CORE_INTERNAL_CHANNEL_EXTRACTOR_H
