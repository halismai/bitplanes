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

#include "bitplanes/core/internal/normalization.h"
#include <opencv2/core.hpp>

namespace bp {

void HartlyNormalization(const cv::Rect& box, Matrix33f& T, Matrix33f& T_inv)
{
  float n_valid = (box.width-2)*(box.height-2);
  Vector2f c(0.0f, 0.0f);
  for(int y = 1; y < box.height-1; ++y)
    for(int x = 1; x < box.width-1; ++x)
      c += Vector2f(x + box.x, y + box.y);
  c /= n_valid;

  float m = 0.0f;
  for(int y = 1; y < box.height-1; ++y)
    for(int x = 1; x < box.width-1; ++x)
      m += (Vector2f(x + box.x, y + box.y) - c).norm();

  m /= n_valid;

  float s = sqrt(2.0f) / std::max(m, 1e-6f);

  T << s, 0, -s*c[0],
       0, s, -s*c[1],
       0, 0, 1;

  T_inv << 1.0f/s, 0, c[0],
           0, 1.0f/s, c[1],
           0, 0, 1;
}

} // bp


