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
  Vector2f c(0.0f, 0.0f);
  for(int y = 0; y <  box.height; ++y)
    for(int x = 0; x < box.width; ++x)
      c += Vector2f(x, y);
  c /= static_cast<float>( box.area() );

  float m = 0.0f;
  for(int y = 0; y < box.height; ++y)
    for(int x = 0; x < box.width; ++x)
      m += (Vector2f(x,y) - c).norm();

  m /= static_cast<float>( box.area() );

  float s = sqrt(2.0f) / std::max(m, 1e-6f);

  T << s, 0, -s*c[0],
       0, s, -s*c[1],
       0, 0, 1;

  T_inv << 1.0f/s, 0, c[0],
           0, 1.0f/s, c[1],
           0, 0, 1;
}

} // bp


