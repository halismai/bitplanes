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

#ifndef BITPLANES_CORE_CVFWD_H
#define BITPLANES_CORE_CVFWD_H

namespace cv {
class Mat;

template <class> class Rect_;
typedef Rect_<int> Rect;

template <class> class Point_;
typedef Point_<int> Point2i;
typedef Point2i Point;

}; // cv

#endif // BITPLANES_CORE_CVFWD_H
