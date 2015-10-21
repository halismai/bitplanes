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

#include "bitplanes/core/config.h"
#include "bitplanes/core/internal/v128.h"
#include <iostream>

namespace bp {

#if BITPLANES_HAVE_SSE2
  std::ostream& operator<<(std::ostream& os, const v128& v)
  {
    ALIGNED(16) uint8_t buf[16];
    _mm_store_si128((__m128i*) buf, v);

    for(int i = 0; i < 16; ++i)
      os << static_cast<int>( buf[i] ) << " ";
    return os;
  }
#endif

}
