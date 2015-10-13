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

#include "bitplanes/utils/timer.h"
#include "bitplanes/core/config.h"

namespace bp {

void Timer::start()
{
#if defined(BITPLANES_WITH_TIMING)
  _start_time = std::chrono::high_resolution_clock::now();
#endif
}

auto Timer::stop() -> Milliseconds
{
#if defined(BITPLANES_WITH_TIMING)
  auto t_now = std::chrono::high_resolution_clock::now();
  auto ret = std::chrono::duration_cast<Milliseconds>(t_now - _start_time);
  _start_time = t_now;
  return ret;
#else
  return Milliseconds();
#endif
}

auto Timer::elapsed() -> Milliseconds
{
#if defined(BITPLANES_WITH_TIMING)
  auto t_now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<Milliseconds>(t_now - _start_time);
#else
  return Milliseconds();
#endif
}

} // bp

