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

#ifndef BITPLANES_UTILS_TIMER_H
#define BITPLANES_UTILS_TIMER_H

#include <chrono>

namespace bp {

class Timer
{
  typedef std::chrono::milliseconds Milliseconds;

 public:
  inline Timer() { start(); }

  void start();
  Milliseconds stop();
  Milliseconds elapsed();

 protected:
  std::chrono::high_resolution_clock::time_point _start_time;
}; // Timer


template <class Func, class ...Args> static inline
double TimeCode(int N_rep, Func&& f, Args... args)
{
  Timer timer;
  for(int i = 0; i < N_rep; ++i)
    f(args...);
  auto t = timer.stop();
  return t.count() / (double) N_rep;
}

}; // bp

#endif // BITPLANES_UTILS_TIMER_H
