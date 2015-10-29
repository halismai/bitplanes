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

#include <unistd.h>
#include <sys/time.h>

#include "bitplanes/core/config.h"
#include "bitplanes/utils/utils.h"

#include <cmath>
#include <chrono>
#include <cstdarg>
#include <vector>
#include <thread>
#include <iostream>
#include <functional>
#include <algorithm>

namespace bp {

int getNumberOfPyramidLevels(int min_image_dim, int min_image_res)
{
  return 1 + std::round(std::log2(min_image_dim / double(min_image_res)));
}

int roundUpTo(int n, int m)
{
  return m ? ( (n % m) ? n + m - (n % m) : n) : n;
}

using std::string;
using std::vector;

string MakeSuffix()
{
  auto suffix = datetime();
  std::replace(std::begin(suffix), std::end(suffix), ' ', '_');

  return suffix;
}

string datetime()
{
  auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  char buf[128];
  std::strftime(buf, sizeof(buf), "%a %b %d %H:%M:%S %Z %Y", std::localtime(&tt));
  return std::string(buf);
}


double GetWallClockInSeconds()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}

uint64_t UnixTimestampSeconds()
{
  return std::chrono::seconds(std::time(NULL)).count();
}

uint64_t UnixTimestampMilliSeconds()
{
  return static_cast<int>( 1000.0 * UnixTimestampSeconds() );
}

void Sleep(int32_t ms)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

string Format(const char* fmt, ...)
{
  vector<char> buf(1024);

  while(true) {
    va_list va;
    va_start(va, fmt);
    auto len = vsnprintf(buf.data(), buf.size(), fmt, va);
    va_end(va);

    if(len < 0 || len >= (int) buf.size()) {
      buf.resize(std::max((int)(buf.size() << 1), len + 1));
      continue;
    }

    return string(buf.data(), len);
  }
}

} // bp

