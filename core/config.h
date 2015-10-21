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

/** DO NOT EDIT auto generated header */
#ifndef BITPLANES_CONFIG_H
#define BITPLANES_CONFIG_H

#define BITPLANES_VERSION_MAJOR "0"
#define BITPLANES_VERSION_MINOR "2"
#define BITPLANES_VERSION_PATCH "0"
#define BITPLANES_BUILD_DATE    "Tue Oct 20 23:45:13 EDT 2015"
#define BITPLANES_BUILD_STRING \
    "BitPlanes version: "      \
    BITPLANES_VERSION_MAJOR "." BITPLANES_VERSION_MINOR "." BITPLANES_VERSION_PATCH \
    "\nbuilt on: " BITPLANES_BUILD_DATE

#define BITPLANES_WITH_TCMALLOC 0
#define BITPLANES_WITH_PROFILER 0
#define BITPLANES_WITH_TBB 0
#define BITPLANES_WITH_BOOST 0
#define BITPLANES_WITH_TIMING 1

#define BITPLANES_HAVE_SSE2 1
#define BITPLANES_HAVE_SSE3 1
#define BITPLANES_HAVE_SSSE3 1
#define BITPLANES_HAVE_SSE4_1 1
#define BITPLANES_HAVE_SSE4_2 0
#define BITPLANES_HAVE_AVX 0
#define BITPLANES_HAVE_AVX2 0
#define BITPLANES_HAVE_POPCNT 0

#include <bitplanes/core/debug.h>

#endif // BITPLANES_CONFIG_H
