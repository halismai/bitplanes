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
#define BITPLANES_VERSION_MINOR "4"
#define BITPLANES_VERSION_PATCH "0"
<<<<<<< HEAD
#define BITPLANES_BUILD_DATE    "Mon Oct 26 17:48:46 EDT 2015"
=======
#define BITPLANES_BUILD_DATE    "Sun Oct 25 00:50:40 EDT 2015"
>>>>>>> 7fc55994140d684ba0792504fc801940a36ef859
#define BITPLANES_BUILD_STRING \
    "BitPlanes version: "      \
    BITPLANES_VERSION_MAJOR "." BITPLANES_VERSION_MINOR "." BITPLANES_VERSION_PATCH \
    "\nbuilt on: " BITPLANES_BUILD_DATE

#define BITPLANES_WITH_TCMALLOC 0
#define BITPLANES_WITH_PROFILER 0
#define BITPLANES_WITH_TBB 1
#define BITPLANES_WITH_BOOST 0
#define BITPLANES_WITH_TIMING 1
#define BITPLANES_WITH_OPENMP 1

<<<<<<< HEAD
#define BITPLANES_HAVE_ARM 1
#define BITPLANES_HAVE_SSE2 0
#define BITPLANES_HAVE_SSE3 0
#define BITPLANES_HAVE_SSSE3 0
#define BITPLANES_HAVE_SSE4_1 0
#define BITPLANES_HAVE_SSE4_2 0
#define BITPLANES_HAVE_AVX 0
=======
#define BITPLANES_HAVE_ARM 0
#define BITPLANES_HAVE_SSE2 1
#define BITPLANES_HAVE_SSE3 1
#define BITPLANES_HAVE_SSSE3 1
#define BITPLANES_HAVE_SSE4_1 1
#define BITPLANES_HAVE_SSE4_2 1
#define BITPLANES_HAVE_AVX 1
>>>>>>> 7fc55994140d684ba0792504fc801940a36ef859
#define BITPLANES_HAVE_AVX2 0
#define BITPLANES_HAVE_POPCNT 1

#include <bitplanes/core/debug.h>

#endif // BITPLANES_CONFIG_H
