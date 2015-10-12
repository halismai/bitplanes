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

#ifndef BITPLANES_CORE_INTERNAL_INTRIN_H
#define BITPLANES_CORE_INTERNAL_INTRIN_H

#include "bitplanes/core/config.h"

#if defined(BITPLANES_HAVE_SSE2) && defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(BITPLANES_HAVE_SSE3) && defined(__SSE3__)
#include <pmmintrin.h>
#endif
#if defined(BITPLANES_HAVE_SSSE3) && defined(__SSSE3__)
#include <tmmintrin.h>
#endif
#if defined(BITPLANES_HAVE_SSE4_1) && defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#if (defined(BITPLANES_HAVE_SSE4_2) && defined(__SSE4_2__)) || \
    (defined(BITPLANES_HAVE_POPCNT) && defined(__POPCNT__))
#include <nmmintrin.h>
#endif
#if (defined(BITPLANES_HAVE_AVX) && defined(__AVX__)) || \
    (defined(BITPLANES_HAVE_AVX2) && defined(__AVX2__))
#include <immintrin.h>
#endif

#if defined(BITPLANES_HAVE_AVX) || defined(BITPLANES_HAVE_AVX2)
#define BITPLANES_DEFAULT_ALIGNMENT 32
#else
#define BITPLANES_DEFAULT_ALIGNMENT 16
#endif


#endif // BITPLANES_CORE_INTERNAL_INTRIN_H
