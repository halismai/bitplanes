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

#ifndef BITPLANES_CORE_INTERNAL_V128_H
#define BITPLANES_CORE_INTERNAL_V128_H

#include "bitplanes/core/internal/intrin.h"
#include <cstdint>
#include <iosfwd>

#if defined(BITPLANES_HAVE_SSE2)
/**
 * Holds a vector of 16 bytes (128 bits)
 */
struct v128
{
  __m128i _xmm; //< the vector

  FORCE_INLINE v128() {}

  /**
   * loads the data from vector (unaligned load)
   */
  FORCE_INLINE v128(const uint8_t* p)
      : _xmm(_mm_loadu_si128((const __m128i*)p)) {}

  /**
   * assign from __m128i
   */
  FORCE_INLINE v128(__m128i x) : _xmm(x) {}

  /**
   * set to constant
   */
  FORCE_INLINE v128(int n) : _xmm(_mm_set1_epi8(n)) {}

  FORCE_INLINE operator __m128i() const { return _xmm; }

  FORCE_INLINE v128 Zero()    { return _mm_setzero_si128(); }
  FORCE_INLINE v128 InvZero() { return v128(0xff); }
  FORCE_INLINE v128 One()     { return v128(0x01); }

  /**
   * stores the 16 byte values as 16 floats
   */
  FORCE_INLINE void storeFloat(float* p)
  {
    const auto t1 = _mm_unpacklo_epi8( _xmm, _mm_setzero_si128() );
    const auto t2 = _mm_unpackhi_epi8( _xmm, _mm_setzero_si128() );

    _mm_store_ps(p + 0,  _mm_cvtepi32_ps( _mm_unpacklo_epi16(t1, _mm_setzero_si128()) ) );
    _mm_store_ps(p + 4,  _mm_cvtepi32_ps( _mm_unpackhi_epi16(t1, _mm_setzero_si128()) ) );
    _mm_store_ps(p + 8,  _mm_cvtepi32_ps( _mm_unpacklo_epi16(t2, _mm_setzero_si128()) ) );
    _mm_store_ps(p + 12, _mm_cvtepi32_ps( _mm_unpackhi_epi16(t2, _mm_setzero_si128()) ) );
  }

  /*
  FORCE_INLINE void storeFloatSign(float* p, v128 sign_mask)
  {
  }*/

  friend std::ostream& operator<<(std::ostream&, const v128&);
}; // v128

/**
 */
FORCE_INLINE v128 max(v128 a, v128 b)
{
  return _mm_max_epu8(a, b);
}

FORCE_INLINE v128 min(v128 a, v128 b)
{
  return _mm_min_epu8(a, b);
}

FORCE_INLINE v128 operator==(v128 a, v128 b)
{
  return _mm_cmpeq_epi8(a, b);
}

FORCE_INLINE v128 operator>=(v128 a, v128 b)
{
  return (a == max(a, b));
}

FORCE_INLINE v128 operator>(v128 a, v128 b)
{
  return _mm_andnot_si128( min(a, b) == a, _mm_set1_epi8(0xff) );
}

FORCE_INLINE v128 operator<(v128 a, v128 b)
{
  return _mm_andnot_si128( max(a, b) == a, _mm_set1_epi8(0xff) );
}

FORCE_INLINE v128 operator<=(v128 a, v128 b)
{
  return (a == min(a, b));
}

FORCE_INLINE v128 operator&(v128 a, v128 b)
{
  return _mm_and_si128(a, b);
}

FORCE_INLINE v128 operator|(v128 a, v128 b)
{
  return _mm_or_si128(a, b);
}

FORCE_INLINE v128 operator^(v128 a, v128 b)
{
  return _mm_xor_si128(a, b);
}

FORCE_INLINE v128 operator>>(v128 a, int i)
{
  return _mm_srli_epi32(a, i);
}


#else
#error "Need SSE2"
#endif

#endif // BITPLANES_CORE_INTERNAL_V128_H
