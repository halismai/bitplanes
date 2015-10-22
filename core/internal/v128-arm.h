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

#ifndef BITPLANES_CORE_INTERNAL_V128_ARM_H
#define BITPLANES_CORE_INTERNAL_V128_ARM_H

#include "arm_neon.h"

namespace bp {

  typedef int32x4_t __m128i;

  struct v128
  {
    FORCE_INLINE v128() {}

    FORCE_INLINE v128(const uint8_t*) {}

    //FORCE_INLINE v128(__m128i) {}

    FORCE_INLINE const v128& load(const uint8_t* p)
    {
      return *this;
    }

    FORCE_INLINE const v128& loadu(const uint8_t* p)
    {
      return *this;
    }

    /**
     * set to constant
     */
    //FORCE_INLINE v128(int n) : _xmm(_mm_set1_epi8(n)) {}

    //FORCE_INLINE operator __m128i() const { return _xmm; }

    FORCE_INLINE v128 Zero()    { return _mm_setzero_si128(); }
    FORCE_INLINE v128 InvZero() { return v128(0xff); }
    FORCE_INLINE v128 One()     { return v128(0x01); }

    FORCE_INLINE void store(const void* p) const
    {
      _mm_store_si128((__m128i*) p, _xmm);
    }

    /**
     * stores the 16 byte values as 16 floats
     */
    FORCE_INLINE void storeFloat(float* p) const
    {
      const auto t1 = _mm_unpacklo_epi8( _xmm, _mm_setzero_si128() );
      const auto t2 = _mm_unpackhi_epi8( _xmm, _mm_setzero_si128() );

      _mm_storeu_ps(p + 0,  _mm_cvtepi32_ps( _mm_unpacklo_epi16(t1, _mm_setzero_si128()) ) );
      _mm_storeu_ps(p + 4,  _mm_cvtepi32_ps( _mm_unpackhi_epi16(t1, _mm_setzero_si128()) ) );
      _mm_storeu_ps(p + 8,  _mm_cvtepi32_ps( _mm_unpacklo_epi16(t2, _mm_setzero_si128()) ) );
      _mm_storeu_ps(p + 12, _mm_cvtepi32_ps( _mm_unpackhi_epi16(t2, _mm_setzero_si128()) ) );
    }

    /*
       FORCE_INLINE void storeFloatSign(float* p, v128 sign_mask)
       {
       }*/

    friend std::ostream& operator<<(std::ostream&, const v128&);


  }; // v128

}; // bp

#endif // BITPLANES_CORE_INTERNAL_V128_ARM_H

