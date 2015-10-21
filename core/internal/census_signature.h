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

#ifndef BITPLANES_CORE_INTERNAL_CENSUS_SIGNATURE_H
#define BITPLANES_CORE_INTERNAL_CENSUS_SIGNATURE_H

#include "bitplanes/core/internal/v128.h"

namespace bp {

/**
 * Compute the census signature at the pixel pointed to by 'p'
 *
 * \param p pointer to pixel
 * \param s image stride
 * \return census signature
 */
FORCE_INLINE uint8_t CensusSignature(const uint8_t* p, int s)
{
  return
      ((*(p - s - 1) >= *p) << 0) |
      ((*(p - s    ) >= *p) << 1) |
      ((*(p - s + 1) >= *p) << 2) |
      ((*(p     - 1) >= *p) << 3) |
      ((*(p     + 1) >= *p) << 4) |
      ((*(p + s - 1) >= *p) << 5) |
      ((*(p + s    ) >= *p) << 6) |
      ((*(p + s + 1) >= *p) << 7) ;
}


template <typename T = uint8_t>
FORCE_INLINE T CensusBit(uint8_t ct, int b)
{
  return static_cast<T>( (ct & (1 << b)) >> b );
}

namespace {
static const __m128i K0x01 = _mm_set1_epi8(0x01);
static const __m128i K0x02 = _mm_set1_epi8(0x02);
static const __m128i K0x04 = _mm_set1_epi8(0x04);
static const __m128i K0x08 = _mm_set1_epi8(0x08);
static const __m128i K0x10 = _mm_set1_epi8(0x10);
static const __m128i K0x20 = _mm_set1_epi8(0x20);
static const __m128i K0x40 = _mm_set1_epi8(0x40);
static const __m128i K0x80 = _mm_set1_epi8(0x80);
}; // namespace


template <int> v128 CensusBit(const uint8_t*, const v128& c, int s);

template<> FORCE_INLINE
v128 CensusBit<0>(const uint8_t* p, const v128& c, int s)
{
  return (v128(p - s - 1) >= c) & K0x01;
}

template<> FORCE_INLINE
v128 CensusBit<1>(const uint8_t* p, const v128& c, int s)
{
  return (v128(p - s + 0) >= c) & K0x02;
}

template<> FORCE_INLINE
v128 CensusBit<2>(const uint8_t* p, const v128& c, int s)
{
  return (v128(p - s + 1) >= c) & K0x04;
}

template<> FORCE_INLINE
v128 CensusBit<3>(const uint8_t* p, const v128& c, int /*s*/)
{
  return (v128(p - 1) >= c) & K0x08;
}

template<> FORCE_INLINE
v128 CensusBit<4>(const uint8_t* p, const v128& c, int /*s*/)
{
  return (v128(p + 1) >= c) & K0x10;
}

template<> FORCE_INLINE
v128 CensusBit<5>(const uint8_t* p, const v128& c, int s)
{
  return (v128(p + s - 1) >= c) & K0x20;
}

template<> FORCE_INLINE
v128 CensusBit<6>(const uint8_t* p, const v128& c, int s)
{
  return (v128(p + s ) >= c) & K0x40;
}

template<> FORCE_INLINE
v128 CensusBit<7>(const uint8_t* p, const v128& c, int s)
{
  return (v128(p + s + 1) >= c) & K0x80;
}

/**
 * computes the signature 16 pixels at once
 */
FORCE_INLINE v128 CensusSignatureSIMD(const uint8_t* p, int s)
{
  const v128 c(p);
  return
      CensusBit<0>(p, c, s) | CensusBit<1>(p, c, s) | CensusBit<2>(p, c, s) |
      CensusBit<3>(p, c, s) |                         CensusBit<4>(p, c, s) |
      CensusBit<5>(p, c, s) | CensusBit<6>(p, c, s) | CensusBit<7>(p, c, s) ;
}

FORCE_INLINE void CensusSignature(const uint8_t* p, int s, uint8_t* dst)
{
  _mm_storeu_si128((__m128i*) dst, CensusSignatureSIMD(p, s));
}

}; // bp

#endif // BITPLANES_CORE_INTERNAL_CENSUS_SIGNATURE_H
