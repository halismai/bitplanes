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

#ifndef BP_UTILS_UTILS_H
#define BP_UTILS_UTILS_H

#include <cstring>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <cinttypes>
#include <string>
#include <utility>
#include <type_traits>

namespace bp {

template <int Alignment = 16, class T> inline
bool IsAligned(const T* ptr)
{
  return 0 == ((unsigned long) ptr & (Alignment-1));
}

/**
 * \return the minimum number >= size that is divisible by 'N'
 */
template <int N> static inline
size_t alignSize(size_t size)
{
  assert( !(N & (N-1)) );
  return (size + N-1) & -N;
}

/**
 * Align the pointer to N bytes
 */
template <typename T, int N = static_cast<int>(sizeof(T))> static inline
T* alignPtr(T* ptr)
{
  return (T*)(((size_t) ptr + N-1) & -N);
}

template <class ...T> inline void UNUSED(const T&...) {}

/**
 * \return first (least significant) bit (see ffs())
 *
 * NOTE from folly
 */
template <class T> inline constexpr
typename std::enable_if<
  std::is_integral<T>::value &&
  std::is_unsigned<T>::value && sizeof(T) <= sizeof(unsigned int),
unsigned int>::type findFirstSet(T x)
{
  return __builtin_ffs(x);
}

template <class T> inline constexpr
typename std::enable_if<
  (std::is_integral<T>::value && std::is_unsigned<T>::value &&
  sizeof(T) > sizeof(unsigned int) && sizeof(T) <= sizeof(unsigned long)),
unsigned int>::type findFirstSet(T x)
{
  return __builtin_ffsl(x);
}

template <class T> inline constexpr
typename std::enable_if<
  (std::is_integral<T>::value && std::is_unsigned<T>::value &&
  sizeof(T) > sizeof(unsigned long) && sizeof(T) <= sizeof(unsigned long long)),
unsigned int>::type findFirstSet(T x)
{
  return __builtin_ffsll(x);
}

template <class T> inline constexpr
typename std::enable_if<
  (std::is_integral<T>::value && std::is_signed<T>::value &&
   sizeof(T) <= sizeof(unsigned int)), unsigned int>::type
findFirstSet(T x)
{
  return findFirstSet(static_cast<typename std::make_unsigned<T>::type>(x));
}

/**
 * \return last (most significant) bit set
 */
template <class T> inline constexpr
typename std::enable_if<
  (std::is_integral<T>::value && std::is_unsigned<T>::value &&
   sizeof(T) <= sizeof(unsigned int)),
  unsigned int>::type
findLastSet(T x)
{
  return x ? 8 * sizeof(unsigned int) - __builtin_clz(x) : 0;
}

template <class T> inline constexpr
typename std::enable_if<
  (std::is_integral<T>::value && std::is_unsigned<T>::value &&
   sizeof(T) > sizeof(unsigned int) && sizeof(T) <= sizeof(unsigned long)),
  unsigned int>::type
findLastSet(T x)
{
  return x ? 8 * sizeof(unsigned long) - __builtin_clzl(x) : 0;
}


template <class T> inline constexpr
typename std::enable_if<
  (std::is_integral<T>::value && std::is_unsigned<T>::value &&
   sizeof(T) > sizeof(unsigned long) && sizeof(T) <= sizeof(unsigned long long)),
  unsigned int>::type
findLastSet(T x)
{
  return x ? 8 * sizeof(unsigned long long) - __builtin_clzll(x) : 0;
}

/**
 * \return the next power of two >= x
 * based on folly
 */
template <class T> inline constexpr
typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, T>::type
nextPowerofTwo(T v)
{
  return v ? (1 << findLastSet(v-1)) : 1;
}

template <class T> inline constexpr
typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value,
bool>::type
isPowerOfTwo(T x)
{
  return (x != 0) && !(x & (x-1));
}

template <class T> inline
typename std::enable_if<
  (std::is_integral<T>::value && std::is_unsigned<T>::value &&
   sizeof(T) <= sizeof(unsigned int)),
size_t>::type
popcount(T x)
{
  return __builtin_popcount(x);
}

template <class T> inline
typename std::enable_if<
  (std::is_integral<T>::value && std::is_unsigned<T>::value &&
   sizeof(T) > sizeof(unsigned long) && sizeof(T) <= sizeof(unsigned long long)),
size_t>::type
popcount(T x)
{
  return __builtin_popcountll(x);
}


/**
 * \return the number of pyramid levels such that reducing the image size by
 * half at level results in at most min_image_res image
 *
 * \param min_image_dim the smallest dimension of the image min(rows, cols)
 * \param min_image_res desired minimum image resolution
 */
int getNumberOfPyramidLevels(int min_image_dim, int min_image_res = 160*80);

//
// misc
//

/**
 * round up the input argument n to be a multiple of m
 */
int roundUpTo(int n, int m);

/** vsprintf like */
std::string Format(const char* fmt, ...);


/**
 * return the date & time as a string
 */
std::string datetime();

/**
 * \return wall clock in seconds
 */
double GetWallClockInSeconds();

/**
 * \return the current unix timestamp
 */
uint64_t UnixTimestampSeconds();
inline uint64_t getTimestamp() { return UnixTimestampSeconds(); }

/**
 * \return the current unix timestamp as milliseconds
 */
uint64_t UnixTimestampMilliSeconds();

/**
 * sleep for the given milliseconds
 */
void Sleep(int32_t milliseconds);

/**
 * backtrace
 */
std::string GetBackTrace();

}; // bp

#endif // BP_UTILS_UTILS_H
