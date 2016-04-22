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


#include "bitplanes/utils/memory.h"

#include <cstdlib>
#include <cassert>
#include <stdexcept>
#include <cstring>

namespace bp {

static inline void throw_bad_alloc()
{
#if 1 || TT_USE_EXCEPTIONS
  throw std::bad_alloc();
#else
  new int[static_cast<size_t>(-1)];
#endif
}

static inline bool is_non_negative_and_power_of_2(int n)
{
  return n && !(n & (n - 1));
}

static inline void* _aligned_malloc(size_t nbytes, int alignment)
{
  assert( is_non_negative_and_power_of_2(alignment) );

  void* p = std::malloc(nbytes + alignment);
  if(!p)
    return nullptr;

  void* aligned = reinterpret_cast<void*>((reinterpret_cast<size_t>(p) &
                                          (~(alignment-1))) + alignment);
  *(reinterpret_cast<void**>(aligned) - 1) = p;
  return aligned;
}

static inline void _aligned_free(void *ptr)
{
  if(ptr)
    std::free(*(reinterpret_cast<void**>(ptr)-1));
}

//
// based on Eigen soruce code
//   src/Core/util/Memory.h
//
// Mozilla Public License v. 2.0.
//
static inline void* _aligned_realloc(void *ptr, size_t nbytes, int alignment)
{
  assert( is_non_negative_and_power_of_2(alignment) );

  if(ptr == nullptr)
    return _aligned_malloc(nbytes, alignment);

  void* original = *(reinterpret_cast<void**>(ptr) - 1);
  auto offset = static_cast<char*>(ptr) - static_cast<char*>(original);
  original = std::realloc(original, nbytes + alignment);
  if(original == nullptr)
    return nullptr;

  void* aligned = reinterpret_cast<void*>((reinterpret_cast<size_t>(original) &
      (~(alignment-1))) + alignment);
  void* prev_aligned = static_cast<char*>(original) + offset;
  if(aligned != prev_aligned)
    std::memmove(aligned, prev_aligned, nbytes);
  *(reinterpret_cast<void**>(aligned) - 1) = original;

  return aligned;
}


void* aligned_malloc(size_t nbytes, int alignment)
{
  return _aligned_malloc(nbytes, alignment);
}

void* aligned_realloc(void* ptr, size_t nbytes, int alignment)
{
  void* ret = _aligned_realloc(ptr, nbytes, alignment);

  if(!ret && nbytes)
    throw_bad_alloc();

  return ret;
}

void aligned_free(void* ptr)
{
  _aligned_free(ptr);
}

} // bp

