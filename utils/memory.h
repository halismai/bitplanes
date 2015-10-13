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

#ifndef BITPLANES_UTILS_MEMORY_H
#define BITPLANES_UTILS_MEMORY_H

#include <memory>
#include <stdexcept>

#include "bitplanes/core/internal/intrin.h"

namespace bp {

static constexpr int TT_DEFAULT_ALIGNMENT = BITPLANES_DEFAULT_ALIGNMENT;

template <typename T> using SharedPointer = std::shared_ptr<T>;

template <typename T> using UniquePointer = std::unique_ptr<T>;

template <class T, class ... Args> inline
UniquePointer<T> make_unique(Args&& ... args)
{
  return UniquePointer<T>(new T(std::forward<Args>(args)...));
}

void* aligned_malloc(size_t nbytes, int alignment = TT_DEFAULT_ALIGNMENT);
void* aligned_realloc(void* ptr, size_t nbytes, int alignment = TT_DEFAULT_ALIGNMENT);
void aligned_free(void*);

template <class T, int Alignment = TT_DEFAULT_ALIGNMENT>
class AlignedAllocator;

template <int Alignment>
class AlignedAllocator<void, Alignment>
{
 public:
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template <class U> struct rebind { typedef AlignedAllocator<U,Alignment> other; };
}; // AlignedAllocator

template <class T, int Align>
class AlignedAllocator
{
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef std::true_type propagate_on_container_move_assignment;

  template <class U>
  struct rebind { typedef AlignedAllocator<U, Align> other; };

 public:
  inline AlignedAllocator() noexcept {}

  template <class U> inline
  AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

  inline size_type max_size() const noexcept {
    return (size_type(~0) - size_type(Align)) / sizeof(T);
  }

  inline pointer address(reference x) const noexcept {
    return std::addressof(x);
  }

  inline const_pointer address(const_reference x) const noexcept {
    return std::addressof(x);
  }

  inline pointer allocate(size_type n, typename AlignedAllocator<void,Align>::const_pointer=0) {
    auto alignment = static_cast<size_type>(Align);
    void* p = aligned_malloc(sizeof(T)*n, alignment);
    if(nullptr == p)
      throw std::bad_alloc();
    return reinterpret_cast<pointer>(p);
  }

  inline void deallocate(pointer p, size_type) noexcept {
    aligned_free(p);
  }

  template <class U, class ... Args> inline
  void construct(U* p, Args&& ... args) {
    ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  inline void destroy(pointer p) { p->~T(); }

}; // AlignedAllocator

template <typename T, int Align>
class AlignedAllocator<const T, Align>
{
 public:
  typedef T value_type;
  typedef const T* pointer;
  typedef const T* const_pointer;
  typedef const T& reference;
  typedef const T& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef std::true_type propagate_on_container_move_assignment;

  template <class U>
  struct rebind { typedef AlignedAllocator<U, Align> other; };

 public:
  inline AlignedAllocator() noexcept {}

  template <class U> inline
  AlignedAllocator(const AlignedAllocator<U,Align>&) noexcept {}

  inline size_type max_size() const noexcept {
    return (size_type(~0) - size_type(Align)) / sizeof(T);
  }

  inline const_pointer address(const_reference x) const noexcept {
    return std::addressof(x);
  }

  inline pointer allocate(size_type n, typename AlignedAllocator<void,Align>::const_pointer=0) {
    const size_type alignment = static_cast<size_type>(Align);
    void* p = aligned_malloc(sizeof(T)*n, alignment);
    if(nullptr == p)
      throw std::bad_alloc();
    return reinterpret_cast<pointer>(p);
  }

  inline void deallocate(pointer p, size_type) noexcept {
    aligned_free(p);
  }

  template <class U, class ... Args> inline
  void construct(U* p, Args&& ... args) {
    ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  inline void destroy(pointer p) { p->~T(); }
}; // AlignedAllocator

template <typename T, int TAlign, typename U, int UAlign> inline
bool operator==(const AlignedAllocator<T, TAlign>&,
                const AlignedAllocator<U, UAlign>&) { return TAlign == UAlign; }

template <typename T, int TAlign, typename U, int UAlign> inline
bool operator!=(const AlignedAllocator<T, TAlign>&,
                const AlignedAllocator<U, UAlign>&) { return TAlign != UAlign; }

}; // bp

#endif // BITPLANES_UTILS_MEMORY_H

