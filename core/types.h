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

#ifndef BP_CORE_TYPES_H
#define BP_CORE_TYPES_H

#include <Eigen/StdVector>
#include <Eigen/Core>

#include <vector>

namespace bp {

using Eigen::Dynamic;

template <typename T>
using Matrix_ = Eigen::Matrix<T, Dynamic, Dynamic>;

template <typename T>
using Vector_ = Eigen::Matrix<T, Dynamic, 1>;

template <typename T>
using ColVector_ = Vector_<T>;

template <typename T>
using RowVector_ = Eigen::Matrix<T, 1, Dynamic>;

typedef Eigen::Matrix2f Matrix22f;
typedef Eigen::Matrix3f Matrix33f;
typedef Eigen::Matrix4f Matrix44f;
typedef Eigen::Matrix<float, 3, 4> Matrix34f;
typedef Eigen::Matrix<float, 6, 6> Matrix66f;
typedef Eigen::Matrix<float, 8, 8> Matrix88f;

typedef Eigen::Vector2f Vector2f;
typedef Eigen::Vector3f Vector3f;
typedef Eigen::Vector4f Vector4f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 8, 1> Vector8f;

template <class M>
struct EigenStdVector
{
  typedef Eigen::aligned_allocator<M> allocator;
  typedef std::vector<M, allocator>   type;
}; // EigenStdVector

}; // bp

#endif // BP_CORE_TYPES_H

