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

#include <Eigen/Core>

#include <iosfwd>
#include <string>
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

typedef typename EigenStdVector<Vector3f>::type       PointVector;
typedef typename EigenStdVector<Vector_<float>>::type ResidualsVector;

enum class OptimizerStatus
{
  NotStarted,             //< optimization has not started
  MaxIterations,          //< max iterations reached
  FirstOrderOptimality,   //< norm of the gradient is small
  SmallRelativeReduction, //< relative reduction in objective is small
  SmallAbsError,          //< absolute error value is small
  SmallParameterUpdate,   //< current delta parameters is small
  SmallAbsParameters,     //< absolute parameter step is small
}; // OptimizerStatus

/**
 * converts OptimizerStatus enum to a string
 */
std::string ToString(OptimizerStatus);

/**
 * The trackers results, estimated motion model and other info
 */
struct Result
{
  inline Result(const Matrix33f& tform = Matrix33f::Identity())
      : T(tform) {}

  /** status of the optimizer */
  OptimizerStatus status = OptimizerStatus::NotStarted;

  /** number of iterations */
  int num_iterations = -1;

  /** final sum of squared errors */
  float final_ssd_error = -1.0f;

  /** first order optimiality, Inf norm of the gradient */
  float first_order_optimality = -1.0f;

  /** time in milliseconds if timing is enabled */
  float time_ms = -1.0;

  /** estimated transform */
  Matrix33f T = Matrix33f::Identity();

  bool successfull = true;

  friend std::ostream& operator<<(std::ostream&, const Result&);
}; // Result


/**
 * Type of the motion to estimate
 */
enum class MotionType
{
  Translation,
  Affine,
  Homography
}; // MotionType

std::string ToString(MotionType);

}; // bp

#endif // BP_CORE_TYPES_H

