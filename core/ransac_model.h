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

#ifndef BITPLANES_CORE_RANSAC_MODEL_H
#define BITPLANES_CORE_RANSAC_MODEL_H

#include "bitplanes/core/types.h"
#include <vector>
#include <array>
#include <iosfwd>

namespace bp {

template <class T> struct ransac_model_traits;

template <class T>
class RansacModel
{
 public:
  typedef typename ransac_model_traits<T>::Result   Result;

  static constexpr int MinSampleSize = ransac_model_traits<T>::MinSampleSize;
  typedef std::vector<uint32_t> Indices;
  typedef std::array<uint32_t, MinSampleSize> SampleIndices;

 public:

  /**
   * \return number of points
   */
  inline size_t size() const { return derived()->size(); }

  /**
   * Estimate a model given indices into the data
   *
   * \param inds sampling indices
   * \return estimated result
   */
  inline Result run(const SampleIndices& inds) const
  {
    return derived()->run(inds);
  }

  /**
   * Fit a least sqaures on the inliers, or refine result based on initialization
   *
   * \return refined/least squares result
   */
  inline Result fitFinal(const Result& result, const Indices& inds) const
  {
    return derived()->fitFinal(result, inds);
  }

  inline Indices findInliers(const Result& R, float thresh) const
  {
    return derived()->findInliers(R, thresh);
  }

 protected:
  inline const T* derived() const { return static_cast<const T*>(this); }
  inline       T* derived()       { return static_cast<T*>(this); }
}; // Ransac


class RansacHomography;

template <> struct ransac_model_traits<RansacHomography>
{
  typedef Eigen::Matrix<float,3,3> Result;
  static constexpr int MinSampleSize = 4;
}; // ransac_model_traits

template <typename T = float>
struct Correspondence
{
  typedef Eigen::Matrix<float,3,1> Point;

  inline Correspondence() {}

  inline Correspondence(const Point& p1, const Point& p2)
      : x1(p1), x2(p2) {}

  Point x1;
  Point x2;
  bool _is_inlier;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; // Correspndence

class RansacHomography : public RansacModel<RansacHomography>
{
 public:
  typedef Correspondence<float> CorrespondenceType;
  typedef typename EigenStdVector<CorrespondenceType>::type CorrespondencesType;

  typedef RansacModel<RansacHomography> Base;
  typedef typename Base::Indices Indices;
  typedef typename Base::SampleIndices SampleIndices;

 public:
  RansacHomography(const CorrespondencesType& c) : _corrs(c) {}

  inline size_t size() const { return _corrs.size(); }

  Result run(const SampleIndices& inds) const;
  Result fitFinal(const Result&, const Indices& inds) const;

  Indices findInliers(const Result&, float thresh) const;

 protected:
  const CorrespondencesType& _corrs;
}; // RansacHomography

}; // bp

#endif // BITPLANES_CORE_RANSAC_MODEL_H

