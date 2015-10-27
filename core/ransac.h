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

#ifndef BITPLANES_CORE_RANSAC_H
#define BITPLANES_CORE_RANSAC_H

#include "bitplanes/core/ransac_model.h"
#include <array>
#include <cmath>
#include <limits>
#include <random>

namespace bp {

template <class Model>
class Ransac
{
 public:
  static constexpr int MinSampleSize = RansacModel<Model>::MinSampleSize;
  typedef typename RansacModel<Model>::Result Result;
  typedef typename RansacModel<Model>::Indices Indices;

  Ransac(RansacModel<Model>& model, float threshold, float inlier_prob=0.99)
      : _model(model), _inlier_threshold(threshold), _p(inlier_prob) {}

  inline Result fit(int max_iters = -1);

  inline const Indices& inliers() const { return _inliers; }

 protected:
  RansacModel<Model>& _model;
  Indices _inliers;
  float _inlier_threshold;

  /* desired probability of selecting at least one outlier-free sample */
  float _p=0.99;

  bool _verbose = true;
}; // Ransac

template <class Array> static inline
bool isUnique(const Array& a, typename Array::value_type v)
{
  for(const auto& a_i : a)
    if(a_i == v)
      return false;
  return true;
}

template <int N, typename T = uint32_t> static inline
std::array<T, N> RandomSample(uint32_t n, std::mt19937& gen)
{
  std::uniform_int_distribution<T> dist(0, n-1);
  std::array<T, N> ret;
  ret.fill(n);

  int i = 0;
  while( i < N )
  {
    auto s = (uint32_t) dist(gen);
    if(isUnique(ret, s)) {
      ret[i++] = s;
    }
  }

  return ret;
}

template <int S> static inline
size_t UpdateN(float fracinliers, float p)
{
  constexpr auto eps = std::numeric_limits<float>::epsilon();
  auto pNoOutliers = 1.0f - std::pow(fracinliers, (float) S);
  pNoOutliers = std::min(1.0f-eps, std::max(eps, pNoOutliers));
  return static_cast<size_t>( std::log(1.0f-p) / std::log(pNoOutliers) );
}

template <class M> inline
auto Ransac<M>::fit(int max_iters) -> Result
{
  // implementation based on Peter Kovesi's matlab's ransac.m
  size_t maxTrials = (size_t) max_iters;
  auto N = maxTrials;

  std::random_device rd;
  std::mt19937 gen(rd());

  int best_num_inliers = -1;
  Indices best_inds;
  Result result, best_result;
  size_t trialcount = 0;
  while(N > trialcount && trialcount++ < maxTrials)
  {
    result = _model.run(RandomSample<MinSampleSize>(_model.size(), gen));
    _inliers = _model.findInliers(result, _inlier_threshold);

    if(_verbose)
      printf("%zu/%zu N:%zu [%zu inliers]\n", trialcount, maxTrials, N, _inliers.size());

    if((int) _inliers.size() > best_num_inliers)
    {
      best_num_inliers = (int) _inliers.size();
      best_inds.swap(_inliers);
      best_result = result;

      float fracinliers = best_num_inliers / (float) _model.size();
      N = UpdateN<MinSampleSize>(fracinliers, _p);
    }
  }

  result = _model.fitFinal(best_result, best_inds);
  _inliers = _model.findInliers(result, _inlier_threshold);
  return result;
}

}; // bp

#endif // BITPLANES_CORE_RANSAC_H

