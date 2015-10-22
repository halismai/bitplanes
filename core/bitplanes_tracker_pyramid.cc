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

#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/debug.h>
#include <bitplanes/utils/error.h>

#include <opencv2/imgproc.hpp>

#include <iostream>

namespace bp {

static inline
AlgorithmParameters ReduceAlgorithmParameters(AlgorithmParameters p)
{
  p.max_iterations = 25;
  p.parameter_tolerance *= 10;
  p.function_tolerance  *= 10;
  p.sigma = 0.8;

  return p;
}

static inline
std::vector<AlgorithmParameters>
MakeAlgorithmParametersPyramid(AlgorithmParameters p)
{
  THROW_ERROR_IF(p.num_levels < 1, "auto pyramid levels not implemented");

  std::vector<AlgorithmParameters> ret(p.num_levels);
  ret[0] = p;

  for(size_t i = 1; i < ret.size(); ++i)
    ret[i] = ReduceAlgorithmParameters(ret[0]);

  return ret;
}


template <class M>
void BitPlanesTrackerPyramid<M>::setTemplate(const cv::Mat& I, const cv::Rect& bbox)
{
  auto alg_params = MakeAlgorithmParametersPyramid(_alg_params);

  cv::Mat I0;
  I.copyTo(I0);

  _pyramid.clear();
  for(size_t i = 0; i < alg_params.size(); ++i)
    _pyramid.push_back( Tracker(alg_params[i]) );

  _pyramid[0].setTemplate(I0, bbox);

  cv::Rect bbox_copy(bbox);
  for(size_t i = 1; i < _pyramid.size(); ++i)
  {
    cv::pyrDown(I0, I0);
    bbox_copy.x /= 2; bbox_copy.y /= 2;
    bbox_copy.width /= 2; bbox_copy.height /= 2;
    _pyramid[i].setTemplate(I0, bbox_copy);
  }

  _T_init.setIdentity();
}

template <class M>
Result BitPlanesTrackerPyramid<M>::track(const cv::Mat& I, const Transform& T_init)
{
  float s = 1.0f / (1 << (_pyramid.size()-1));
  Result ret( MotionModelType::Scale(T_init, s) );

  std::vector<cv::Mat> I_pyr(_pyramid.size());
  I.copyTo(I_pyr[0]);
  for(size_t i = 1; i < I_pyr.size(); ++i)
    cv::pyrDown(I_pyr[i-1], I_pyr[i]);

  for(int i = (int) _pyramid.size() - 1; i >= 0; --i)
  {
    ret = _pyramid[i].track(I_pyr[i], ret.T);
    if(i != 0) ret.T = MotionModelType::Scale(ret.T, 2.0);
  }

  _T_init = ret.T;
  return ret;
}

template class BitPlanesTrackerPyramid<Homography>;

}; // bp
