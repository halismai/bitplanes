#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/homography.h>

using namespace bp;

int main()
{
  cv::Mat image(480, 640, CV_8UC1);
  cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(255));

  AlgorithmParameters params;
  params.num_levels = 1;
  params.max_iterations = 100;
  params.function_tolerance = 1e-6;
  params.parameter_tolerance = 1e-6;
  params.sigma = 1.2;

  BitPlanesTrackerPyramid<Homography> tracker(params);

  return 0;
}



