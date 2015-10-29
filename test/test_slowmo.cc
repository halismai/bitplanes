#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <string>
#include <array>

#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/viz.h>

#include <bitplanes/utils/fs.h>
#include <bitplanes/utils/timer.h>

static const std::array<cv::Rect,3> RECTS
{
  cv::Rect(263, 129, 613, 463), // vid1.png
  cv::Rect(314, 205, 511, 392), // vid2.png
  cv::Rect(418, 194, 367, 356)
};

static inline cv::Rect GetRectFromFilename(std::string name)
{
  auto s = bp::fs::remove_extension(bp::fs::getBasename(name));
  if("v1" == s) {
    return RECTS[0];
  } else if("v2" == s) {
    return RECTS[1];
  } else if("v3" == s) {
    return RECTS[2];
  } else {
    throw std::runtime_error("unknown video name " + s);
  }
}

static inline void RunBitPlanes(cv::VideoCapture& cap, const cv::Rect& bbox)
{
  using namespace bp;

  std::cout << bbox << std::endl;

  /*
   v2
   */
  bp::AlgorithmParameters params;
  params.num_levels = 4;
  params.max_iterations = 50;
  params.parameter_tolerance = 1e-5;
  params.function_tolerance = 1e-5;
  params.verbose = false;
  params.sigma = 1.2;

  /*
  bp::AlgorithmParameters params;
  params.num_levels = 3;
  params.max_iterations = 2000;
  params.parameter_tolerance = 1e-6;
  params.function_tolerance = 1e-6;
  params.verbose = false;
  params.sigma = 0.5;
  */

  BitPlanesTrackerPyramid<Homography> tracker(params);

  cv::Mat image, image_gray;
  cap >> image;
  if(image.empty())
  {
    std::cerr << "could not read data from video\n";
    return;
  }

  cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  tracker.setTemplate(image_gray, bbox);

  cv::namedWindow("bp");
  bp::Matrix33f H( bp::Matrix33f::Identity() );

  cap >> image;
  bool stop = false;
  while( !image.empty() && !stop )
  {
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    auto result = tracker.track(image_gray, H);
    H = result.T / result.T(2,2);
    DrawTrackingResult(image, image, bbox, H.data());

    std::cout << result << "\n\n";

    cv::imshow("bp", image);
    stop = ('q' == (0xff & cv::waitKey(1)));
    cap >> image;
  }

}


int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cerr << "usage: " << argv[0] << " video_name" << std::endl;
    return EXIT_FAILURE;
  }

  std::string video_name(argv[1]);
  cv::VideoCapture cap(video_name);
  if(!cap.isOpened())
  {
    std::cerr << "Failed to open " << video_name << std::endl;
    return EXIT_FAILURE;
  }

  RunBitPlanes(cap, GetRectFromFilename(video_name));

  return EXIT_SUCCESS;
}
