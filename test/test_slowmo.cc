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

#if BITPLANES_WITH_PROFILER
#include <gperftools/profiler.h>
#endif

static const std::array<cv::Rect,3> RECTS
{
  cv::Rect(263, 129, 613, 463), // vid1.png
  cv::Rect(314, 205, 511, 392), // vid2.png
  cv::Rect(418, 194, 367, 356)
};

static const double SCALE = 0.25;

static inline cv::Rect GetRectFromFilename(std::string name)
{
  cv::Rect ret;
  auto s = bp::fs::remove_extension(bp::fs::getBasename(name));
  if("v1" == s) {
    ret = RECTS[0];
  } else if("v2" == s) {
    ret = RECTS[1];
  } else if("v3" == s) {
    ret = RECTS[2];
  } else {
    throw std::runtime_error("unknown video name " + s);
  }

  if(SCALE > 0)
  {
    ret.x *= SCALE;
    ret.y *= SCALE;
    ret.width *= SCALE;
    ret.height *= SCALE;
  }

  std::cout << ret << std::endl;
  return ret;
}

static inline void RunBitPlanes(cv::VideoCapture& cap, const cv::Rect& bbox)
{
  using namespace bp;

  std::cout << bbox << std::endl;

  /*
   v2
   */
  bp::AlgorithmParameters params;
  params.num_levels = 2;
  params.max_iterations = 50;
  params.parameter_tolerance = 5e-5;
  params.function_tolerance = 1e-4;
  params.verbose = false;
  params.sigma = 1.0;
  params.subsampling = 1;

  BitPlanesTrackerPyramid<Homography> tracker(params);

  cv::Mat image, image_gray;
  cap >> image;
  if(image.empty())
  {
    std::cerr << "could not read data from video\n";
    return;
  }

  if(SCALE > 0)
    cv::resize(image, image, cv::Size(), SCALE, SCALE);

  cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  tracker.setTemplate(image_gray, bbox);

  cv::namedWindow("bp");
  bp::Matrix33f H( bp::Matrix33f::Identity() );

#if BITPLANES_WITH_PROFILER
  ProfilerStart("/tmp/prof");
#endif

  double total_time = 0.0f;
  int frame = 1;
  cap >> image;

  bool stop = false;
  char text_buf[64];

  bool save_video = false;
  int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  cv::VideoWriter video_writer;
  if(save_video) {
    video_writer.open("results.avi", fourcc, 25, image.size());
    video_writer.set(cv::VIDEOWRITER_PROP_QUALITY, 90);
  }

  while( !image.empty() && !stop )
  {
    if(SCALE > 0)
      cv::resize(image, image, cv::Size(), SCALE, SCALE);

    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

    Timer timer;
    auto result = tracker.track(image_gray, H);
    total_time += timer.stop().count() / 1000.0;

    H = result.T;
    DrawTrackingResult(image, image, bbox, H.data());

    snprintf(text_buf, 64, "Frame %05d @ %3.2f Hz", frame, frame/total_time);
    cv::putText(image, text_buf, cv::Point(10,40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0,0,0), 2, cv::LINE_AA);

    fprintf(stdout, "Frame %04d @ %3.2f Hz [%03d iters : %03d ms]\r",
            frame, frame / total_time, result.num_iterations, (int) result.time_ms);
    fflush(stdout);

    cv::imshow("bp", image);

    if(video_writer.isOpened())
      video_writer << image;

    stop = ('q' == (0xff & cv::waitKey(1)));
    cap >> image;

    frame += 1;

    /*if(frame > 200)
      break;
      */
  }

  printf("\nRan at @ %0.2f Hz\n", frame / total_time);

#if BITPLANES_WITH_PROFILER
  ProfilerStop();
#endif

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

