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

static const std::array<cv::Rect,9> RECTS
{
  cv::Rect(263, 129, 613, 463), // vid1.png
  cv::Rect(314, 205, 511, 392), // vid2.png
  cv::Rect(418, 194, 367, 356), // 3
  cv::Rect(259, 131, 625, 479), // 4
  cv::Rect(314, 206, 516, 386), // 5
  cv::Rect(428, 199, 372, 344), // 6
  cv::Rect(295, 120, 626, 500), // 7
  cv::Rect(349, 71,  493, 494), // 8
  cv::Rect(219, 101, 674, 497)  // 9
};

static const double SCALE = 0.25;

static inline cv::Mat getScaledImage(const cv::Mat& I)
{
  cv::Mat tmp = I;
  if(SCALE < 1.0)
  {
    tmp = I.clone();
    cv::resize(tmp, tmp, cv::Size(), SCALE, SCALE);
  }

  cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);

  return tmp;
}

static inline cv::Rect ScaleRect(cv::Rect r, double s)
{
  return cv::Rect(r.x*s, r.y*s, r.width*s, r.height*s);
}

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
  } else if("v4" == s) {
    ret = RECTS[3];
  } else if("v5" == s) {
    ret = RECTS[4];
  } else if("v6" == s) {
    ret = RECTS[5];
  } else if("v7" == s) {
    ret = RECTS[6];
  } else if("v8" == s) {
    ret = RECTS[7];
  } else if("v9" == s) {
    ret = RECTS[8];
  } else {
    throw std::runtime_error("unknown video name " + s);
  }

  return SCALE > 0 ? ScaleRect(ret, SCALE) : ret;
}

static bp::AlgorithmParameters GetDefaultParams()
{
  bp::AlgorithmParameters params;
  params.num_levels = 2;
  params.max_iterations = 50;
  params.parameter_tolerance = 5e-5;
  params.function_tolerance = 1e-4;
  params.verbose = false;
  params.sigma = 2.0;
  params.subsampling = 4;
  return params;
}

static bool GetFrame(cv::VideoCapture& cap, cv::Mat& image_original,
                        cv::Mat& image_gray)
{
  cap >> image_original;
  if(!image_original.empty())
  {
    if(SCALE > 0)
      cv::resize(image_original, image_gray, cv::Size(), SCALE, SCALE);
    else
      image_gray = image_original;

    cv::cvtColor(image_gray, image_gray, cv::COLOR_BGR2GRAY);
  }

  return !image_original.empty();
}

static inline
void RunBitPlanes(cv::VideoCapture& cap, const cv::Rect& bbox, const char* output_video)
{
  using namespace bp;

  cv::Rect bbox_original;
  bbox_original = SCALE > 0 ? ScaleRect(bbox, 1.0 / SCALE) : bbox;

  std::cout << "template: " << bbox << std::endl;

  BitPlanesTrackerPyramid<Homography> tracker(GetDefaultParams());

  cv::Mat image, image_gray;
  if(!GetFrame(cap, image, image_gray)) {
    std::cerr << "could not read data from video\n";
    return;
  }

  tracker.setTemplate(image_gray, bbox);

  cv::namedWindow("bp");
  cv::imshow("bp", image);
  cv::waitKey(10);

  bp::Matrix33f H( bp::Matrix33f::Identity() );

#if BITPLANES_WITH_PROFILER
  ProfilerStart("/tmp/prof");
#endif

  double total_time = 0.0f;
  int frame = 1;

  if(!GetFrame(cap, image, image_gray)) {
    std::cerr << "could not read data from video\n";
    return;
  }

  char text_buf[64];

  bool save_video = NULL != output_video;
  int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  cv::VideoWriter video_writer;
  if(save_video) {
    video_writer.open(output_video, fourcc, 25, image.size());
    video_writer.set(cv::VIDEOWRITER_PROP_QUALITY, 95);
  }

  bool verbose = true;

  Result result(bp::Matrix33f::Identity());

  while( GetFrame(cap, image, image_gray) && ('q' != cv::waitKey(1)) ) {
    Timer timer;
    result = tracker.track(image_gray, result.T);
    total_time += timer.stop().count() / 1000.0;

    bp::Matrix33f H_scaled = SCALE > 0 ?
        bp::Homography::Scale(result.T, 1.0/SCALE) : result.T;
    DrawTrackingResult(image, image, bbox_original, H_scaled.data());

    snprintf(text_buf, 64, "Frame %05d @ %3.2f Hz", frame, frame/total_time);
    cv::putText(image, text_buf, cv::Point(10,40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0,0,0), 2, cv::LINE_AA);

    if(verbose) {
      fprintf(stdout, "Frame %04d @ %3.2f Hz [%03d iters : %03d ms]\r",
              frame, frame / total_time, result.num_iterations, (int) result.time_ms);
      fflush(stdout);
    }

    cv::imshow("bp", image);

    if(video_writer.isOpened())
      video_writer << image;

    frame += 1;
  }

  if(verbose) fprintf(stdout, "\n");
  Info("Ran at @ %0.2f Hz\n", frame / total_time);


#if BITPLANES_WITH_PROFILER
  ProfilerStop();
#endif

}


int main(int argc, char** argv)
{
  cv::setNumThreads(0);

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


  RunBitPlanes(cap, GetRectFromFilename(video_name), argc > 2 ? argv[2] : NULL);

  return EXIT_SUCCESS;
}

