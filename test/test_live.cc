#include <bitplanes/core/debug.h>
#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/viz.h>

#include <bitplanes/utils/timer.h>
#include <bitplanes/utils/error.h>
#include <bitplanes/utils/config_file.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <cstdlib>
#include <iostream>

using namespace bp;


struct GuiState
{
  cv::Rect bbox;
  cv::Mat I0;
  UniquePointer<BitPlanesTrackerPyramid<Homography>> gTracker;
}; // GuiState

UniquePointer<GuiState> gState;

void trackButtonCallback(int state, void*)
{
  printf("hello %d\n", state);
}

static cv::Rect SelectTemplate(cv::VideoCapture& cap);
static void StartTracking(const cv::Mat& I0, const cv::Rect& bbox);

int main(int argc, char** argv)
{
  /*
  if(argc < 1) {
    static const char* USAGE = "%s <config_file>\n";
    printf(USAGE, argv[0]);
    return EXIT_FAILURE;
  }
  std::string config_file(argv[1]);
  */

  cv::VideoCapture cap(0);
  if(!cap.isOpened()) {
    Fatal("failed to open camera\n");
  }

  cv::namedWindow("bitplanes");
  cv::namedWindow("main", CV_WINDOW_NORMAL | CV_GUI_NORMAL);

  cv::String btnName = "trackButton";
  cv::createButton("track", trackButtonCallback, &btnName, cv::QT_PUSH_BUTTON);

  auto bbox = SelectTemplate(cap);

  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}



cv::Rect SelectTemplate(cv::VideoCapture& cap)
{

  int k = 0;
  while(k != 'q') {
    cv::Mat image;
    cap >> image;
    cv::imshow("bitplanes", image);
    k = 0xff & cv::waitKey(10);
  }

  return cv::Rect();
}

void StartTracking(const cv::Mat& I0, const cv::Rect& bbox)
{
}
