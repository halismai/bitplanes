#include <opencv2/highgui.hpp>

void buttonCallback(int, void*) { }

int main()
{
  int value1 = 50;
  int value2 = 0;

  cv::namedWindow("main1", CV_WINDOW_NORMAL);
  cv::namedWindow("main2", CV_WINDOW_NORMAL | CV_GUI_NORMAL);

  cv::createTrackbar("bar", "main1", &value1, 255, NULL);

  cv::String nameb1 = "button1";
  cv::String nameb2 = "button2";

  cv::createButton(nameb1, buttonCallback, &nameb1, cv::QT_PUSH_BUTTON);

  return 0;
}
