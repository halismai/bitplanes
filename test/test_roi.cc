#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(10, 30, 200, 150);

  cv::imshow("image", I(roi)); cv::waitKey();

  return 0;
}

