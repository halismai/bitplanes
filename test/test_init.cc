#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/types.h>

#include <Eigen/LU>

using namespace bp;

int main()
{
  cv::Mat image(480*3, 640*3, CV_8UC1);
  cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(255));

  AlgorithmParameters params;
  params.num_levels = 1;
  params.max_iterations = 100;
  params.function_tolerance = 1e-6;
  params.parameter_tolerance = 1e-6;
  params.sigma = -1.2;

  BitPlanesTrackerPyramid<Homography> tracker(params);
  tracker.setTemplate(image, cv::Rect(300,400,320,240));

  std::cout << tracker.track(image) << std::endl;

  {
    bp::Matrix33f T_true;
    T_true << 1.0, 0.0, 2.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0;
    cv::Mat I0;
    cv::Mat M = (cv::Mat_<float>(2,3) <<
                 T_true(0,0), T_true(0,1), T_true(0,2),
                 T_true(1,0), T_true(1,1), T_true(1,2));
    cv::warpAffine(image, I0, M, cv::Size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar::all(0));

    tracker.setTemplate(I0, cv::Rect(300, 400, 320, 240));
    //cv::imshow("I0", I0); cv::waitKey();

    bp::Matrix33f T_init(bp::Matrix33f::Identity());
    T_init <<
        0.99998,  6.91337e-07,     -2.0057,
        3.528e-05,    1.00003,   -0.0365097,
        3.44926e-08, -3.52903e-08,     0.999988;

    auto result = tracker.track(image, T_init);
    std::cout << "got: " << result << std::endl;

    std::cout <<"ERROR: " <<
        (bp::Homography::MatrixToParams( T_true.inverse() ) -
         bp::Homography::MatrixToParams( result.T ) ).norm() <<
        std::endl;
  }

  return 0;
}



