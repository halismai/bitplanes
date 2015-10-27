#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "bitplanes/core/homography.h"
#include "bitplanes/utils/timer.h"

#include <vector>
#include <iostream>
#include <random>

#include <Eigen/LU>

static std::random_device RandomDevice;
static std::mt19937 RandomGenerator(RandomDevice());

static inline
void MakeCorres(const bp::Matrix33f& H, std::vector<cv::Point2f>& x1,
                std::vector<cv::Point2f>& x2, size_t npts = 1000)
{
  auto normHomog = [=](const Eigen::Matrix<float,3,1>& x)
  {
    return Eigen::Matrix<float,3,1>( x / x[2] );
  };

  x1.resize(npts);
  x2.resize(npts);

  std::normal_distribution<float> noise(0, .01);

  for(size_t i = 0; i < npts; ++i)
  {
    Eigen::Matrix<float, 3, 1> x; x.setRandom();
    x = normHomog(x);
    x1[i] = cv::Point2f(x[0], x[1]);

    x1[i].x += noise(RandomGenerator);
    x1[i].y += noise(RandomGenerator);

    x = normHomog(H * x);
    x2[i] = cv::Point2f(x[0], x[1]);

    x2[i].x += noise(RandomGenerator);
    x2[i].y += noise(RandomGenerator);
  }
}

int main()
{
  bp::Matrix33f H_true = bp::Homography::ParamsToMatrix(
      bp::Homography::ParameterVector::Random());

  std::vector<cv::Point2f> x1, x2;
  MakeCorres(H_true, x1, x2, cv::RANSAC);

  cv::Mat H_est = cv::findHomography(x1, x2);

  std::cout << "GOT:\n" << H_est << "\n" << std::endl;
  std::cout << "TRUE:\n" << H_true/H_true(2,2) << std::endl;

  bp::Matrix33f HH;
  cv::cv2eigen(H_est, HH);
  std::cout << "ERROR: " <<
      ((H_true.inverse() * HH) - bp::Matrix33f::Identity()).norm()
      << std::endl;


  auto t_ms = bp::TimeCode(1000, [&]()
              {
              H_true = bp::Homography::ParamsToMatrix(
                  bp::Homography::ParameterVector::Random());
              std::vector<cv::Point2f> x1, x2;
              MakeCorres(H_true, x1, x2, cv::RANSAC);
              H_est = cv::findHomography(x1, x2, cv::RANSAC, 0.1, cv::noArray(), 5000, 0.995);
              });

  printf("Time: %0.2f ms\n", t_ms);

  return 0;
}
