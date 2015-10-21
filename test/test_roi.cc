#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <opencv2/core/eigen.hpp>

#include <iostream>

static inline
void imgradient(const cv::Mat& I, cv::Mat& Ix, cv::Mat& Iy)
{
  const cv::Mat k = (cv::Mat_<float>(1,3) << -1.0, 0.0, 1.0)/2.0;
  cv::filter2D(I, Ix, CV_32F, k);
  cv::filter2D(I, Iy, CV_32F, k.t());
}

static inline
void warpImage(const cv::Mat& src, const cv::Matx<float,2,3>& M, cv::Mat& dst)
{
  cv::Mat_<float> map_x(src.size());
  cv::Mat_<float> map_y(src.size());

  for(int y = 0; y < src.rows; ++y)
  {
    for(int x = 0; x < src.cols; ++x)
    {
      map_x(y,x) = M(0,0)*x + M(0,1)*y + M(0,2);
      map_y(y,x) = M(1,0)*x + M(1,1)*y + M(1,2);
    }
  }

  cv::remap(src, dst, map_x, map_y, cv::INTER_CUBIC, cv::BORDER_CONSTANT);
}

static inline
Eigen::Matrix<double,2,1> LK(const cv::Mat& I0_, const cv::Mat& I1_)
{
  cv::Mat Ix, Iy;

  cv::Mat I0, I1, Iw, It;
  I0_.convertTo(I0, CV_32F);
  I1_.convertTo(I1, CV_32F);

  Eigen::Matrix<double,2,1> p(0.0f, 0.0f);

  cv::Matx<float,2,3> M;
  M << 1.0, 0.0, p[0],
       0.0, 1.0, p[1];

  Eigen::Matrix<double,2,2> A;
  Eigen::Matrix<double,2,1> b;

  //cv::GaussianBlur(I0, I0, cv::Size(3,3), 0.0, 0.0);
  //cv::GaussianBlur(I1, I1, cv::Size(3,3), 0.0, 0.0);

  int max_it = 100;
  for(int i = 0; i < max_it; ++i)
  {
    M(0,2) = p[0];
    M(1,2) = p[1];

    warpImage(I1, M, Iw);
    imgradient(Iw, Ix, Iy);

    A(0,0) = cv::sum(cv::sum(Ix.mul(Ix)))[0];
    A(0,1) = cv::sum(cv::sum(Ix.mul(Iy)))[0];
    A(1,1) = cv::sum(cv::sum(Iy.mul(Iy)))[0];
    A(1,0) = A(0,1);

    It = Iw - I0;

    cv::imshow("E", It); cv::waitKey(10);

    b[0] = cv::sum(cv::sum(It.mul(Ix)))[0];
    b[1] = cv::sum(cv::sum(It.mul(Iy)))[0];

    Eigen::Matrix<double,2,1> dp = A.ldlt().solve(b);

    double p_norm = dp.norm();

    printf("%d %e\n", i, p_norm);
    if(p_norm < 1e-6)
      break;

    p = p - dp;
  }

  return p;
}

static inline Eigen::Vector2d IC(const cv::Mat& I0_, const cv::Mat& I1_)
{
  cv::Mat I0, I1;
  I0_.convertTo(I0, CV_32F);
  I1_.convertTo(I1, CV_32F);

  // pre-compute the Hessian
  cv::Mat Ix, Iy;
  Eigen::Matrix<double,2,2> H;
  Eigen::Matrix<double,2,1> rhs;
  {
    imgradient(I0, Ix, Iy);

    H(0,0) = cv::sum(cv::sum(Ix.mul(Ix)))[0];
    H(0,1) = cv::sum(cv::sum(Ix.mul(Iy)))[0];
    H(1,1) = cv::sum(cv::sum(Iy.mul(Iy)))[0];
    H(1,0) = H(0,1);
  }

  Eigen::Vector2d p(0.0, 0.0);
  cv::Matx<float,2,3> A;
  A(0,0) = 1.0; A(0,1) = 0.0;
  A(1,0) = 0.0; A(1,1) = 1.0;

  cv::Mat Iw, It;
  for(int i = 0; i < 100; ++i)
  {
    A(0,2) = p[0]; A(1,2) = p[1];
    warpImage(I1, A, Iw);

    It = Iw - I0;

    cv::imshow("E", It); cv::waitKey(10);

    rhs[0] = cv::sum(cv::sum(It.mul(Ix)))[0];
    rhs[1] = cv::sum(cv::sum(It.mul(Iy)))[0];

    Eigen::Vector2d dp = -H.ldlt().solve(rhs);
    double p_norm = dp.norm();
    printf("%d %e\n", i, p_norm);

    if(p_norm < 1e-6)
      break;

    p += dp;
  }

  return p;
}


int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(10, 30, 200, 150);

  //auto t = LK(I, I);
  //std::cout << t << std::endl;
  std::cout << LK(I, I) << std::endl;

  cv::Matx<float,2,3> H;
  H <<
      1.0, 0.0, 0.5,
      0.0, 1.0, 0.0;

  cv::Mat Iw;
  warpImage(I, H, Iw);

  auto t = IC(I, Iw);
  std::cout << t.transpose() << std::endl;

  cv::imshow("image", Iw);
  cv::waitKey();

  return 0;
}

