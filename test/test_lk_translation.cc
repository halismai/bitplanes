#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <opencv2/core/eigen.hpp>

#include <iostream>

class ImageRegistration
{
 public:
  typedef Eigen::Matrix<double,2,2> Hessian;
  typedef Eigen::Matrix<double,2,1> Gradient;
  typedef Eigen::Matrix<double,2,1> Vector2d;

  typedef Eigen::MatrixXf MatrixXf;
  typedef Eigen::Map<MatrixXf, Eigen::Aligned> Map;

 public:
  ImageRegistration() = default;

  /**
   * \param I0 the template
   * \param I1 the input
   * \return motion
   */
  Vector2d lucasKanade(const cv::Mat& I0, const cv::Mat& I1);
  Vector2d lucasKanade(const cv::Mat& I0, const cv::Rect&, const cv::Mat& I1);

  void imshift(const cv::Mat&, cv::Mat&, const Vector2d&);
  void imshift(const cv::Mat&, const cv::Rect&, cv::Mat&, const Vector2d&);

 protected:
  cv::Mat _interp_maps[2];
  cv::Mat _I0, _I1, _Iw;
  cv::Mat _Ix, _Iy, _It;

  void imgradient(const cv::Mat&);

  int _max_iters = 100;
  int _interp = cv::INTER_CUBIC;
  double _p_tol = 1e-6;
}; // ImageRegistration

auto ImageRegistration::lucasKanade(const cv::Mat& I0, const cv::Mat& I1)
  -> Vector2d
{
  I0.convertTo(_I0, CV_32F);
  I1.convertTo(_I1, CV_32F);

  _It.create(_I0.size(), CV_32F);

  Hessian H;
  Gradient rhs;
  Vector2d p(0.0, 0.0), dp;
  for(int i = 0; i < _max_iters; ++i)
  {
    imshift(_I1, _Iw, p);
    imgradient(_Iw);
    _It = _Iw - _I0;

    H(0,0) = cv::sum(cv::sum(_Ix.mul(_Ix)))[0];
    H(0,1) = cv::sum(cv::sum(_Ix.mul(_Iy)))[0];
    H(1,1) = cv::sum(cv::sum(_Iy.mul(_Iy)))[0];
    H(1,0) = H(0,1);
    rhs[0] = cv::sum(cv::sum(_It.mul(_Ix)))[0];
    rhs[1] = cv::sum(cv::sum(_It.mul(_Iy)))[0];

    dp = -H.ldlt().solve(rhs);
    p += dp;

    auto dp_norm = dp.norm();
    printf("%d ||dp||=%e ||G||=%e\n", i, dp_norm, rhs.lpNorm<Eigen::Infinity>());

    if(dp_norm < _p_tol)
      break;
  }

  return p;
}

auto ImageRegistration::lucasKanade(const cv::Mat& I0, const cv::Rect& roi,
                                    const cv::Mat& I1) -> Vector2d
{
  I0(roi).convertTo(_I0, CV_32F);
  I1.convertTo(_I1, CV_32F);

  _It.create(_I0.size(), CV_32F);

  Hessian H;
  Gradient rhs;
  Vector2d p(0.0, 0.0), dp;
  for(int i = 0; i < _max_iters; ++i)
  {
    imshift(_I1, roi, _Iw, p);
    imgradient(_Iw);
    _It = _Iw - _I0;

    H(0,0) = cv::sum(cv::sum(_Ix.mul(_Ix)))[0];
    H(0,1) = cv::sum(cv::sum(_Ix.mul(_Iy)))[0];
    H(1,1) = cv::sum(cv::sum(_Iy.mul(_Iy)))[0];
    H(1,0) = H(0,1);
    rhs[0] = cv::sum(cv::sum(_It.mul(_Ix)))[0];
    rhs[1] = cv::sum(cv::sum(_It.mul(_Iy)))[0];

    dp = -H.ldlt().solve(rhs);
    p += dp;

    auto dp_norm = dp.norm();
    printf("%d ||dp||=%e ||G||=%e\n", i, dp_norm, rhs.lpNorm<Eigen::Infinity>());

    if(dp_norm < _p_tol)
      break;
  }

  return p;

}

void ImageRegistration::imgradient(const cv::Mat& I)
{
  cv::Mat k = (cv::Mat_<float>(1,3) << -1.0, 0.0, 1.0) / 2.0;
  cv::filter2D(I, _Ix, -1, k);
  cv::filter2D(I, _Iy, -1, k.t());
}

void ImageRegistration::imshift(const cv::Mat& src, cv::Mat& dst, const Vector2d& t)
{
  _interp_maps[0].create(src.size(), CV_32F);
  _interp_maps[1].create(src.size(), CV_32F);

  cv::Mat_<float>& x_map = (cv::Mat_<float>&) _interp_maps[0];
  cv::Mat_<float>& y_map = (cv::Mat_<float>&) _interp_maps[1];

  for(int y = 0; y < src.rows; ++y)
    for(int x = 0; x < src.cols; ++x)
    {
      x_map(y,x) = x + t[0];
      y_map(y,x) = y + t[1];
    }

  cv::remap(src, dst, _interp_maps[0], _interp_maps[1], _interp);
}

void ImageRegistration::imshift(const cv::Mat& src, const cv::Rect& roi,
                                cv::Mat& dst, const Vector2d& t)
{
  _interp_maps[0].create(roi.size(), CV_32F);
  _interp_maps[1].create(roi.size(), CV_32F);

  cv::Mat_<float>& x_map = (cv::Mat_<float>&) _interp_maps[0];
  cv::Mat_<float>& y_map = (cv::Mat_<float>&) _interp_maps[1];

  for(int y = 0; y < roi.height; ++y)
    for(int x = 0; x < roi.width; ++x)
    {
      x_map(y,x) = x + t[0] + roi.x;
      y_map(y,x) = y + t[1] + roi.y;
    }

  cv::remap(src, dst, _interp_maps[0], _interp_maps[1], _interp);
}


int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);

  ImageRegistration reg;
  auto t = reg.lucasKanade(I, I);
  std::cout << "got: " << t.transpose() << std::endl;
  std::cout << "error: " << t.norm() << std::endl;

  auto t_true = typename ImageRegistration::Vector2d(1.0, 0.5);
  cv::Mat Iw;
  reg.imshift(I, Iw, -t_true);

  cv::Rect roi(10,10,100,100);
  t = reg.lucasKanade(I, roi, Iw);
  std::cout << "got: " << t.transpose() << std::endl;
  std::cout << "error: " << (t-t_true).norm() << std::endl;

  return 0;
}

