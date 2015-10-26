#include "bitplanes/core/internal/fit_homography.h"
#include "bitplanes/core/types.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/timer.h"

#include <iostream>

using namespace bp;

Eigen::Vector3f normHomog(const Eigen::Vector3f& x)
{
  return x / x[2];
}

int main()
{
  typedef bp::EigenStdVector<Eigen::Vector3f>::type PointVector;

  const int N = 500;
  PointVector x1(N), x2(N);

  for(int it = 0; it < 50; ++it)
  {
    const auto H_true = Homography::ParamsToMatrix(Homography::ParameterVector::Random());

    for(int i = 0; i < N; ++i)
    {
      x1[i] = normHomog(x1[i].setRandom());
      x2[i] = normHomog(H_true * x1[i]);
    }

    const auto H_est = FitHomography(x1, x2);

    Eigen::VectorXf err(x1.size());
    for(size_t i = 0; i < x1.size(); ++i)
    {
      err[i] = (normHomog(H_est * x1[i]) - normHomog(x2[i])).squaredNorm();
    }

    float e = err.template lpNorm<Eigen::Infinity>();
    if(N < 10 && e > 1e-6) {
      printf("Error: %g\n", e);
    }
  }


  Eigen::Matrix3f H;
  auto t_ms = TimeCode(1000, [&] () { H = FitHomography(x1,x2); });
  printf("time %f ms\n", t_ms);

  return 0;
}


