#include "bitplanes/core/ransac.h"
#include "bitplanes/core/ransac_model.h"
#include "bitplanes/core/homography.h"
#include <iostream>

using namespace bp;

inline Eigen::Vector3f normHomog(const Eigen::Vector3f& x)
{
  return x / x[2];
}

/**
 */
static inline typename RansacHomography::CorrespondencesType
GetCorrespondences(const bp::Matrix33f& H, size_t n = 500)
{
  typename RansacHomography::CorrespondencesType ret(n);

  for(auto& c : ret)
  {
    c.x1 = normHomog( c.x1.setRandom() );
    c.x2 = normHomog( H * c.x1 );
  }

  return ret;
}

int main()
{
  const auto H_true = Homography::ParamsToMatrix(Homography::ParameterVector::Random());
  const auto& corrs = GetCorrespondences(H_true, 500);
  RansacHomography model(corrs);

  typename RansacHomography::SampleIndices s_inds;
  for(size_t i = 0; i < s_inds.size(); ++i)
    s_inds[i] = i;

  auto H_est = model.run(s_inds);

  const auto inliers  = model.findInliers(H_est, 1.0);
  std::cout << "GOT: " << inliers.size() << std::endl;
  exit(0);

  Ransac<RansacHomography> ransac(model, 1.2);

  auto result = ransac.fit(100);
  std::cout << result << std::endl;

  std::cout << "H_true\n" << H_true << std::endl;

  return 0;
}



