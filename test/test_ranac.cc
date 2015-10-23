#include "bitplanes/core/ransac.h"
#include <iostream>

using namespace bp;

static typename RansacHomography::CorrespondencesType
GetCorrespondences(size_t n = 1000)
{
  typename RansacHomography::CorrespondencesType ret(n);

  return ret;
}

int main()
{
  const auto& corrs = GetCorrespondences();
  RansacHomography model(corrs);
  Ransac<RansacHomography> ransac(model, 1.2);

  auto result = ransac.fit();
  std::cout << result << std::endl;

  return 0;
}



