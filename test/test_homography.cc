#include "bitplanes/core/homography.h"
#include <iostream>
#include <cmath>
#include <Eigen/LU>

int main()
{
  typename bp::Homography::ParameterVector p;

  for(int j = 0; j < 1000; ++j)
  {
    for(int i = 0; i < 8; ++i) p[i] = rand()/(float) RAND_MAX;

    auto H = bp::Homography::ParamsToMatrix(p);
    auto p2 = bp::Homography::MatrixToParams(H);

    if( std::abs(H.determinant() - 1) > 1e-6 )
      std::cerr << "determinant is bad " << H.determinant() << std::endl;

    float err = (p2 - p).squaredNorm();
    if(err > 1e-6)
      std::cerr <<  "bad error " << err << std::endl;
  }

  return 0;
}

