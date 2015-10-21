#include "bitplanes/core/internal/census_signature.h"
#include <iostream>

int main()
{
  int stride = 32;
  alignas(16) uint8_t data[ stride*6 ];

  std::cout << bp::CensusSignatureSIMD(data + 16, stride) << std::endl;

  return 0;
}
