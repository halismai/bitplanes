/*
  This file is part of bitplanes.

  bitplanes is free software: you can redistribute it and/or modify
  it under the terms of the Lesser GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  bitplanes is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  Lesser GNU General Public License for more details.

  You should have received a copy of the Lesser GNU General Public License
  along with bitplanes.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <bitplanes/core/algorithm_parameters.h>
#include <bitplanes/core/config.h>

#include <iostream>

using namespace bp;

int main()
{
  Info("\n%s\n\n", BITPLANES_BUILD_STRING);

  auto params = AlgorithmParameters::FromConfigFile("../config/test.cfg");
  std::cout << params << std::endl;

  params.save("/tmp/test.cfg");
  return 0;
}


