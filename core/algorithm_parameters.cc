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

#include "bitplanes/core/algorithm_parameters.h"
#include "bitplanes/utils/config_file.h"
#include "bitplanes/utils/icompare.h"

#include <iostream>

namespace bp {

std::ostream& operator<<(std::ostream& os, const AlgorithmParameters& p)
{
  os << "MultiChannelFunction = " << ToString(p.multi_channel_function) << "\n";
  os << "ParameterTolerance = " << p.parameter_tolerance << "\n";
  os << "FunctionTolerance = " << p.function_tolerance << "\n";
  os << "NumLevels = " << p.num_levels << "\n";
  os << "sigma = " << p.sigma << "\n";
  os << "verbose = " << p.verbose;
  return os;
}

std::string ToString(AlgorithmParameters::MultiChannelExtractorType m)
{
  std::string ret;
  switch(m) {
    case AlgorithmParameters::MultiChannelExtractorType::IntensityGrayChannel:
      ret = "IntensityGrayChannel";
      break;

    case AlgorithmParameters::MultiChannelExtractorType::GradientAbsMag:
      ret = "GradientAbsMag";
      break;

    case AlgorithmParameters::MultiChannelExtractorType::IntensityAndGradient:
      ret = "IntensityAndGradient";
      break;

    case AlgorithmParameters::MultiChannelExtractorType::CensusChannel:
      ret = "CensusChannel";
      break;

    case AlgorithmParameters::MultiChannelExtractorType::DescriptorFields1:
      ret = "DescriptorFields1";
      break;

    case AlgorithmParameters::MultiChannelExtractorType::DescriptorFields2:
      ret = "DescriptorFields2";
      break;

    case AlgorithmParameters::MultiChannelExtractorType::BitPlanes:
      ret = "BitPlanes";
      break;
  }

  return ret;
}


AlgorithmParameters::MultiChannelExtractorType FromString(std::string name)
{
  if(icompare("IntensityGrayChannel", name))
    return AlgorithmParameters::MultiChannelExtractorType::IntensityGrayChannel;
  else if(icompare("GradientAbsMag", name))
    return AlgorithmParameters::MultiChannelExtractorType::GradientAbsMag;
  else if(icompare("IntensityAndGradient", name))
    return AlgorithmParameters::MultiChannelExtractorType::IntensityAndGradient;
  else if(icompare("CensusChannel", name))
    return AlgorithmParameters::MultiChannelExtractorType::CensusChannel;
  else if(icompare("DescriptorFields1", name))
    return AlgorithmParameters::MultiChannelExtractorType::DescriptorFields1;
  else if(icompare("DescriptorFields2", name))
    return AlgorithmParameters::MultiChannelExtractorType::DescriptorFields2;
  else if(icompare("BitPlanes", name))
    return AlgorithmParameters::MultiChannelExtractorType::BitPlanes;

  return AlgorithmParameters::MultiChannelExtractorType::BitPlanes;
}

} // bp
