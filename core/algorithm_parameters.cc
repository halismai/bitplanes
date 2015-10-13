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
#include "bitplanes/core/debug.h"
#include "bitplanes/utils/config_file.h"
#include "bitplanes/utils/icompare.h"

#include <iostream>

namespace bp {

AlgorithmParameters AlgorithmParameters::FromConfigFile(std::string filename)
{
  AlgorithmParameters ret;
  if(!ret.load(filename)) {
    Warn("Failed to load config from '%s'\n", filename.c_str());
    return AlgorithmParameters(); // return default config
  }

  return ret;
}

bool AlgorithmParameters::load(std::string filename)
{

  try {
    ConfigFile cf(filename);

    num_levels = cf.get<int>("NumLevels", -1);
    max_iterations = cf.get<int>("MaxIterations", 50);
    parameter_tolerance = cf.get<float>("ParameterTolerance", 1e-5f);
    function_tolerance = cf.get<float>("FunctionTolerance", 1e-5f);
    sigma = cf.get<float>("Sigma", 1.2f);
    verbose = cf.get<bool>("Verbose", true);
    multi_channel_function = MultiChannelExtractorTypeFromString(
    cf.get<std::string>("MultiChannelExtractorType", "BitPlanes"));
    linearizer = LinearizerTypeFromString(
        cf.get<std::string>("LinearizerType", "InverseCompositional"));

  } catch(const std::exception& ex) {
    Warn("Failed to load config from '%s'\n", filename.c_str());
    Warn("error '%s'\n", ex.what());
    return false;
  }

  return true;
}

bool AlgorithmParameters::save(std::string filename)
{
  try {
    ConfigFile cf;

    cf
        ("MultiChannelExtractorType", ToString(multi_channel_function))
        ("LinearizerType", ToString(linearizer)).set
        ("NumLevels", num_levels).set
        ("MaxIterations", max_iterations).set
        ("ParameterTolerance", parameter_tolerance).set
        ("FunctionTolerance", function_tolerance).set
        ("Sigma", sigma).set
        ("Verbose", verbose);

    cf.save(filename);
  } catch(const std::exception& ex) {
    Warn("Failed to save AlgorithmParameters to '%s'\n", filename.c_str());
    Warn("Error: '%s'\n", ex.what());
    return false;
  }

  return true;
}

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

AlgorithmParameters::MultiChannelExtractorType
MultiChannelExtractorTypeFromString(std::string name)
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
  else
    Warn("Unknown MultiChannelExtractorType '%s'\n", name.c_str());

  return AlgorithmParameters::MultiChannelExtractorType::BitPlanes;
}

std::string ToString(AlgorithmParameters::LinearizerType t)
{
  std::string ret;

  switch(t)
  {
    case AlgorithmParameters::LinearizerType::InverseCompositional:
      ret = "InverseCompositional";
      break;

    case AlgorithmParameters::LinearizerType::ForwardCompositional:
      ret = "ForwardCompositional";
      break;
  }

  return ret;
}

AlgorithmParameters::LinearizerType
LinearizerTypeFromString(std::string name)
{
  if(icompare("InverseCompositional", name) || icompare("IC", name))
    return AlgorithmParameters::LinearizerType::InverseCompositional;
  else if(icompare("ForwardCompositional", name) || icompare("FC", name))
    return AlgorithmParameters::LinearizerType::ForwardCompositional;
  else
    Warn("Unknown LinearizerType '%s'\n", name.c_str());

  return AlgorithmParameters::LinearizerType::InverseCompositional;
}


} // bp

