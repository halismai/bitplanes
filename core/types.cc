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

#include "bitplanes/core/types.h"
#include <iostream>

namespace bp {

std::string ToString(OptimizerStatus status)
{
  std::string s;
  switch(status)
  {
    case OptimizerStatus::NotStarted:
      s = "NotStarted";
      break;
    case OptimizerStatus::MaxIterations:
      s = "MaxIterations";
      break;
    case OptimizerStatus::FirstOrderOptimality:
      s = "FirstOrderOptimality";
      break;
    case OptimizerStatus::SmallRelativeReduction:
      s = "SmallRelativeReduction";
      break;
    case OptimizerStatus::SmallAbsError:
      s = "SmallAbsError";
      break;
    case OptimizerStatus::SmallParameterUpdate:
      s = "SmallParameterUpdate";
      break;
    case OptimizerStatus::SmallAbsParameters:
      s = "SmallAbsParameters";
      break;
  }

  return s;
}

std::ostream& operator<<(std::ostream& os, const Result& r)
{
  os << "OptimizerStatus: " << ToString(r.status) << "\n";
  os << "NumIterations: " << r.num_iterations << "\n";
  os << "FinalSsdError: " << r.final_ssd_error << "\n";
  os << "FirstOrderOptimality: " << r.first_order_optimality << "\n";
  os << "TimeMilliSeconds: " << r.time_ms << "\n";
  os << "T:\n" << r.T;

  return os;
}

std::string ToString(MotionType m)
{
  std::string ret;

  switch(m)
  {
    case MotionType::Translation:
      ret = "Translation";
      break;

    case MotionType::Affine:
      ret = "Affine";
      break;

    case MotionType::Homography:
      ret = "Homography";
      break;
  }

  return ret;
}

} // bp
