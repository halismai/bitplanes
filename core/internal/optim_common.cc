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


#include "bitplanes/core/internal/optim_common.h"
#include <cmath>
#include <cstdio>

namespace bp {

bool TestConverged(float dp_norm, float p_norm, float x_tol, float g_norm,
                   float tol_opt, float rel_factor, float new_f, float old_f,
                   float f_tol, float sqrt_eps, int it, int max_iters, bool verbose,
                   OptimizerStatus& status)
{
  if(it > max_iters)
  {
    if(verbose) printf("MaxIterations reached\n");

    return true;
  }

  if(g_norm < tol_opt * rel_factor)
  {
    if(verbose) printf("First order optimality reached\n");

    status = OptimizerStatus::FirstOrderOptimality;
    return true;
  }

  if(dp_norm < x_tol)
  {
    if(verbose) printf("Small abs step\n");

    status = OptimizerStatus::SmallAbsParameters;
    return true;
  }

  if(dp_norm < x_tol * (sqrt_eps * p_norm))
  {
    if(verbose) printf("Small change in parameters\n");

    status = OptimizerStatus::SmallParameterUpdate;
    return true;
  }

  if(std::fabs(old_f - new_f) < f_tol * old_f)
  {
    if(verbose) printf("Small relative reduction in error\n");

    status = OptimizerStatus::SmallRelativeReduction;
    return true;
  }

  return false;
}

} // bp

