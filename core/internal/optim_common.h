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


#ifndef BITPLANES_CORE_INTERNAL_OPTIM_COMMON_H
#define BITPLANES_CORE_INTERNAL_OPTIM_COMMON_H

#include "bitplanes/core/types.h"

namespace bp {

bool TestConverged(float dp_norm, float p_norm, float x_tol, float g_norm,
                   float tol_opt, float rel_factor, float new_f, float old_f,
                   float f_tol, float sqrt_eps, int it, int max_iters, bool verbose,
                   OptimizerStatus& status);

}; // bp

#endif // BITPLANES_CORE_INTERNAL_OPTIM_COMMON_H

