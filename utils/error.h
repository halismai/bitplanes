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
#ifndef BP_UTILS_ERROR_H
#define BP_UTILS_ERROR_H

#include "bitplanes/core/debug.h"
#include "bitplanes/utils/utils.h"

#include <string>
#include <stdexcept>

namespace bp {

std::string errno_string();

struct Error : public std::logic_error
{
  inline Error(std::string what)
      : logic_error(what) {}
}; // Error

#define THROW_ERROR(msg) \
    throw Error(Format("[ %s:%04d ] %s", MYFILE, __LINE__, msg))

#define THROW_ERROR_IF(cond, msg) if( !!(cond) ) THROW_ERROR( (msg) )

}; // bp

#endif // BP_UTILS_ERROR_H

