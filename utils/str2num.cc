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

#include "bitplanes/utils/str2num.h"
#include "bitplanes/utils/icompare.h"

#include <stdexcept>

namespace bp {

template<> int str2num<int>(const std::string& s) { return std::stoi(s); }

template <> double str2num<double>(const std::string& s) { return std::stod(s); }

template <> float str2num<float>(const std::string& s) { return std::stof(s); }

template <> bool str2num<bool>(const std::string& s)
{
  if(icompare(s, "true")) {
    return true;
  } else if(icompare(s, "false")) {
    return false;
  } else {
    // try to parse a bool from int {0,1}
    int v = str2num<int>(s);
    if(v == 0)
      return false;
    else if(v == 1)
      return true;
    else
      throw std::invalid_argument("string is not a boolean");
  }
}

} // bp

