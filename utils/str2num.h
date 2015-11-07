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

#ifndef BP_UTILS_STR2NUM_H
#define BP_UTILS_STR2NUM_H

#include <string>
#include <sstream>
#include <vector>

namespace bp {

/**
 * converts the input string to a number
 */
template <typename T> T str2num(const std::string&);

template <> int str2num<int>(const std::string& s);
template <> double str2num<double>(const std::string& s);
template <> float str2num<float>(const std::string& s);
template <> bool str2num<bool>(const std::string& s);

/**
 * converts string to number
 * \return false if conversion to the specified type 'T' fails
 *
 * e.g.:
 *
 * double num;
 * assert( true == str2num("1.6", num) );
 * assert( false == str2num("hello", num) );
 *
 */
template <typename T> inline
bool str2num(std::string str, T& num)
{
  std::istringstream ss(str);
  return !(ss >> num).bad();
}

/**
 * Uses the delimiter to split the string into tokens of numbers, e.g,
 *
 * string str = "1.2 1.3 1.4 1.5";
 * auto tokens = splitstr(str, ' ');
 *
 * // 'tokens' now has [1.2, 1.3, 1.4, 1.5]
 */
std::vector<std::string> splitstr(const std::string& str, char delim = ' ');

template <typename T> inline
std::vector<T> str2num(const std::vector<std::string>& strs)
{
  std::vector<T> ret(strs.size());
  for(size_t i = 0; i < strs.size(); ++i)
    ret[i] = str2num<T>(strs[i]);

  return ret;
}


}; // bp

#endif // BP_UTILS_STR2NUM_H

