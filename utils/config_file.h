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

#ifndef BP_UTIL_CONFIG_FILE_H
#define BP_UTIL_CONFIG_FILE_H

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <map>

#include "bitplanes/utils/error.h"
#include "bitplanes/utils/memory.h"
#include "bitplanes/utils/str2num.h"

namespace bp {

/**
 * simple config file with data of the form
 *   VarName = Value
 *
 * Lines that begin with a '#' or '%' are treated as comments
 */
class ConfigFile
{
 public:
  typedef SharedPointer<ConfigFile> Pointer;

 public:
  ConfigFile(std::string filename);
  ConfigFile(std::ifstream& ifs);

  bool save(std::string filename) const;

  template <typename T> inline
  T get(std::string var_name) const;

  template <typename T> inline
  T get(std::string var_name, const T& default_val) const;

  template <typename T> inline
  ConfigFile& set(std::string var_name, const T& value);

  ConfigFile& operator()(const std::string&, const std::string&);

  friend std::ostream& operator<<(std::ostream&, const ConfigFile&);

 protected:
  void parse(std::ifstream&);
  std::map<std::string, std::string> _data;
}; // ConfigFile


template <typename T>
T ConfigFile::get(std::string name) const
{
  const auto& value_it = _data.find(name);
  if(value_it == _data.end())
    throw Error("no key " + name);

  T ret;
  if(!str2num(value_it->second, ret))
    throw Error("failed to convert '" + value_it->second +
                "' to type " + typeid(T).name());

  return ret;
}

template <typename T>
T ConfigFile::get(std::string name, const T& default_val) const
{
  try {
    return get<T>(name);
  } catch(const std::exception& ex) {
    if(0) std::cerr<<"config_file: get("<<name<<"): error: "<<ex.what()<<std::endl;
    return default_val;
  }
}

template <typename T> inline
ConfigFile& ConfigFile::set(std::string name, const T& value)
{
  _data[name] = value;
  return *this;
}

}; // bp

#endif // TT_UTIL_CONFIG_FILE_H
