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

#include "bitplanes/utils/fs.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <libgen.h>

#include <iostream>
#include <cstring>

namespace bp {
namespace fs {

using std::string;

string expand_tilde(string fn)
{
  if(fn.front() == '~') {
    string home = getenv("HOME");
    if(home.empty()) {
      std::cerr << "could not query $HOME\n";
      return fn;
    }

    // handle the case when name == '~' only
    return home + dirsep(home) + ((fn.length()==1) ? "" :
                                  fn.substr(1,std::string::npos));
  } else {
    return fn;
  }
}

string dirsep(string dname)
{
  return (dname.back() == '/') ? "" : "/";
}

string extension(string filename)
{
  auto i = filename.find_last_of(".");
  return (string::npos != i) ? filename.substr(i) : "";
}

string remove_extension(string filename)
{
  auto i = filename.find_last_of(".");
  return (string::npos != i && i != 0) ? filename.substr(0, i) : filename;
}

string getBasename(string filename)
{
  char* s = strndup(filename.c_str(), filename.length());
  std::string ret(::basename(s));
  free(s);
  return ret;
}

bool exists(string path)
{
  struct stat buf;
  return (0 == stat(path.c_str(), &buf));
}

bool is_regular(string path)
{
  struct stat buf;
  return (0 == stat(path.c_str(), &buf)) ? S_ISREG(buf.st_mode) : false;
}

bool is_dir(string path)
{
  struct stat buf;
  return (0 == stat(path.c_str(), &buf)) ? S_ISDIR(buf.st_mode) : false;
}

}
}


