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

#ifndef BITPLANES_CORE_UTILS_FS_H
#define BITPLANES_CORE_UTILS_FS_H

#include <string>

namespace bp {
namespace fs {

/**
 * \return directory separator, this is a slash '/'
 */
std::string dirsep(std::string fn);

/**
 * Expands '~' to user's home directory
 */
std::string expand_tilde(std::string);


/**
 * \return the extension of the input filename
 */
std::string extension(std::string filename);

/**
 * \return filename without the extension
 */
std::string remove_extension(std::string filename);

/**
 * \return the basename
 */
std::string getBasename(std::string filename);

/**
 * \return true if path exists
 */
bool exists(std::string path);

/**
 * \return true if path is a regular file
 */
bool is_regular(std::string path);

/**
 * \return true if directory
 */
bool is_dir(std::string path);


}; // fs
}; // bp

#endif // BITPLANES_CORE_UTILS_FS_H
