## TODO check version
##  This file is part of bitplanes.
##
##  bitplanes is free software: you can redistribute it and/or modify
##  it under the terms of the Lesser GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  bitplanes is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  Lesser GNU General Public License for more details.
##
##  You should have received a copy of the Lesser GNU General Public License
##  along with bitplanes.  If not, see <http://www.gnu.org/licenses/>.
##

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(bitplanes C CXX)

set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation directory")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Config" FORCE)


#if(NOT IOS)
#  include(cmake/ConfigureSSE.cmake)
#endif()

include(cmake/MakeVersion.cmake)
MakeVersion(BITPLANES 1 8 0)

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "BitPlanes")
set(CPACK_PACKAGE_VENDOR              "halismai Carnegie Mellon University")
set(CPACK_PACKAGE_CONTACT             "halismai@cs.cmu.edu")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_RESOURCE_FILE_LICENSE    "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_SOURCE_GENERATOR "TBZ2")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${BITPLANES_VERSION}")

set(CPACK_SOURCE_IGNORE_FILES
  ${CPACK_SOURCE_IGNORE_FILES}
  "/build/"
  "/.hg/"
  "/.git/"
  "/*.swp"
  "${PROJECT_SOURCE_DIR}/.hgignore"
  "${PROJECT_SOURCE_DIR}/.gitignore")

include(CPack)
add_custom_target(srcdist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)
message(STATUS "dist file: ${CPACK_SOURCE_PACKAGE_FILE_NAME}")
message(STATUS "version  : ${BITPLANES_VERSION}")

option(BUILD_STATIC            "Build a static library instead of shared" ON)
option(BITPLANES_WITH_OPENMP   "Enable openmp"                            ON)
option(BITPLANES_WITH_PROFILER "Use google profiler"                      OFF)
option(BITPLANES_WITH_TCMALLOC "Use google tcmalloc"                      OFF)
option(BITPLANES_WITH_TBB      "Use Intel TBB"                            ON)
option(BITPLANES_WITH_TIMING   "Enabling timing the code"                 ON)
option(BITPLANES_WITH_BOOST    "Enable boost"                            OFF)

option(ENABLE_PROFILING           "enable profiling"  OFF)
option(ENABLE_FAST_MATH           "enable fast math"  OFF)
option(ENABLE_OMIT_FRAME_POINTER  "omit frame pointer" ON)


# IOS does not have many of the packages, unset them
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  message(STATUS "Building for: ${CMAKE_SYSTEM_NAME}")
  unset(BITPLANES_WITH_OPENMP     CACHE)
  unset(BITPLANES_WITH_TCMALLOC   CACHE)
  unset(BITPLANES_WITH_BOOST      CACHE)
  unset(BITPLANES_WITH_TBB        CACHE)
  unset(BITPLANES_WITH_PROFILER   CACHE)
endif()

if(IOS)
  message(STATUS "Building for IOS: ${IOS_PLATFORM}")
  unset(BITPLANES_WITH_OPENMP     CACHE)
  unset(BITPLANES_WITH_TCMALLOC   CACHE)
  unset(BITPLANES_WITH_BOOST      CACHE)
  unset(BITPLANES_WITH_TBB        CACHE)
  unset(BITPLANES_WITH_PROFILER   CACHE)

  set(OpenCV_DIR "/usr/local/opt/opencv3/share/OpenCV/")

  if(IOS_PLATFORM STREQUAL "OS")
    message(STATUS "Enabling NEON")
    set(BITPLANES_HAVE_ARM 1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -ftree-vectorize -O3")
  else()
    set(BITPLANES_HAVE_SSE2 1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
  endif()
else()
  set(BITPLANES_HAVE_ARM 0)
endif()


include(cmake/SetCompilerOptions.cmake)

configure_file("${PROJECT_SOURCE_DIR}/bitplanes_config.h.in"
  "${PROJECT_SOURCE_DIR}/core/config.h")


if(BITPLANES_WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DBITPLANES_WITH_OPENMP)
  endif()
endif()


#
# REQUIRED libraries
#
# we need opencv3! TODO check version
set(OpenCV_DIR "/opt/opencv3/share/OpenCV/")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs features2d calib3d)
list(APPEND MY_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND MY_LINK_DIRS    ${OpenCV_LINK_DIRECTORIES})
list(APPEND MY_LIBRARIES    ${OpenCV_LIBS})

find_package(Eigen REQUIRED)
if(EIGEN_VERSION VERSION_LESS 3.2.0)
  message(FATAL_ERROR "bitplanes requires Eigen version >= 3.2.0")
endif()
list(APPEND MY_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})

#
# optional stuff
#
if(BITPLANES_WITH_PROFILER)
  find_package(GooglePerfTools)
  if(GOOGLE_PERFTOOLS_FOUND)
    list(APPEND MY_INCLUDE_DIRS ${GOOGLE_PERFTOOLS_INCLUDE_DIR})
    list(APPEND MY_LIBRARIES    ${PROFILER_LIBRARY})
  else()
    message(WARNING "could not find GooglePerfTools")
    unset(BITPLANES_WITH_PROFILER CACHE)
  endif()
endif()

if(BITPLANES_WITH_TCMALLOC)
  find_package(GooglePerfTools)
  if(GOOGLE_PERFTOOLS_FOUND)
    list(APPEND MY_INCLUDE_DIRS ${GOOGLE_PERFTOOLS_INCLUDE_DIR})
    list(APPEND MY_LIBRARIES ${TCMALLOC_LIBRARY})
  else()
    message(WARNING "could not find GooglePerfTools")
    unset(BITPLANES_WITH_TCMALLOC CACHE)
  endif()
endif()

if(BITPLANES_WITH_BOOST)
  if(BUILD_STATIC)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_STATIC_RUNTIME ON)
  else()
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_STATIC_RUNTIME OFF)
  endif()

  set(Boost_USE_MULTITHREAD ON)
  find_package(Boost 1.55.0 REQUIRED COMPONENTS program_options timer system)

  if(Boost_FOUND)
    list(APPEND MY_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
    list(APPEND MY_LIBRARIES ${Boost_LIBRARIES})
    list(APPEND MY_DEFINES ${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
  else()
    message(WARNING "could not find boost")
    unset(BITPLANES_WITH_BOOST CACHE)
  endif()
endif()

if(BITPLANES_WITH_TBB)
  find_package(TBB)
  if(TBB_FOUND)
    list(APPEND MY_INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
    list(APPEND MY_LIBRARIES    ${TBB_LIBRARIES})
  else()
    unset(BITPLANES_WITH_TBB CACHE)
    message(WARNING "could not find TBB")
  endif()
endif()


get_filename_component(P_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
list(APPEND MY_INCLUDE_DIRS "${P_DIR}")

add_definitions(${MY_DEFINES})
add_definitions(-D_DEBUG_)
include_directories(${MY_INCLUDE_DIRS})
link_directories(${MY_LINK_DIRS})

add_subdirectory(utils)
add_subdirectory(core)

if(NOT IOS)
  add_subdirectory(test)
endif()

add_subdirectory(demo_cv)

