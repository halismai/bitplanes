##
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

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

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

project(bitplanes CXX)

include(cmake/CheckC++11.cmake)
CheckCpp11()

if(NOT IOS)
  include(cmake/ConfigureSSE.cmake)
endif()

set(BITPLANES_VERSION_MAJOR "1")
set(BITPLANES_VERSION_MINOR "5")
set(BITPLANES_VERSION_PATCH "0")

execute_process(COMMAND date OUTPUT_VARIABLE build_date OUTPUT_STRIP_TRAILING_WHITESPACE)
set(BITPLANES_BUILD_DATE "${build_date}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fstrict-aliasing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")

option(BUILD_STATIC            "Build a static library instead of shared" ON)
option(BITPLANES_WITH_OPENMP   "Enable openmp"                            ON)
option(BITPLANES_WITH_PROFILER "Use google profiler"                      OFF)
option(BITPLANES_WITH_TCMALLOC "Use google tcmalloc"                      OFF)
option(BITPLANES_WITH_TBB      "Use Intel TBB"                            ON)
option(BITPLANES_WITH_TIMING   "Enabling timing the code"                 ON)
option(BITPLANES_WITH_BOOST    "Enable boost"                            OFF)

if(BUILD_STATIC)
  set(LIBRARY_TYPE STATIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
else()
  set(LIBRARY_TYPE SHARED)
endif()

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

if(BITPLANES_WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

configure_file("${PROJECT_SOURCE_DIR}/bitplanes_config.h.in"
  "${PROJECT_SOURCE_DIR}/core/config.h")

#
# REQUIRED libraries
#
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs features2d calib3d)
list(APPEND MY_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND MY_LINK_DIRS    ${OpenCV_LINK_DIRECTORIES})
list(APPEND MY_LIBRARIES    ${OpenCV_LIBS})

#pkg_check_modules(Eigen REQUIRED eigen3)
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
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${MY_LINK_DIRS})

add_subdirectory(utils)
add_subdirectory(core)

if(NOT IOS)
  add_subdirectory(test)
endif()

