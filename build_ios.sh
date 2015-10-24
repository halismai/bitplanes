#!/bin/bash

# based on ceres-solver build_ios.sh
export OpenCV_DIR=/usr/local/opt/opencv3/include/
EIGEN_INCLUDE_DIR=/usr/local/opt/eigen/include/eigen3/

mkdir -p build/os && cd build/os
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=../../cmake/iOS.cmake  \
  -DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}                \
  -DIOS_PLATFORM=OS                                       \
  -DCMAKE_BUILD_TYPE=Release                              \
  -DCMAKE_INSTALL_PREFIX=..

make VERBOSE=1 -j8 && make install

cd ../../

mkdir -p build/simulator && cd build/simulator
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=../../cmake/iOS.cmake  \
  -DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}                \
  -DIOS_PLATFORM=SIMULATOR                                \
  -DCMAKE_BUILD_TYPE=Release                              \
  -DCMAKE_INSTALL_PREFIX=..

make -j8

cd ../../

mkdir -p build/simulator64 && cd build/simulator64
cmake ../.. -DCMAKE_TOOLCHAIN_FILE=../../cmake/iOS.cmake  \
  -DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}                \
  -DIOS_PLATFORM=SIMULATOR64                              \
  -DCMAKE_BUILD_TYPE=Release                              \
  -DCMAKE_INSTALL_PREFIX=..

make -j8

cd ../

mkdir -p lib
lipo -create                            \
  os/bin/libbitplanes_core.a            \
  simulator/bin/libbitplanes_core.a     \
  simulator64/bin/libbitplanes_core.a   \
  -output libbitplanes_core-universal.a

lipo -create                            \
  os/bin/libbitplanes_utils.a           \
  simulator/bin/libbitplanes_utils.a    \
  simulator64/bin/libbitplanes_utils.a  \
  -output libbitplanes_utils-universal.a

