# src/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(float12_lib C CXX)

add_library(float12_lib STATIC
    operation.cpp
    convert.cpp
    error.cpp
    math.cpp
    constants.cpp
    ml_support.cpp
)

# Make the include directory available to this target and any
# other target that links to it (like the main executable).
target_include_directories(float12_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
