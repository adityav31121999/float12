
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(float12 C CXX)
message("> Project Name: ${PROJECT_NAME} | Version: ${VERSION}")
message("> FLOAT12 (E5M6) Experimental Project for structure and logic verification")

# language settings
enable_language(C CXX)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# binaries for debug and release
set(OSbit 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OSbit 64)
    message("> Running on ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OSbit}-bit System")
endif()
set(FullOutputDir "${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OSbit}/${CMAKE_BUILD_TYPE}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static_libs")
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/shared_libs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

# Include directories
include_directories(src)
add_subdirectory(src)

# Create test executables
add_executable(float12 main.cpp float12.h)
add_executable(test_math test_math.cpp)
add_executable(test_advanced_math test_advanced_math.cpp)
add_executable(test_ml_training test_ml_training.cpp)

# Link executables to the float12 library
target_link_libraries(float12 float12_lib)
target_link_libraries(test_math float12_lib)
target_link_libraries(test_advanced_math float12_lib)
target_link_libraries(test_ml_training float12_lib)

if(MSVC)
    set_target_properties(float12 PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set_target_properties(test_math PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set_target_properties(test_advanced_math PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set_target_properties(test_ml_training PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
