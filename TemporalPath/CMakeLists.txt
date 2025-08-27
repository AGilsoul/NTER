cmake_minimum_required(VERSION 3.21) # windows
# cmake_minimum_required(VERSION 3.18) # linux
project(TemporalPath)

set(CMAKE_CXX_STANDARD 14)

add_library(TemporalPath SHARED src/PathMaker.cpp include/PathMaker.h)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} lib)
target_link_libraries(TemporalPath ${PYTHON_LIBRARIES})
