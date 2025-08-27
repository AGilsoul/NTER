# cmake_minimum_required(VERSION 3.18) # linux
cmake_minimum_required(VERSION 3.21) # windows
# set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86) # linux
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc) # linux


project(SingleTemporalFlame LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 14)

#add_executable(GradAscent
#        src/GradAscent.cu include/GradAscent.cuh)

# include(/home/u0667058/clion-2023.1.4/bin/cmake/linux/x64/share/cmake-3.25/Modules/FindCUDA/select_compute_arch.cmake)
# CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
# string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
# string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
# string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")


add_library(SingleTemporalFlame SHARED
        src/PathMaker.cu include/PathMaker.cuh)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} lib)
target_link_libraries(SingleTemporalFlame ${PYTHON_LIBRARIES})

set_target_properties(SingleTemporalFlame PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

