cmake_minimum_required(VERSION 3.17)
project(marathon_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)
include_directories("./lib/Cimg" "./lib/glm" "./src/utils")
add_executable(marathon_cuda src/main.cu src/kernel.cu src/kernel.cuh src/utils/scene.cpp src/utils/scene.hpp)

set_target_properties(
        marathon_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)