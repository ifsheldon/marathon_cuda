cmake_minimum_required(VERSION 3.17)
project(marathon_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(marathon_cuda main.cu kernel.cu kernel.cuh)

set_target_properties(
        marathon_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)