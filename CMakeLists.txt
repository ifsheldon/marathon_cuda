cmake_minimum_required(VERSION 3.17)
project(marathon_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)
include_directories("./lib/Cimg" "./lib/glm" "./src/utils" "./src" "./lib")
add_executable(marathon_cuda src/main.cu src/render.cu src/render.cuh src/utils/scene.hpp src/utils/camera.hpp src/utils/util_funcs.hpp src/utils/helper_cuda.h src/utils/helper_string.h src/utils/parser.hpp src/utils/render_setting.hpp)

set_target_properties(
        marathon_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)