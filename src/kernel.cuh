//
// Created by Maple on 2020/12/4.
//

#ifndef MARATHON_CUDA_KERNEL_CUH
#define MARATHON_CUDA_KERNEL_CUH

#include "camera.hpp"
#include "scene.hpp"
#include "glm/glm.hpp"

using glm::vec2;

void test();

__global__ void
renderer(const unsigned int random_seed, const Camera camera, const CameraConfig cameraConfig, const vec2 window_size,
         const float z,
         const Light* __restrict__ lights,
         const int light_num,
         const Material* __restrict__ materials,
         const Object* __restrict__ objects,
         const int obj_num,
         const int ray_marching_level,
         vec3* output_colors);

#endif //MARATHON_CUDA_KERNEL_CUH
