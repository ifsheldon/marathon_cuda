//
// Created by Maple on 2020/12/4.
//

#ifndef MARATHON_CUDA_RENDER_CUH
#define MARATHON_CUDA_RENDER_CUH

#include "camera.hpp"
#include "scene.hpp"
#include "glm/glm.hpp"

#define MAX_LIGHT_NUM 10
#define MAX_MATERIAL_NUM 50
#define MAX_OBJ_NUM 50

using glm::vec2;

__global__ void
renderer(const unsigned int random_seed, const Camera camera, const CameraConfig cameraConfig, const vec2 window_size,
         const float z,
         const int light_num,
         const int obj_num,
         const int ray_marching_level,
         vec3* output_colors);

void run();

#endif //MARATHON_CUDA_RENDER_CUH
