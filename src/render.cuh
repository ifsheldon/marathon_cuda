//
// Created by Maple on 2020/12/4.
//

#ifndef MARATHON_CUDA_RENDER_CUH
#define MARATHON_CUDA_RENDER_CUH

#include "camera.hpp"
#include "scene.hpp"
#include "glm/glm.hpp"
#include "render_setting.hpp"

#define MAX_LIGHT_NUM 10
#define MAX_MATERIAL_NUM 50
#define MAX_OBJ_NUM 50

__constant__ const unsigned int MAX_SSR = 5;

using glm::vec2;
typedef glm::vec<3, unsigned char, glm::defaultp> color_u8;

__global__ void
renderer(const Camera camera, const CameraConfig cameraConfig, const vec2 window_size,
         const float z,
         const unsigned int light_num,
         const unsigned int obj_num,
         const vec3 background_color,
         const RenderSetting renderSetting,
         color_u8* output_colors);

void configure_all_device_funcs();

#endif //MARATHON_CUDA_RENDER_CUH
