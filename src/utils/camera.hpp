//
// Created by Maple on 2020/12/4.
//

#ifndef MARATHON_CUDA_CAMERA_HPP
#define MARATHON_CUDA_CAMERA_HPP

#include "glm/glm.hpp"

using glm::vec3;
using glm::mat4;

struct Camera
{
    vec3 position;
    vec3 up;
    vec3 center;
    mat4 look_at_mat;
};

struct CameraConfig
{
    vec3 config; // near, far, fovy_radian
};

enum Mode
{
    Orbit, Zoom
};

#endif //MARATHON_CUDA_CAMERA_HPP
