//
// Created by Maple on 2020/12/5.
//

#ifndef MARATHON_CUDA_UTIL_FUNCS_HPP
#define MARATHON_CUDA_UTIL_FUNCS_HPP

#include "scene.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "render.cuh"

using glm::vec4;
using glm::vec3;

Scene setupScene()
{
    Scene scene(vec3(0.5));
    mat4 identity(1.0);
    scene.addMaterial(vec3(1.0), vec3(0.5), vec3(1.0), vec3(0.5), 64.0f);
    scene.addMaterial(vec3(0.16, 0.14, 0.02), vec3(0.8, 0.7, 0.1), vec3(1.0), vec3(0.5), 64.0f);
    scene.addLight(vec3(0.4, -3, 0.1), vec3(0.1), vec3(1.0));
    scene.addSphere(0.5, 0, identity);
    mat4 cylinderTransformation = glm::translate(identity, vec3(-2.0, 0.0, 0.0));
    scene.addCylinder(0.5, 1.0, 0, cylinderTransformation);
    return scene;
}

mat4 lookAt(vec3 eye, vec3 center, vec3 up)
{
    using namespace glm;
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
            vec4(s, 0.0),
            vec4(u, 0.0),
            vec4(-f, 0.0),
            vec4(0.0, 0.0, 0.0, 1)
    );
}

#endif //MARATHON_CUDA_UTIL_FUNCS_HPP
