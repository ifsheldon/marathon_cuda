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

Scene* setupScene()
{
    auto scene = new Scene(vec3(0.5));
    mat4 identity(1.0);
    scene->addMaterial(vec3(1.0), vec3(0.5), vec3(1.0), vec3(0.5), 64.0f);
    scene->addMaterial(vec3(0.16, 0.14, 0.02), vec3(0.8, 0.7, 0.1), vec3(1.0), vec3(0.5), 64.0f);
    scene->addLight(vec3(0.4, -3, 0.1), vec3(0.1), vec3(1.0));
    for (int x = -2; x <= 2; x++)
    {
        for (int z = 0; z <= 2; z++)
        {
            mat4 sphere_trans = glm::translate(identity, vec3(x, 1, z));
            scene->addSphere(0.25, 0, sphere_trans);
            mat4 cylinder_trans = glm::translate(identity, vec3(x, 0, z));
            scene->addCylinder(0.25, 0.75, 1, cylinder_trans);
        }
    }
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

bool queryGPUCapabilitiesCUDA()
{
    // Device Count
    int devCount;

    // Get the Device Count
    cudaGetDeviceCount(&devCount);

    // Print Device Count
    printf("Device(s): %i\n", devCount);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    std::cout << "Max Block Dim in a Grid (" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", "
              <<
              properties.maxGridSize[2] << ")" << std::endl;
    std::cout << "Max Thread per Block: " << properties.maxThreadsPerBlock << std::endl;
    std::cout << "Max Thread Dim in a Block (" << properties.maxThreadsDim[0] << ", " << properties.
            maxThreadsDim[1] <<
              ", " << properties.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Share Memory per Block (bytes): " << properties.sharedMemPerBlock << std::endl;
    std::cout << "Mem Pitch (bytes) " << properties.memPitch << std::endl;
    std::cout << "Total Constant Memory (bytes): " << properties.totalConstMem << std::endl;
    std::cout << "Warp Size: " << properties.warpSize << std::endl;
    return devCount > 0;
}

#endif //MARATHON_CUDA_UTIL_FUNCS_HPP
