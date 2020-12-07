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


void test()
{
    int size = 32;
    Scene scene = setupScene();
    dim3 dimGrid(1);
    dim3 dimBlock(size, size);
    auto mem_size = sizeof(color_u8) * size * size;
    color_u8* output_d;
    cudaMalloc(&output_d, mem_size);
    Camera camera = {vec3(0.0, 0.0, -6.0), vec3(0.0, 1.0, 0.0), vec3(0.0),
                     lookAt(vec3(0.0, 0.0, -6.0), vec3(0.0), vec3(0.0, 1.0, 0.0))};
    CameraConfig cameraConfig = {vec3(0.01, 100.0, glm::radians(90.0))};
    float z = size / tan(cameraConfig.config.z / 2.0);
    Light* lights_d;
    cudaMalloc(&lights_d, sizeof(Light) * scene.getLightNum());
    cudaMemcpy(lights_d, &scene.lights[0], scene.getLightNum(), cudaMemcpyHostToDevice);

    Material* materials_d;
    cudaMalloc(&materials_d, sizeof(Material) * scene.getMaterialNum());
    cudaMemcpy(materials_d, &scene.materials[0], scene.getMaterialNum(), cudaMemcpyHostToDevice);

    Object* objects_d;
    cudaMalloc(&objects_d, sizeof(Object) * scene.getObjNum());
    cudaMemcpy(objects_d, &scene.objects[0], scene.getObjNum(), cudaMemcpyHostToDevice);

    renderer <<< dimGrid, dimBlock>>>(camera, cameraConfig, vec2(size, size), z, scene.getLightNum(),
                                      scene.getObjNum(),
                                      2,
                                      vec3(0.5),
                                      2,
                                      output_d);
    color_u8* output_h = new color_u8[size * size];
    cudaMemcpy(output_h, output_d, mem_size, cudaMemcpyDeviceToHost);
    vec3 sum(0.f);
    for (int i = 0; i < size * size; i++)
    {
        sum += output_h[i];
    }
    printf("sum = %f, %f, %f\n", sum.x, sum.y, sum.z);
    delete[] output_h;
    cudaFree(output_d);
    cudaFree(lights_d);
    cudaFree(materials_d);
    cudaFree(objects_d);
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
