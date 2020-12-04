//
// Created by Maple on 2020/12/4.
//

#include "render.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

using namespace glm;

__constant__ Light Lights[MAX_LIGHT_NUM];
__constant__ Object Objects[MAX_OBJ_NUM];
__constant__ Material Materials[MAX_MATERIAL_NUM];

struct Ray
{
    vec3 origin;
    vec3 direction;
};

__constant__ float EPSILON = 0.001;
__constant__ int MAX_MARCHING_STEPS = 255;

__device__ int light_num;
__device__ int obj_num;

__device__ float sdSphere(vec3 ref_pos, float s)
{
    return length(ref_pos) - s;
}

__device__ float sdCylinder(vec3 p, float r, float h)
{
    vec2 d = abs(vec2(length(vec2(p.x, p.z)), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}

__device__ float calcDist(vec3 ref_point, uint objIdx, float far)
{
    vec4 refP = vec4(ref_point, 1.0);
    refP = Objects[objIdx].transformation * refP;
    ref_point = vec3(refP) / refP.w;
    switch (Objects[objIdx].shape)
    {
        case Shape::Sphere:
            return sdSphere(ref_point, Objects[objIdx].dims.x);
        case Shape::Cylinder:
            return sdCylinder(ref_point, Objects[objIdx].dims.x, Objects[objIdx].dims.y);
        default :
            return far;
    }
}

__device__ float unionSDF(float* distances, int* objIdx)
{
    float min_dist = distances[0];
    *objIdx = 0;
    for (int i = 0; i < obj_num; i++)
    {
        if (distances[i] < min_dist)
        {
            min_dist = distances[i];
            *objIdx = i;
        }
    }
    return min_dist;
}

__device__ void sceneSDF(vec3 ref_point, float* distances, float far)
{
    for (uint i = 0; i < obj_num; i++)
    {
        distances[i] = calcDist(ref_point, i, far);
    }
}

__device__ float
shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start_dist,
                          float limit_dist,
                          int preObj,
                          float far,
                          int* objectIndex)
{
    float depth = start_dist;
    float distances[MAX_OBJ_NUM] = {0.f};
    for (int i = 0; i < MAX_MARCHING_STEPS; i++)
    {
        sceneSDF(eye + depth * marchingDirection, distances, far);
        if (preObj != -1)
            distances[preObj] = 2.0f * far;
        int hitObjIdx;
        float dist = unionSDF(distances, &hitObjIdx);
        if (dist < EPSILON)
        {
            *objectIndex = hitObjIdx;
            return depth;
        }
        depth += dist;
        if (depth >= limit_dist)
        {
            *objectIndex = obj_num;
            return limit_dist;
        }
    }
    return limit_dist;
}

__device__ vec3 estimateNormal(vec3 ref_pos, uint obj)
{
    vec4 refP = vec4(ref_pos, 1.0);
    refP = Objects[obj].transformation * refP;
    vec3 ref_point = vec3(refP) / refP.w;
    if (Objects[obj].shape == Shape::Sphere)
    {
        vec3 normal_dir = normalize(
                vec3((Objects[obj].normal_transformation * vec4(normalize(ref_point), 0.0))));
        return normal_dir;
    } else
    {
        vec2 cylinder_r_h = Objects[obj].dims;
        if (ref_point.y < 0.0)
        {
            if (abs(length(vec2(ref_point.x, ref_point.z)) - cylinder_r_h.x) >= EPSILON)
                return normalize(vec3(Objects[obj].normal_transformation * vec4(0.0, -1.0, 0.0, 0.0)));
            else
                return normalize(
                        vec3(Objects[obj].normal_transformation * vec4(ref_point.x, 0.0, ref_point.z, 0.0)));
        } else
        {
            if (abs(length(vec2(ref_point.x, ref_point.z)) - cylinder_r_h.x) >= EPSILON)
                return normalize(vec3(Objects[obj].normal_transformation * vec4(0.0, 1.0, 0.0, 0.0)));
            else
                return normalize(
                        vec3(Objects[obj].normal_transformation * vec4(ref_point.x, 0.0, ref_point.z, 0.0)));
        }
    }
}

__device__ vec3 PhongLighting(vec3 L, vec3 N, vec3 V, bool inShadow,
                              uint materialID, int lightIdx)
{
    if (inShadow)
    {
        return Lights[lightIdx].ambient * Materials[materialID].ambient;
    } else
    {
        vec3 R = reflect(-L, N);
        float N_dot_L = max(0.0, dot(N, L));
        float R_dot_V = max(0.0, dot(R, V));
        float R_dot_V_pow_n = (R_dot_V == 0.0) ? 0.0 : pow(R_dot_V, Materials[materialID].specular);
        return Lights[lightIdx].ambient * Materials[materialID].ambient +
               Lights[lightIdx].source *
               (Materials[materialID].diffuse * N_dot_L + Materials[materialID].reflect * R_dot_V_pow_n);
    }
}

__device__ vec3
castRay(const Ray* ray, const int preObj,
        const float near,
        const float far,
        bool* hasHit,
        vec3* hitPos,
        vec3* hitNormal,
        vec3* reflectDecay, int* hitObj)
{
    int objIndex;
    float dist = shortestDistanceToSurface(ray->origin, ray->direction, near, far, preObj, far, &objIndex);
    if (dist > far - EPSILON)
    {
        *hasHit = false;
        return vec3(0.5); // TODO
    } else
    {
        *hitObj = objIndex;
        *hasHit = true;
        vec3 ref_pos = ray->origin + dist * ray->direction;
        *hitPos = ref_pos;
        *hitNormal = estimateNormal(ref_pos, *hitObj);
        *reflectDecay = Materials[Objects[objIndex].material_id].reflect_decay;
        vec3 localColor = vec3(0.0);
        // shadow ray
        for (int lightIdx = 0; lightIdx < light_num; lightIdx++)
        {
            vec3 shadowRay = Lights[lightIdx].position - (*hitPos);
            Ray sRay = {*hitPos, normalize(shadowRay)};
            float max_dist = far;
            int hitObjIndex;
            float distTemp = shortestDistanceToSurface(sRay.origin, sRay.direction, EPSILON, max_dist, objIndex,
                                                       far,
                                                       &hitObjIndex);
            bool hitSth = (distTemp < max_dist - EPSILON);
            localColor += PhongLighting(sRay.direction, *hitNormal, -ray->direction, hitSth,
                                        Objects[objIndex].material_id,
                                        lightIdx);
        }
        return localColor;
    }
}

__device__ vec3 shade(const Ray* ray, const float near, const float far,
                      const int ray_marching_level)
{
    Ray nextRay = {ray->origin, ray->direction};
    vec3 colorResult = vec3(0.0);
    vec3 compoundedGlobalReflectDecayCoef = vec3(1.0);
    int preObj = -1;
    for (int i = 0; i < ray_marching_level; i++)
    {
        bool hasHit = false;
        vec3 hitPos, hitNormal, reflectDecay;
        vec3 localColor = castRay(&nextRay, preObj, near, far,
                                  &hasHit, &hitPos, &hitNormal, &reflectDecay, &preObj);
        colorResult += compoundedGlobalReflectDecayCoef * localColor;
        if (!hasHit)
            break;
        compoundedGlobalReflectDecayCoef *= reflectDecay;
        nextRay.origin = hitPos;
        nextRay.direction = normalize(reflect(nextRay.direction, hitNormal));
    }
    return colorResult;
}

__global__ void
renderer(const unsigned int random_seed, const Camera camera, const CameraConfig cameraConfig, const vec2 window_size,
         const float z,
         const int lightNum,
         const int objNum,
         const int ray_marching_level,
         vec3* output_colors)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int width = (int) window_size.x;
    int height = (int) window_size.y;
    if (x >= width || y >= height)
        return;

    float near = cameraConfig.config.x;
    float far = cameraConfig.config.y;
    light_num = lightNum;
    obj_num = objNum;
    vec2 coord_sc = vec2(x - window_size.x / 2.0f + 0.5f, y - window_size.y / 2.0f + 0.5f);
    vec3 rayDir_ec = normalize(vec3(coord_sc, -z));
    vec3 rayDir_wc = normalize(vec3(camera.look_at_mat * vec4(rayDir_ec, 0.0)));
    Ray primary = {camera.position, rayDir_wc};
    vec3 colorResult_f = shade(&primary, near, far, ray_marching_level);
    vec3 colorResult = max(min(colorResult_f * 255.f, vec3(255.f)), vec3(0.f)); // convert to [0-255]
    output_colors[y * width + x] = colorResult;
}


const unsigned int WINDOW_WIDTH = 512;
const unsigned int WINDOW_HEIGHT = 512;
const unsigned int MAX_BLOCK_SIZE = 32;

unsigned int ceil_div(unsigned int dividee, unsigned int devider)
{
    if (dividee % devider == 0)
        return dividee / devider;
    else
        return dividee / devider + 1;
}

dim3 getGridSize()
{
    return dim3(ceil_div(WINDOW_WIDTH, MAX_BLOCK_SIZE), ceil_div(WINDOW_HEIGHT, MAX_BLOCK_SIZE));
}

#include "CImg.h"
#include "util_funcs.hpp"

//FIXME: need debugging! seems the kernel does not run
//TODO: refactor
void run()
{
    cimg_library::CImg<unsigned char> image(WINDOW_WIDTH, WINDOW_HEIGHT, 1, 3);
    Scene scene = setupScene();
    unsigned int ray_marching_level = 2;
    Camera camera = {vec3(0.0, 0.0, -6.0), vec3(0.0, 1.0, 0.0), vec3(0.0),
                     lookAt(vec3(0.0, 0.0, -6.0), vec3(0.0), vec3(0.0, 1.0, 0.0))};

    CameraConfig cameraConfig = {vec3(0.01, 100.0, glm::radians(90.0))};
    float z = WINDOW_HEIGHT / tan(cameraConfig.config.z / 2.0);

    cudaMemcpyToSymbol(Lights, &scene.lights[0], sizeof(Light) * scene.getLightNum());
    cudaMemcpyToSymbol(Objects, &scene.objects[0], sizeof(Object) * scene.getObjNum());
    cudaMemcpyToSymbol(Materials, &scene.materials[0], sizeof(Material) * scene.getMaterialNum());


    dim3 dimGrid = getGridSize();
    dim3 dimBlock(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
    auto image_size = sizeof(vec3) * WINDOW_WIDTH * WINDOW_HEIGHT;
    vec3* output_d;
    cudaMalloc(&output_d, image_size);
    renderer <<< dimGrid, dimBlock>>>(1, camera, cameraConfig, vec2(WINDOW_WIDTH, WINDOW_HEIGHT), z,
                                      scene.getLightNum(),
                                      scene.getObjNum(),
                                      ray_marching_level,
                                      output_d);

    vec3* output_h = new vec3[WINDOW_WIDTH * WINDOW_HEIGHT];
    cudaMemcpy(output_h, output_d, image_size, cudaMemcpyDeviceToHost);
    for (int y = 0; y < WINDOW_HEIGHT; y++)
    {
        int base = y * WINDOW_WIDTH;
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {
            int idx = base + x;
            *image.data(x, y, 0, 0) = (unsigned char) output_h[idx].r;
            *image.data(x, y, 0, 1) = (unsigned char) output_h[idx].g;
            *image.data(x, y, 0, 2) = (unsigned char) output_h[idx].b;
        }
    }
    cimg_library::CImgDisplay inputImageDisplay(image, "Marathon on CUDA");
    while (!inputImageDisplay.is_closed())
    {
        inputImageDisplay.wait();
        image.display(inputImageDisplay);
        std::cout << "Refreshed" << std::endl;
    }
    delete[] output_h;
    cudaFree(output_d);
}