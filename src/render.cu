//
// Created by Maple on 2020/12/4.
//

#include "render.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

struct Ray
{
    vec3 origin;
    vec3 direction;
};

struct Scene_d
{
    const Light* __restrict__ lights;
    const Material* __restrict__ materials;
    const Object* __restrict__ objects;
    const int light_num;
    const int obj_num;
};

using namespace glm;

__device__ const float EPSILON = 0.001;
__device__ const int MAX_MARCHING_STEPS = 255;
__device__ const int MAX_OBJ_NUM = 50;

__device__ float sdSphere(vec3 ref_pos, float s)
{
    return length(ref_pos) - s;
}

__device__ float sdCylinder(vec3 p, float r, float h)
{
    vec2 d = abs(vec2(length(vec2(p.x, p.z)), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}

__device__ float calcDist(const Scene_d* __restrict__ scene, vec3 ref_point, uint objIdx, float far)
{
    vec4 refP = vec4(ref_point, 1.0);
    refP = scene->objects[objIdx].transformation * refP;
    ref_point = vec3(refP) / refP.w;
    switch (scene->objects[objIdx].shape)
    {
        case Shape::Sphere:
            return sdSphere(ref_point, scene->objects[objIdx].dims.x);
        case Shape::Cylinder:
            return sdCylinder(ref_point, scene->objects[objIdx].dims.x, scene->objects[objIdx].dims.y);
        default :
            return far;
    }
}

__device__ float unionSDF(const Scene_d* __restrict__ scene, float* distances, int* objIdx)
{
    float min_dist = distances[0];
    *objIdx = 0;
    for (int i = 0; i < scene->obj_num; i++)
    {
        if (distances[i] < min_dist)
        {
            min_dist = distances[i];
            *objIdx = i;
        }
    }
    return min_dist;
}

__device__ void sceneSDF(const Scene_d* __restrict__ scene, vec3 ref_point, float* distances, float far)
{
    for (uint i = 0; i < scene->obj_num; i++)
    {
        distances[i] = calcDist(scene, ref_point, i, far);
    }
}

__device__ float
shortestDistanceToSurface(const Scene_d* __restrict__ scene, vec3 eye, vec3 marchingDirection, float start_dist,
                          float limit_dist,
                          int preObj,
                          float far,
                          int* objectIndex)
{
    float depth = start_dist;
    float distances[MAX_OBJ_NUM] = {0.f};
    for (int i = 0; i < MAX_MARCHING_STEPS; i++)
    {
        sceneSDF(scene, eye + depth * marchingDirection, distances, far);
        if (preObj != -1)
            distances[preObj] = 2.0f * far;
        int hitObjIdx;
        float dist = unionSDF(scene, distances, &hitObjIdx);
        if (dist < EPSILON)
        {
            *objectIndex = hitObjIdx;
            return depth;
        }
        depth += dist;
        if (depth >= limit_dist)
        {
            *objectIndex = scene->obj_num;
            return limit_dist;
        }
    }
    return limit_dist;
}

__device__ vec3 estimateNormal(const Scene_d* __restrict__ scene, vec3 ref_pos, uint obj)
{
    vec4 refP = vec4(ref_pos, 1.0);
    refP = scene->objects[obj].transformation * refP;
    vec3 ref_point = vec3(refP) / refP.w;
    if (scene->objects[obj].shape == Shape::Sphere)
    {
        vec3 normal_dir = normalize(
                vec3((scene->objects[obj].normal_transformation * vec4(normalize(ref_point), 0.0))));
        return normal_dir;
    } else
    {
        vec2 cylinder_r_h = scene->objects[obj].dims;
        if (ref_point.y < 0.0)
        {
            if (abs(length(vec2(ref_point.x, ref_point.z)) - cylinder_r_h.x) >= EPSILON)
                return normalize(vec3(scene->objects[obj].normal_transformation * vec4(0.0, -1.0, 0.0, 0.0)));
            else
                return normalize(
                        vec3(scene->objects[obj].normal_transformation * vec4(ref_point.x, 0.0, ref_point.z, 0.0)));
        } else
        {
            if (abs(length(vec2(ref_point.x, ref_point.z)) - cylinder_r_h.x) >= EPSILON)
                return normalize(vec3(scene->objects[obj].normal_transformation * vec4(0.0, 1.0, 0.0, 0.0)));
            else
                return normalize(
                        vec3(scene->objects[obj].normal_transformation * vec4(ref_point.x, 0.0, ref_point.z, 0.0)));
        }
    }
}

__device__ vec3 PhongLighting(const Scene_d* __restrict__ scene, vec3 L, vec3 N, vec3 V, bool inShadow,
                              uint materialID, int lightIdx)
{
    if (inShadow)
    {
        return scene->lights[lightIdx].ambient * scene->materials[materialID].ambient;
    } else
    {
        vec3 R = reflect(-L, N);
        float N_dot_L = max(0.0, dot(N, L));
        float R_dot_V = max(0.0, dot(R, V));
        float R_dot_V_pow_n = (R_dot_V == 0.0) ? 0.0 : pow(R_dot_V, scene->materials[materialID].specular);
        return scene->lights[lightIdx].ambient * scene->materials[materialID].ambient +
               scene->lights[lightIdx].source *
               (scene->materials[materialID].diffuse * N_dot_L + scene->materials[materialID].reflect * R_dot_V_pow_n);
    }
}

__device__ vec3
castRay(const Ray* ray, const Scene_d* __restrict__ scene, const int preObj,
        const float near,
        const float far,
        bool* hasHit,
        vec3* hitPos,
        vec3* hitNormal,
        vec3* reflectDecay, int* hitObj)
{
    int objIndex;
    float dist = shortestDistanceToSurface(scene, ray->origin, ray->direction, near, far, preObj, far, &objIndex);
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
        *hitNormal = estimateNormal(scene, ref_pos, *hitObj);
        *reflectDecay = scene->materials[scene->objects[objIndex].material_id].reflect_decay;
        vec3 localColor = vec3(0.0);
        // shadow ray
        for (int lightIdx = 0; lightIdx < scene->light_num; lightIdx++)
        {
            vec3 shadowRay = scene->lights[lightIdx].position - (*hitPos);
            Ray sRay = {*hitPos, normalize(shadowRay)};
            float max_dist = far;
            int hitObjIndex;
            float distTemp = shortestDistanceToSurface(scene, sRay.origin, sRay.direction, EPSILON, max_dist, objIndex,
                                                       far,
                                                       &hitObjIndex);
            bool hitSth = (distTemp < max_dist - EPSILON);
            localColor += PhongLighting(scene, sRay.direction, *hitNormal, -ray->direction, hitSth,
                                        scene->objects[objIndex].material_id,
                                        lightIdx);
        }
        return localColor;
    }
}

__device__ vec3 shade(const Ray* ray, const Scene_d* __restrict__ scene, const float near, const float far,
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
        vec3 localColor = castRay(&nextRay, scene, preObj, near, far, &hasHit, &hitPos, &hitNormal, &reflectDecay,
                                  &preObj);
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
         const Light* __restrict__ lights,
         const int light_num,
         const Material* __restrict__ materials,
         const Object* __restrict__ objects,
         const int obj_num,
         const int ray_marching_level,
         vec3* output_colors)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int width = (int) window_size.x;
    int height = (int) window_size.y;
    if (x >= width || y >= height)
    {
        return;
    }
    float near = cameraConfig.config.x;
    float far = cameraConfig.config.y;
    Scene_d scene = {lights, materials, objects, light_num, obj_num};
    vec2 coord_sc = vec2(x - window_size.x / 2.0f + 0.5f, y - window_size.y / 2.0f + 0.5f);
    vec3 rayDir_ec = normalize(vec3(coord_sc, -z));
    vec3 rayDir_wc = normalize(vec3(camera.look_at_mat * vec4(rayDir_ec, 0.0)));
    Ray primary = {camera.position, rayDir_wc};
    vec3 colorResult_f = shade(&primary, &scene, near, far, ray_marching_level);
    vec3 colorResult = max(min(colorResult_f * 255.f, vec3(255.f)), vec3(0.f)); // convert to [0-255]
    output_colors[y * width + x] = colorResult;
}