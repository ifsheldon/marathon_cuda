#include <iostream>
#include <cuda_runtime.h>
#include "CImg.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "scene.hpp"
#include "render.cuh"
#include "util_funcs.hpp"

using namespace std;

static int max_x_threads = 0;
static int max_share_mem = 0;
static int warp_size = 1;

bool queryGPUCapabilitiesCUDA()
{
    // Device Count
    int devCount;

    // Get the Device Count
    cudaGetDeviceCount(&devCount);

    // Print Device Count
    printf("Device(s): %i\n", devCount);

    // TODO: query anything else you will need
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    std::cout << "Max Block Dim in a Grid (" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", "
              <<
              properties.maxGridSize[2] << ")" << std::endl;
    std::cout << "Max Thread per Block: " << properties.maxThreadsPerBlock << std::endl;
    std::cout << "Max Thread Dim in a Block (" << (max_x_threads = properties.maxThreadsDim[0]) << ", " << properties.
            maxThreadsDim[1] <<
              ", " << properties.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Share Memory per Block (bytes): " << (max_share_mem = properties.sharedMemPerBlock) << std::endl;
    std::cout << "Mem Pitch (bytes) " << properties.memPitch << std::endl;
    std::cout << "Total Constant Memory (bytes): " << properties.totalConstMem << std::endl;
    std::cout << "Warp Size: " << (warp_size = properties.warpSize) << std::endl;
    return devCount > 0;
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


int main()
{
    std::cout << "Hello, World!" << std::endl;
    if (!queryGPUCapabilitiesCUDA())
        exit(EXIT_FAILURE);

    cimg_library::CImg<unsigned char> image(WINDOW_WIDTH, WINDOW_HEIGHT, 1, 3);
    Scene scene = setupScene();
    unsigned int ray_marching_level = 2;
    Camera camera = {vec3(0.0, 0.0, -6.0), vec3(0.0, 1.0, 0.0), vec3(0.0),
                     lookAt(vec3(0.0, 0.0, -6.0), vec3(0.0), vec3(0.0, 1.0, 0.0))};

    CameraConfig cameraConfig = {vec3(0.01, 100.0, glm::radians(90.0))};
    float z = WINDOW_HEIGHT / tan(cameraConfig.config.z / 2.0);

    Light* lights_d;
    cudaMalloc(&lights_d, sizeof(Light) * scene.getLightNum());
    cudaMemcpy(lights_d, &scene.lights[0], scene.getLightNum(), cudaMemcpyHostToDevice);

    Material* materials_d;
    cudaMalloc(&materials_d, sizeof(Material) * scene.getMaterialNum());
    cudaMemcpy(materials_d, &scene.materials[0], scene.getMaterialNum(), cudaMemcpyHostToDevice);

    Object* objects_d;
    cudaMalloc(&objects_d, sizeof(Object) * scene.getObjNum());
    cudaMemcpy(objects_d, &scene.objects[0], scene.getObjNum(), cudaMemcpyHostToDevice);

    dim3 dimGrid = getGridSize();
    dim3 dimBlock(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
    auto mem_size = sizeof(vec3) * WINDOW_WIDTH * WINDOW_HEIGHT;
    vec3* output_d;
    cudaMalloc(&output_d, mem_size);
    renderer <<< dimGrid, dimBlock>>>(1, camera, cameraConfig, vec2(WINDOW_WIDTH, WINDOW_HEIGHT), z,
                                      lights_d, scene.getLightNum(),
                                      materials_d,
                                      objects_d, scene.getObjNum(),
                                      ray_marching_level,
                                      output_d);

    vec3* output_h = new vec3[WINDOW_WIDTH * WINDOW_HEIGHT];
    cudaMemcpy(output_h, output_d, mem_size, cudaMemcpyDeviceToHost);
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
    cimg_library::CImgDisplay inputImageDisplay(image, "RT");
    while (!inputImageDisplay.is_closed())
    {
        inputImageDisplay.wait();
        image.display(inputImageDisplay);
        cout << "Refreshed" << endl;
    }
    delete[] output_h;
    cudaFree(output_d);
    cudaFree(lights_d);
    cudaFree(materials_d);
    cudaFree(objects_d);
    return 0;
}
