#include <iostream>
#include <cuda_runtime.h>
#include "CImg.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "scene.hpp"
#include "render.cuh"

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


int main()
{
    if (!queryGPUCapabilitiesCUDA())
        exit(EXIT_FAILURE);
    run();
    return 0;
}
