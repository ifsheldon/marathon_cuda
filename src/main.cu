#include <iostream>
#include <cuda_runtime.h>
#include "CImg.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "scene.hpp"

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

int main()
{
    std::cout << "Hello, World!" << std::endl;
    if (!queryGPUCapabilitiesCUDA())
        exit(EXIT_FAILURE);

    cimg_library::CImg<unsigned char> image(WINDOW_WIDTH, WINDOW_HEIGHT, 1, 3);

    return 0;
}
