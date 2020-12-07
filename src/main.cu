#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
#include "CImg.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "scene.hpp"
#include "render.cuh"
#include "CImg.h"
#include "util_funcs.hpp"
#include "helper_cuda.h"
#include "parser.hpp"

#define M_PI      3.14159265358979323846
#define M_PI_2    1.57079632679489661923

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


const unsigned int WINDOW_WIDTH = 512;
const unsigned int WINDOW_HEIGHT = 512;
const unsigned int MAX_BLOCK_SIZE = 16;
const unsigned int RM_LEVEL = 2;
const unsigned int DEFAULT_SSR = 0;
const unsigned int MAX_SSR = 5;

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

extern __constant__ Light Lights[];
extern __constant__ Object Objects[];
extern __constant__ Material Materials[];

Mode mode = Mode::Orbit;
Camera default_camera = {vec3(0.0, 0.0, -6.0), vec3(0.0, 1.0, 0.0), vec3(0.0),
                         lookAt(vec3(0.0, 0.0, -6.0), vec3(0.0), vec3(0.0, 1.0, 0.0))};
Camera camera = default_camera;
CameraConfig cameraConfig = {vec3(0.01, 100.0, glm::radians(90.0))};
vec3 original_polar_coords = vec3(0.0, M_PI_2, 6.0);
vec3 polar_coords = vec3(0.0, M_PI_2, 6.0); // (theta, phi, radius)
unsigned int super_sample_rate = DEFAULT_SSR;

static bool handleKeyboardInput(const cimg_library::CImgDisplay &display)
{
    if (display.is_keyARROWUP() && super_sample_rate < MAX_SSR)
    {
        super_sample_rate++;
        printf("Super sample rate = %d\n", super_sample_rate);
    } else if (display.is_keyARROWDOWN() && super_sample_rate > DEFAULT_SSR)
    {
        super_sample_rate--;
        printf("Super sample rate = %d\n", super_sample_rate);
    } else if (mode == Mode::Orbit)
    {
        if (display.is_key2())
        {
            mode = Mode::Zoom;
            printf("Switched to Zoom mode");
        } else if (display.is_keyA())
        {
            polar_coords.x += glm::radians(5.0);
        } else if (display.is_keyD())
        {
            polar_coords.x -= glm::radians(5.0);
        } else if (display.is_keyW())
        {
            polar_coords.y -= glm::radians(5.0);
            if (polar_coords.y <= 0.0f)
                polar_coords.y = 0.01f;
        } else if (display.is_keyS())
        {
            polar_coords.y += glm::radians(5.0);
            if (polar_coords.y >= M_PI)
                polar_coords.y = M_PI - 0.01;
        } else if (display.is_keyQ())
        {
            polar_coords.z -= 0.05;
            if (polar_coords.z <= 0.f)
                polar_coords.z = 0.01;
        } else if (display.is_keyE())
        {
            polar_coords.z += 0.05;
        } else if (display.is_keyR())
        {
            polar_coords = original_polar_coords;
            camera = default_camera;
        } else
        {
            return false;
        }
        float radius = polar_coords.z;
        float theta = polar_coords.x;
        float phi = polar_coords.y;
        camera.position = vec3(sin(theta) * sin(phi) * radius, cos(phi) * radius, -cos(theta) * radius * sin(phi));
        camera.look_at_mat = lookAt(camera.position, camera.center, camera.up);
    } else if (mode == Mode::Zoom)
    {
        if (display.is_key1())
        {
            mode = Mode::Orbit;
            printf("Switched to Orbit mode");
            return false;
        } else if (display.is_keyZ())
        {
            vec3 focus_dir = -camera.look_at_mat[2];
            camera.position += 0.05f * focus_dir;
        } else if (display.is_keyX())
        {
            vec3 focus_dir = -camera.look_at_mat[2];
            camera.position -= 0.05f * focus_dir;
        } else
        {
            return false;
        }
    }
    return true;
}

#define BENCHMARKING

int main()
{
    if (!queryGPUCapabilitiesCUDA())
        exit(EXIT_FAILURE);
    cimg_library::CImg<unsigned char> image(WINDOW_WIDTH, WINDOW_HEIGHT, 1, 3);
    auto scene_json_file_path = "../test.json";
    Scene* scene = read_scene_from_json(scene_json_file_path);
    if (scene == nullptr)
    {
        cerr << "Unable to open Scene json file at " << scene_json_file_path << endl;
        return EXIT_FAILURE;
    }

    float z = WINDOW_HEIGHT / tan(cameraConfig.config.z / 2.0);

    checkCudaErrors(cudaMemcpyToSymbol(Lights, &scene->lights[0], sizeof(Light) * scene->getLightNum()));
    checkCudaErrors(cudaMemcpyToSymbol(Objects, &scene->objects[0], sizeof(Object) * scene->getObjNum()));
    checkCudaErrors(cudaMemcpyToSymbol(Materials, &scene->materials[0], sizeof(Material) * scene->getMaterialNum()));

    dim3 dimGrid = getGridSize();
    dim3 dimBlock(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
    auto image_size = sizeof(color_u8) * WINDOW_WIDTH * WINDOW_HEIGHT;
    color_u8* output_d;
    checkCudaErrors(cudaMalloc(&output_d, image_size));
    renderer <<< dimGrid, dimBlock>>>(camera, cameraConfig, vec2(WINDOW_WIDTH, WINDOW_HEIGHT), z,
                                      scene->getLightNum(),
                                      scene->getObjNum(),
                                      RM_LEVEL,
                                      scene->background_color,
                                      super_sample_rate,
                                      output_d);

    color_u8* output_h = new color_u8[WINDOW_WIDTH * WINDOW_HEIGHT];
    cudaMemcpy(output_h, output_d, image_size, cudaMemcpyDeviceToHost);
    for (int y = 0, base = 0; y < WINDOW_HEIGHT; y++, base += WINDOW_WIDTH)
    {
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {
            int idx = base + x;
            *image.data(x, y, 0, 0) = output_h[idx].r;
            *image.data(x, y, 0, 1) = output_h[idx].g;
            *image.data(x, y, 0, 2) = output_h[idx].b;
        }
    }
    cimg_library::CImgDisplay inputImageDisplay(image, "Marathon on CUDA");
    while (!inputImageDisplay.is_closed())
    {
        inputImageDisplay.wait();
        if (inputImageDisplay.key())
        {
            bool need_rerender = handleKeyboardInput(inputImageDisplay);
            if (!need_rerender)
                continue;
#ifdef BENCHMARKING
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
#endif
            renderer <<< dimGrid, dimBlock>>>(camera, cameraConfig, vec2(WINDOW_WIDTH, WINDOW_HEIGHT), z,
                                              scene->getLightNum(),
                                              scene->getObjNum(),
                                              RM_LEVEL,
                                              scene->background_color,
                                              super_sample_rate,
                                              output_d);
#ifdef BENCHMARKING
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Took %.2f ms to render one frame, super sample rate = %d\n", milliseconds, super_sample_rate);
            auto start_time = clock();
#endif
            cudaMemcpy(output_h, output_d, image_size, cudaMemcpyDeviceToHost);
            for (int y = 0, base = 0; y < WINDOW_HEIGHT; y++, base += WINDOW_WIDTH)
            {
                for (int x = 0; x < WINDOW_WIDTH; x++)
                {
                    int idx = base + x;
                    *image.data(x, y, 0, 0) = output_h[idx].r;
                    *image.data(x, y, 0, 1) = output_h[idx].g;
                    *image.data(x, y, 0, 2) = output_h[idx].b;
                }
            }
            image.display(inputImageDisplay);
#ifdef BENCHMARKING
            auto end_time = clock();
            printf("Took %ld ms to display\n", end_time - start_time);
#endif
        }
    }
    delete[] output_h;
    delete scene;
    cudaFree(output_d);
    return 0;
}

//int main()
//{
////    Scene s = setupScene();
////    parse_scene_raw(s, "test.json");
//    Scene* scene = read_scene_from_json("../test.json");
//    delete scene;
//    return 0;
//}