#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
#include <random>
#include "CImg.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "scene.hpp"
#include "render.cuh"
#include "util_funcs.hpp"
#include "helper_cuda.h"
#include "parser.hpp"
#include "render_setting.hpp"

#define M_PI      3.14159265358979323846
#define M_PI_2    1.57079632679489661923

using namespace std;

const unsigned int WINDOW_WIDTH = 512;
const unsigned int WINDOW_HEIGHT = 512;
const unsigned int MAX_BLOCK_SIZE = 16;
const unsigned int RM_LEVEL = 2;
const unsigned int DEFAULT_SSR = 0;

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
extern __constant__ vec2 Preturbs[];

Mode mode = Mode::Orbit;
Camera default_camera = {vec3(0.0, 0.0, -6.0), vec3(0.0, 1.0, 0.0), vec3(0.0),
                         lookAt(vec3(0.0, 0.0, -6.0), vec3(0.0), vec3(0.0, 1.0, 0.0))};
Camera camera = default_camera;
CameraConfig cameraConfig = {vec3(0.01, 100.0, glm::radians(90.0))};
vec3 original_polar_coords = vec3(0.0, M_PI_2, 6.0);
vec3 polar_coords = vec3(0.0, M_PI_2, 6.0); // (theta, phi, radius)
RenderSetting renderSetting = {RM_LEVEL, DEFAULT_SSR, true, 0.5};

static bool handleKeyboardInput(const cimg_library::CImgDisplay &display)
{
    if (display.is_keyARROWUP() && renderSetting.super_sample_rate < MAX_SSR)
    {
        renderSetting.super_sample_rate++;
        printf("Super sample rate = %d\n", renderSetting.super_sample_rate);
    } else if (display.is_keyARROWDOWN() && renderSetting.super_sample_rate > DEFAULT_SSR)
    {
        renderSetting.super_sample_rate--;
        printf("Super sample rate = %d\n", renderSetting.super_sample_rate);
    } else if (mode == Mode::Orbit)
    {
        if (display.is_key2())
        {
            mode = Mode::Zoom;
            renderSetting.first_pass = true;
            printf("Switched to Zoom mode");
            return true;
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
            camera = default_camera;
            polar_coords = original_polar_coords;
            renderSetting.first_pass = true;
            return true;
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

bool handleMouseClick(const cimg_library::CImgDisplay &display)
{
    if ((display.button() & 1) && mode == Mode::Zoom) // left mouse button clicked
    {
        float z = WINDOW_HEIGHT / tan(cameraConfig.config.z / 2.0);
        vec2 xy = vec2(display.mouse_x(), WINDOW_HEIGHT - display.mouse_y());
        xy.x -= WINDOW_WIDTH / 2.0;
        xy.y -= WINDOW_WIDTH / 2.0;
        vec3 dir_ec = glm::normalize(vec3(xy, -z));
        vec3 dir_wc = glm::normalize(vec3((camera.look_at_mat * vec4(dir_ec, 0.0))));
        camera.center = camera.position + dir_wc;
        camera.look_at_mat = lookAt(camera.position, camera.center, camera.up);
        renderSetting.first_pass = true;
        return true;
    }
    return false;
}

void copy_to_framebuffer(cimg_library::CImg<unsigned char> &image, size_t image_size, const color_u8* device_src,
                         color_u8* host_dest)
{
    cudaMemcpy(host_dest, device_src, image_size, cudaMemcpyDeviceToHost);
    for (int y = 0, base = 0; y < WINDOW_HEIGHT; y++, base += WINDOW_WIDTH)
    {
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {
            int idx = base + x;
            *image.data(x, y, 0, 0) = host_dest[idx].r;
            *image.data(x, y, 0, 1) = host_dest[idx].g;
            *image.data(x, y, 0, 2) = host_dest[idx].b;
        }
    }
}


inline static float gen_rand()
{
    static std::default_random_engine randomEngine;
    static std::uniform_real_distribution<float> randomGen(0.0, 1.0);
    return randomGen(randomEngine);
}

void gen_random_preturbs_to_device()
{
    static vec2 super_sample_randoms[MAX_SSR * MAX_SSR];
    if (renderSetting.super_sample_rate > 0)
    {
        float grid_size = 1.0f / renderSetting.super_sample_rate;
        for (int i = 0; i < renderSetting.super_sample_rate * renderSetting.super_sample_rate; i++)
            super_sample_randoms[i] = vec2(gen_rand() * grid_size, gen_rand() * grid_size);
        checkCudaErrors(cudaMemcpyToSymbol(Preturbs, super_sample_randoms,
                                           sizeof(vec2) * renderSetting.super_sample_rate *
                                           renderSetting.super_sample_rate));
    }
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
    gen_random_preturbs_to_device();
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
                                      scene->background_color,
                                      renderSetting,
                                      output_d);

    color_u8* output_h = new color_u8[WINDOW_WIDTH * WINDOW_HEIGHT];
    copy_to_framebuffer(image, image_size, output_d, output_h);
    renderSetting.first_pass = false;
    cimg_library::CImgDisplay inputImageDisplay(image, "Marathon on CUDA");
    unsigned int render_count = 0;
    unsigned int max_accumulate_frames = 3;
    while (!inputImageDisplay.is_closed())
    {
        inputImageDisplay.wait();
        bool need_render = false;
        need_render = handleMouseClick(inputImageDisplay);
        if (inputImageDisplay.key())
            need_render = handleKeyboardInput(inputImageDisplay);

        if (need_render || (renderSetting.super_sample_rate > 0 && render_count < max_accumulate_frames))
        {
            if (!need_render)
                render_count++;
            else
                render_count = 0;

            gen_random_preturbs_to_device();
#ifdef BENCHMARKING
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
#endif
            renderer <<< dimGrid, dimBlock>>>(camera, cameraConfig, vec2(WINDOW_WIDTH, WINDOW_HEIGHT), z,
                                              scene->getLightNum(),
                                              scene->getObjNum(),
                                              scene->background_color,
                                              renderSetting,
                                              output_d);
            renderSetting.first_pass = false;
#ifdef BENCHMARKING
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Took %.2f ms to render one frame, super sample rate = %d\n", milliseconds,
                   renderSetting.super_sample_rate);
            auto start_time = clock();
#endif
            copy_to_framebuffer(image, image_size, output_d, output_h);
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