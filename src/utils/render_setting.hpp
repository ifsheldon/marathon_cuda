//
// Created by Maple on 2020/12/7.
//

#ifndef MARATHON_CUDA_RENDER_SETTING_HPP
#define MARATHON_CUDA_RENDER_SETTING_HPP

struct RenderSetting
{
    unsigned int ray_marching_level;
    unsigned int super_sample_rate;
    bool first_pass;
    float alpha;
};

#endif //MARATHON_CUDA_RENDER_SETTING_HPP
