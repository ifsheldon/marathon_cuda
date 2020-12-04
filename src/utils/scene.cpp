//
// Created by Maple on 2020/12/4.
//

#include "scene.hpp"

bool validColor(vec3 &color)
{
    return color.r >= 0.0 && color.g >= 0.0 && color.b >= 0.0;
}

bool Scene::addLight(vec3 position, vec3 ambient, vec3 source_color)
{
    if (getLightNum() == max_light_num || !validColor(ambient) || !validColor(source_color))
    {
        return false;
    } else
    {
        return true;
    }
}

bool Scene::addMaterial(vec3 ambient, vec3 diffuse, vec3 reflect, vec3 reflect_decay, float specular)
{
    if (getMaterialNum() == max_material_num || !validColor(diffuse) || !validColor(reflect) ||
        !validColor(reflect_decay))
    {
        return false;
    } else
    {
        return true;
    }
}

bool Scene::addSphere(float radius, unsigned int material_id, mat4 transformation)
{
    if (getObjNum() == max_obj_num || material_id >= getMaterialNum() || radius <= 0.0)
    {
        return false;
    } else
    {
        return true;
    }
}

bool Scene::addCylinder(float radius, float height, unsigned int material_id, mat4 transformation)
{
    if (obj_num == getObjNum() || material_id >= getMaterialNum() || radius <= 0.0 || height <= 0.0)
    {
        return false;
    } else
    {
        return true;
    }
}
