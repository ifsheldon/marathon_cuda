//
// Created by Maple on 2020/12/4.
//

#ifndef MARATHON_CUDA_SCENE_HPP
#define MARATHON_CUDA_SCENE_HPP

#include <vector>
#include "glm/glm.hpp"

enum Shape
{
    Sphere = 1, Cylinder = 2
};

using glm::vec3;
using glm::vec2;
using glm::mat4;

struct Light
{
    vec3 position;
    vec3 ambient;
    vec3 source;
};

struct Material
{
    vec3 ambient;
    vec3 diffuse;
    vec3 reflect;
    vec3 reflect_decay;
    float specular;
};

struct Object
{
    Shape shape;
    vec2 dims;
    mat4 transformation;
    mat4 normal_transformation;
    unsigned int material_id;
};

using std::vector;

class Scene
{
private:
    unsigned int max_light_num;
    unsigned int max_obj_num;
    unsigned int max_material_num;

    static bool validColor(vec3 &color)
    {
        return color.r >= 0.0 && color.g >= 0.0 && color.b >= 0.0;
    }

public:
    vector<Light> lights;
    vector<Material> materials;
    vector<Object> objects;

    explicit Scene(unsigned int max_light_num = 10, unsigned int max_obj_num = 50, unsigned int max_material_num = 50)
    {
        this->max_light_num = max_light_num;
        this->max_obj_num = max_obj_num;
        this->max_material_num = max_material_num;
    }

    unsigned int getMaxLightNum() const
    {
        return max_light_num;
    }

    unsigned int getLightNum() const
    {
        return lights.size();
    }

    unsigned int getMaxObjNum() const
    {
        return max_obj_num;
    }

    unsigned int getObjNum() const
    {
        return objects.size();
    }

    unsigned int getMaxMaterialNum() const
    {
        return max_material_num;
    }

    unsigned int getMaterialNum() const
    {
        return materials.size();
    }


    bool addLight(vec3 position, vec3 ambient, vec3 source_color)
    {
        if (getLightNum() == max_light_num || !validColor(ambient) || !validColor(source_color))
        {
            return false;
        } else
        {
            lights.emplace_back(position, ambient, source_color);
            return true;
        }
    }

    bool addMaterial(vec3 ambient, vec3 diffuse, vec3 reflect, vec3 reflect_decay, float specular)
    {
        if (getMaterialNum() == max_material_num || !validColor(diffuse) || !validColor(reflect) ||
            !validColor(reflect_decay))
        {
            return false;
        } else
        {
            materials.emplace_back(ambient, diffuse, reflect, reflect_decay, specular);
            return true;
        }
    }

    bool addSphere(float radius, unsigned int material_id, mat4 transformation)
    {
        if (getObjNum() == max_obj_num || material_id >= getMaterialNum() || radius <= 0.0)
        {
            return false;
        } else
        {
            objects.emplace_back(Shape::Sphere, vec2(radius), transformation,
                                 glm::transpose(glm::inverse(transformation)), material_id);
            return true;
        }
    }

    bool addCylinder(float radius, float height, unsigned int material_id, mat4 transformation)
    {
        if (getObjNum() == max_obj_num || material_id >= getMaterialNum() || radius <= 0.0 || height <= 0.0)
        {
            return false;
        } else
        {
            objects.emplace_back(Shape::Cylinder, vec2(radius, height), transformation,
                                 glm::transpose(glm::inverse(transformation)), material_id);
            return true;
        }
    }
};


#endif //MARATHON_CUDA_SCENE_HPP
