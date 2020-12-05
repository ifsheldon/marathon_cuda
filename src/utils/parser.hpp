//
// Created by Maple on 2020/12/5.
//

#ifndef MARATHON_CUDA_PARSER_HPP
#define MARATHON_CUDA_PARSER_HPP

#include "jsonxx/json.hpp"
#include "scene.hpp"
#include <string>
#include <sstream>
#include <iostream>

bool parse_scene_raw(const Scene &scene, const char* path)
{
    using namespace jsonxx;
    using namespace std;
    json j;

    j["shape_definitions"] = {{"Sphere",   Shape::Sphere},
                              {"Cylinder", Shape::Cylinder}};
    j["lights"] = {};
    j["light_num"] = scene.getLightNum();
    for (int i = 0; i < scene.getLightNum(); i++)
    {
        ostringstream oss;
        oss << i;
        auto light = scene.lights[i];
        j["lights"] += {
                {"position", {light.position.x, light.position.y, light.position.z}},
                {"ambient",  {light.ambient.r,  light.ambient.g,  light.ambient.b}},
                {"source",   {light.source.r,   light.source.g,   light.source.b}}
        };
    }
    j["materials"] = {};
    j["material_num"] = scene.getMaterialNum();
    for (int i = 0; i < scene.getMaterialNum(); i++)
    {
        ostringstream oss;
        oss << i;
        auto material = scene.materials[i];
        j["materials"] += {
                {"ambient",
                                  {material.ambient.r,       material.ambient.g, material.ambient.b}},
                {"diffuse",
                                  {material.diffuse.r,       material.diffuse.g, material.diffuse.b}},
                {"reflect",
                                  {material.reflect.r,       material.reflect.g, material.diffuse.b}},
                {"reflect_decay", {material.reflect_decay.r, material.reflect_decay.g,
                                                                                 material.reflect_decay.b}},
                {"specular",      material.specular}
        };

    }
    j["objects"] = {};
    j["object_num"] = scene.getObjNum();
    for (int i = 0; i < scene.getObjNum(); i++)
    {
        ostringstream oss;
        oss << i;
        auto object = scene.objects[i];
        j["objects"] += {
                {"shape",       (int) object.shape},
                {"dimensions",  {object.dims.x,               object.dims.y}},
                {"transformation",
                                {object.transformation[0][0], object.transformation[0][1],
                                        object.transformation[0][2], object.transformation[0][3],
                                        object.transformation[1][0], object.transformation[1][1],
                                        object.transformation[1][2], object.transformation[1][3],
                                        object.transformation[2][0], object.transformation[2][1],
                                        object.transformation[2][2], object.transformation[2][3],
                                        object.transformation[3][0], object.transformation[3][1],
                                        object.transformation[3][2], object.transformation[3][3]}},
                {"material_id", object.material_id}
        };
    }
    std::cout << j.dump(2, ' ') << std::endl;
    std::ofstream ofs(path);
    if (!ofs.is_open())
        return false;
    ofs << j.dump(2, ' ') << std::endl;
    ofs.flush();
    ofs.close();
    return true;
}

bool parse_scene_readable(const Scene &scene, const char* path)
{
    return false;
}

Scene* read_scene_from_json(const char* path)
{
    using namespace std;
    using namespace jsonxx;
    ifstream ifs(path);
    if (!ifs.is_open())
        return nullptr;

    Scene* scene = new Scene();
    json j;
    ifs >> j;
    int sphere_code = j["shape_definitions"]["Sphere"].as_int();
    int cylinder_code = j["shape_definitions"]["Cylinder"].as_int();

    int light_num = j["light_num"].as_int();
    int material_num = j["material_num"].as_int();
    int obj_num = j["object_num"].as_int();
    auto &lights = j["lights"].as_array();
    for (int i = 0; i < light_num; i++)
    {
        auto &light = lights[i].as_object();
        auto &amb = light.at("ambient");
        auto &pos = light.at("position");
        auto &src = light.at("source");
        vec3 ambient = vec3(amb[0], amb[1], amb[2]);
        vec3 position = vec3(pos[0], pos[1], pos[2]);
        vec3 source = vec3(src[0], src[1], src[2]);
        scene->addLight(position, ambient, source);
    }
    auto &materials = j["materials"].as_array();
    for (int i = 0; i < obj_num; i++)
    {
        auto &mat = materials[i].as_object();
        auto &amb = mat.at("ambient");
        auto &dif = mat.at("diffuse");
        auto &ref = mat.at("reflect");
        auto &dec = mat.at("reflect_decay");
        float spec = mat.at("specular");
        scene->addMaterial(vec3(amb[0], amb[1], amb[2]),
                           vec3(dif[0], dif[1], dif[2]),
                           vec3(ref[0], ref[1], ref[2]),
                           vec3(dec[0], dec[1], dec[2]),
                           spec);
    }
    auto &objects = j["objects"].as_array();
    for (int i = 0; i < obj_num; i++)
    {
        auto &obj = objects[i].as_object();
        int shape_id = obj.at("shape");
        auto &dims = obj.at("dimensions");
        auto &trans = obj.at("transformation");
        int mat_id = obj.at("material_id");
        mat4 transformation = mat4(trans[0].as_float(), trans[1].as_float(), trans[2].as_float(), trans[3].as_float(),
                                   trans[4].as_float(), trans[5].as_float(), trans[6].as_float(), trans[7].as_float(),
                                   trans[8].as_float(), trans[9].as_float(), trans[10].as_float(), trans[11].as_float(),
                                   trans[12].as_float(), trans[13].as_float(), trans[14].as_float(),
                                   trans[15].as_float());
        if (shape_id == sphere_code)
            scene->addSphere(dims[0], mat_id, transformation);
        else
            scene->addCylinder(dims[0], dims[1], mat_id, transformation);
    }

    ifs.close();
    return scene;
}

#endif //MARATHON_CUDA_PARSER_HPP
