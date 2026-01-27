#ifndef _SCENE_H_
#define _SCENE_H_

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <glm/ext.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Mesh.hpp"

struct Light {
  glm::vec3 position;
  glm::vec3 color;
  float intensity;
  bool isActive;
};
struct Scene {
  // For us, a scene will contain only lights and objects
  // Lights
  std::vector<Light> lights;
  // Objects
  std::vector<std::shared_ptr<Object>> objects = {};

  // Textures
  // GLuint planeNormalMap = 0;
  // GLuint planeColorMap = 0;

  glm::vec3 scene_center = glm::vec3(0);
  float scene_radius = 1.f;

  void render();
};
#endif // _SCENE_H_