#include "Utils.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>

using namespace glm;

enum CamType { ORTHOGRAPHIC, PERSPECTIVE };

struct Camera {
  explicit Camera(vec3 pos = {0, 0, -1}, vec3 at = {0, 0, 0},
                  vec3 up = vec3({0, 1, 0}), float fov = 45)
      : eye(pos), at(at), up(up), fov(fov) {};
  mat4x4 viewProjMat() {
    switch (t) {
    case ORTHOGRAPHIC:
      std::cout << "Not implemented" << std::endl;
      return perspective(radians(fov), aspect, near, far) * lookAt(eye, at, up);
    case PERSPECTIVE:
      return perspective(radians(fov), aspect, near, far) * lookAt(eye, at, up);
    }
    return perspective(radians(fov), aspect, near, far) * lookAt(eye, at, up);
  }

  void rotate(vec2 r) {
    vec3 direction = eye - at;

    // We rotate around the up vector with r.x angle
    mat4 XRot = glm::rotate(mat4(1.f), -r.x * moving_speed.x, up);
    direction = vec3(XRot * vec4(direction, 1.f));

    // Same symmetrically
    vec3 right = glm::cross(up, direction);
    mat4 YRot = glm::rotate(mat4(1.f), -r.y * moving_speed.y, right);
    direction = vec3(YRot * vec4(direction, 1.f));
    if (abs(glm::dot(up, glm::normalize(direction))) > 1 - EPSILON)
      return;

    eye = at + direction;
  }

  void zoomin(float x) { eye += moving_speed.z * x * (at - eye); }

  vec3 eye;
  vec3 at;
  vec3 up;
  float fov;
  float near = 0.01;
  float far = 100000;
  CamType t = PERSPECTIVE;
  vec3 moving_speed = {.01, .01, .01};
};
