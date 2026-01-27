#ifndef MY_CONSTANTS
#define MY_CONSTANTS
#include <glm/glm.hpp>
// constants
const static unsigned int N_SEGMENTS = 32;
const static float kSizeSun = 1;
const static float kSizeEarth = .5;
const static float kSizeMoon = 0.25;

// Movement

const static float kRadOrbitEarth = 10;
const static float kEarthRotationAxisAngle = 0.41;
const static float kRadOrbitMoon = 2;
const static float revolution_speed = 1;
const static float earth_rotation_speed = 20;
const static float moon_rotation_speed = 6;
const static float TIME_UNIT = 8; // Duration (in seconds) of a time unit (year)

const static glm::vec3 UP = {0, 0, 1};

const static glm::vec3 CAMERA_POSITION = {0.0, 20.0, 0.0};
const static glm::vec3 CAMERA_FOCUS = {0.0, 0.0, 0.0};
const static glm::vec3 LIGHT_LOCATION = {0., 0., 0.};

// Materials
enum MaterialType {
  UNDEFINED_MATERIAL,
  PHONG_MATERIAL,
};

#endif