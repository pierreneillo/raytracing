#ifndef UTILS_H
#define UTILS_H
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glad/glad.h>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/trigonometric.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "raytracer.cuh"

// *************
// *  Globals  *
// *************

enum resize_policy {
  EXTEND,
  CLAMP,
  PREFER_WIDTH,
  PREFER_HEIGHT,
};
resize_policy policy = EXTEND;

float TARGET_FPS = 30.f;

GLFWwindow *window;
GLuint program;

// Describe wether the user is moving the camera
bool moving = false;
// Position of the mouse at the start of the move
glm::vec2 moveOrigin;

// ********************
// * Helper functions *
// ********************

// Global pointer for the pipe
FILE *ffmpegPipe = nullptr;

void startRecording(int width, int height) {
  // Command to record at 60fps, yuv420p format for compatibility
  std::string cmd =
      "ffmpeg  -y -f rawvideo -pix_fmt rgb24 -s " + std::to_string(width) +
      "x" + std::to_string(height) + " -r " + std::to_string(TARGET_FPS) +
      " -i - -vf \"vflip\" -c:v libx264 -preset fast -crf 18 -pix_fmt "
      "yuv420p output.mp4";

  ffmpegPipe = popen(cmd.c_str(), "w");
}

// Inside the main loop, after glClear and rendering particles
void recordFrame(int width, int height) {
  if (!ffmpegPipe)
    return;

  // Allocate a buffer for the RGB pixel
  static std::vector<unsigned char> pixels(3 * width * height);

  // Read pixels from the OpenGL back buffer
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

  // Write the pixel data to the FFmpeg pipe
  fwrite(pixels.data(), 1, pixels.size(), ffmpegPipe);
}

#endif // UTILS_H