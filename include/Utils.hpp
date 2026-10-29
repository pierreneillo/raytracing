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
#include <glm/gtc/matrix_inverse.hpp>
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
#include <unordered_map>
#include <vector>

#include "raytracer.cuh"

#include "particles_calculation.cuh"

// *************
// *  Globals  *
// *************

#define PARTICLE_SPHERE_RADIUS .25

enum resize_policy {
  EXTEND,
  CLAMP,
  PREFER_WIDTH,
  PREFER_HEIGHT,
};
extern resize_policy policy;

extern float TARGET_FPS;

extern GLFWwindow *window;
extern GLuint program;

// Describe wether the user is moving the camera
extern bool moving;
// Position of the mouse at the start of the move
extern glm::vec2 moveOrigin;

extern bool playing;

// ***************************
// * Multi-buffer management *
// ***************************

// Multi buffering for greater performances, we function with a minimum of 3
// buffers :
// - 2 buffers for CUDA (one to read from, one to write to, we need to keep the
// old values because they have not been drawn yet)
// - 1 buffer for OpenGL (to render)
extern ind N_BUFFERS;
extern ind current_read_buffer;

inline ind getReadInd() { return current_read_buffer; }

inline ind getWriteInd() { return (current_read_buffer + 1) % N_BUFFERS; }

inline ind getDrawInd() { return (current_read_buffer - 1) % N_BUFFERS; }

inline void incrementBufferInd() {
  current_read_buffer = (current_read_buffer + 1) % N_BUFFERS;
}

// ********************
// * Helper functions *
// ********************

extern bool recording;

// Global pointer for the pipe
extern FILE *ffmpegPipe;

void startRecording(int width, int height);

// Inside the main loop, after glClear and rendering particles
void recordFrame(int width, int height);

TriangleMesh loadMesh(std::string filename, glm::mat4 transform = glm::mat4());

index_t initFetcher(const std::string &filename);

real3 *fetchAndTransformPositions(real scale, real3 translation);

bool fetchAndTransformPositionsInPlace(real3 *positions, real scale,
                                       real3 translation);

void closeFetcher();

Material basicColoredMat(real3 color);

#endif // UTILS_H