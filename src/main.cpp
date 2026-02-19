#include "Camera.hpp"
#include "Error.hpp"
#include "Utils.hpp"
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <driver_types.h>
#include <fstream>
#include <glm/fwd.hpp>
#include <glm/matrix.hpp>
#include <string>

Camera *camPtr = nullptr;

cudaGraphicsResource *current_pos_buffer;

bool windowHasChanged = false;
bool shouldSwapToWaterMonk = false;
bool waterMonkActive = false;
bool playSimulation = false;

// We scale the texture to allow anti-aliasing on one end and low res on the
// other
// Scaling factor corresponds to raytraced texture width / actual window width
float scaling_factor = 3;

// *****************
// *  Shader mgmt  *
// *****************

GLuint compileShader(GLenum type, const std::string src) {
  // std::cout << "Compiling from shader source '" << src << "'" << std::endl;
  GLuint s = glCreateShader(type);

  std::ifstream input(src.c_str());
  if (!input)
    throw std::ios_base::failure("Cannot open source");

  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string src_string = buffer.str();
  if (src_string.empty()) {
    std::cerr << "Source is empty" << std::endl;
    glDeleteShader(s);
    return -1;
  }
  const GLchar *src_string_gl = (const GLchar *)src_string.c_str();
  glShaderSource(s, 1, &src_string_gl, nullptr);
  glCompileShader(s);

  GLint ok;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, 1024, nullptr, log);
    printf("Shader error:\n%s\n", log);
  }
  return s;
}

GLuint createBasicProgram() {
  GLuint program = glCreateProgram();
  GLuint vs = compileShader(GL_VERTEX_SHADER, "src/basic.vert");
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, "src/basic.frag");

  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

// ***************
// *  Callbacks  *
// ***************

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
  w = width * scaling_factor;
  h = height * scaling_factor;
  aspect = (float)width / (float)height;
  windowHasChanged = true;
}

void printHelp() {
  std::cout << "Help : " << std::endl
            << "\tEsc, A\tQuit the application" << std::endl
            << "\tW\tReplace monk by water monk" << std::endl
            << "\tSpace\tPlay / Pause water monk simulation" << std::endl;
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mods) {
  if (action == GLFW_PRESS && key == GLFW_KEY_H) {
    printHelp();
  } else if (action == GLFW_PRESS && key == GLFW_KEY_Z) {
    shouldSwapToWaterMonk = true;
  } else if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {
    playSimulation = !playSimulation;
  } else if ((action == GLFW_PRESS && key == GLFW_KEY_A) ||
             (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)) {
    glfwSetWindowShouldClose(window, true);
  }
}

void cursorPositionCallback(GLFWwindow *window, double posx, double posy) {
  if (moving) {
    if (camPtr)
      camPtr->rotate(vec2({posx, posy}) - moveOrigin);
    moveOrigin = {posx, posy};
  }
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    if (action == GLFW_PRESS) {
      moving = true;
      double xpos, ypos;
      glfwGetCursorPos(window, &xpos, &ypos);
      moveOrigin = {xpos, ypos};
    } else if (action == GLFW_RELEASE) {
      moving = false;
    }
  }
}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
  if (camPtr)
    camPtr->zoomin(yoffset);
}

/*Init OpenGL, GLFW and glad*/
int initGLX() {
  // Initialize GLFW
  if (!glfwInit())
    return -1;

  // Setup window with parameters
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Setup debug

  // Create window
  window = glfwCreateWindow(1920, 1080, "Raytracing", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetKeyCallback(window, keyCallback);

  glfwSetCursorPosCallback(window, cursorPositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetScrollCallback(window, scrollCallback);

  // Initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  glDebugMessageCallback(debugMessageCallback, 0);
  return 0;
}

Object makePlane(real3 centerPos, real size, int axis, Material planeMat) {

  // We generate the vertices and normals for a basic quad that would be aligned
  // at y = 0, and of width 2
  real3 *verts = new real3[4]{
      {-1, 0, -1},
      {-1, 0, 1},
      {1, 0, 1},
      {1, 0, -1},
  };
  real3 *norms = new real3[4]{
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0},
  };
  index_t *indices = new index_t[6]{0, 1, 2, 0, 2, 3};

  glm::mat4 model = glm::mat4(1.0f);

  // First translate to the central pos
  model = glm::translate(model, glm::vec3(centerPos));

  // Rotate based on inputted axis
  if (axis == 0)
    model = glm::rotate(model, glm::radians(-sign(centerPos.x) * 90.0f),
                        glm::vec3(0, 0, 1));
  else if (axis == 2)
    model = glm::rotate(model, glm::radians(sign(centerPos.z) * 90.0f),
                        glm::vec3(1, 0, 0));
  else if (centerPos.y > 0)
    model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0, 1, 0));

  model = glm::scale(model, glm::vec3(size));

  glm::mat4x4 normalModel = glm::inverse(glm::transpose(model));
  for (int i = 0; i < 4; i++)
    verts[i] = glm::vec3(model * glm::vec4(glm::vec3(verts[i]), 1.));
  for (int i = 0; i < 4; i++)
    norms[i] = glm::vec3(normalModel * glm::vec4(glm::vec3(norms[i]), 1.));

  Object plane = Object(planeMat, {verts, norms, 4, indices, 2});

  return plane;
}

real3 *initParticles() { return nullptr; }

int main() {
  // We need to explicitly ask for the NVIDIA card
  setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 1);
  setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
  if (initGLX()) {
    return -1;
  };

  // Compile Shaders and create program
  program = createBasicProgram();

  // Setup the camera (which will always stay centered on the same point)
  static Camera cam = Camera(real3(0, 1, 1. / glm::atan(glm::radians(22.5))),
                             real3(0, 1., 0), real3(0, 1, 0), 45.f);

  camPtr = &cam;

  RCamera rcam = {cam.eye, cam.at, cam.up, cam.fov, cam.near, cam.far, aspect};

  // Temporary setup
#define N_LIGHTS 4
  LightSource lights[N_LIGHTS] = {
      {real3(0., 3, 0.), 1.f, real3(1., 1., 1.), true},
      {real3(1.5, 1.5, 3.7), .5f, real3(1., 1., .8), true},
      {real3(2., 1.5, 0.), 1.f, real3(1., .8, .3), true},
      {real3(0., 1.12, 0.015), 1.f, real3(1., .8, .3), true},

  };

  Material planeMat = {
      real3(1., 1., 1.), .1, real3(1., 1., 1.), .9, real3(1, 1, 1), .1, 32.,
      PHONG_MATERIAL};

  Material monkMaterial = {
      real3(.3, .7, 1.), .1, real3(.3, .7, 1.), 1., real3(1, 1, 1), 0., 32.,
      PHONG_MATERIAL};

  int nTriangles;
  TriangleMesh monkMesh = loadMesh("data/monk_high.obj");
  Object monk = Object(monkMaterial, monkMesh);

  // Water monk
  real waterMonkScale = 30.9755;
  real3 waterMonkTranslation = {10.2633, 2, 10.6204};
  index_t nParticles = initFetcher("positions.txt");
  if (nParticles == 0) {
    fprintf(stderr,
            "Could not read particle count from second_positions.txt\n");
    return -1;
  }
  real3 *particleCenters =
      fetchAndTransformPositions(waterMonkScale, waterMonkTranslation);
  if (particleCenters == nullptr) {
    fprintf(stderr, "Could not read initial particle positions\n");
    closeFetcher();
    return -1;
  }
  std::vector<real> radii = std::vector<real>(nParticles, .0081);
  SphereMesh particles = {particleCenters, radii.data(), nParticles};
  Object waterMonk = Object(monkMaterial, particles);
#define N_OBJ 2
  Object objects[N_OBJ] = {makePlane({0, -.06, 0}, 4, 1, planeMat), monk};

  GLuint vao;
  glCreateVertexArrays(1, &vao);
  glBindVertexArray(vao);

  int iw, ih;
  glfwGetFramebufferSize(window, &iw, &ih);
  w = iw * scaling_factor;
  h = ih * scaling_factor;
  aspect = (float)w / (float)h;

  recording = true;
  TARGET_FPS = 100.f;
  if (recording)
    startRecording(iw, ih);

  float4 *h_framebuffer = (float4 *)malloc(sizeof(float4) * w * h);
  Object *dh_objects =
      init_raytracing(rcam, lights, N_LIGHTS, objects, N_OBJ, h_framebuffer);

  // Creating and setting up the texture to render our CUDA framebuffer
  GLuint texture;
  glCreateTextures(GL_TEXTURE_2D, 1, &texture);
  // RGBA texture because the CUDA framebuffer is a float4
  glTextureStorage2D(texture, 1, GL_RGBA32F, w, h);
  // For minification, i.e. anti-aliasing
  glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  // For magnification, i.e. low resolution (we want the texture to look blocky,
  // not out of focus)
  glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  // Finally we bind the texture to our shader program
  glUseProgram(program);
  glBindTextureUnit(0, texture);

  while (!glfwWindowShouldClose(window)) {

    // We update the camera's position at each frame
    rcam.eye = cam.eye;

    if (shouldSwapToWaterMonk && !waterMonkActive) {
      replaceObjectWithSphere(1, waterMonk);
      waterMonkActive = true;
      shouldSwapToWaterMonk = false;
    }

    if (waterMonkActive && playSimulation) {
      if (!fetchAndTransformPositionsInPlace(particleCenters, waterMonkScale,
                                             waterMonkTranslation)) {
        closeFetcher();
        // We try to reopen the simulation, to loop
        if (initFetcher("positions.txt") == nParticles) {
          if (!fetchAndTransformPositionsInPlace(
                  particleCenters, waterMonkScale, waterMonkTranslation)) {
            fprintf(stderr, "Could not stream particle positions\n");
            playSimulation = false;
            break;
          }
        } else {
          fprintf(stderr, "Could not reopen positions.txt\n");
          playSimulation = false;
          break;
        }
      }
      updateSphereObjectAndRebuildBVH(1, particleCenters, nParticles);
    }
    if (windowHasChanged) {
      // If the geometry of the window has changed, we need to reallocate both
      // the texture and the cuda buffer
      windowHasChanged = false;
      rcam.aspect = aspect;
      h_framebuffer = (float4 *)realloc(h_framebuffer, sizeof(float4) * w * h);
      reallocDeviceFramebuffer();
      glDeleteTextures(1, &texture);
      glCreateTextures(GL_TEXTURE_2D, 1, &texture);
      glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureStorage2D(texture, 1, GL_RGBA32F, w, h);
      glBindTextureUnit(0, texture);
    }
    updateCamera(rcam);
    float last_t = glfwGetTime();
    raytrace(h_framebuffer);
    float delta = glfwGetTime() - last_t;

    printf("%.2f FPS       \r", 1. / delta);
    fflush(stdout);

    glTextureSubImage2D(texture, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT,
                        h_framebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    if (recording)
      recordFrame(iw, ih);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  closeFetcher();
  delete[] particleCenters;
  end_raytracing();
  if (recording && ffmpegPipe)
    pclose(ffmpegPipe);
  glfwTerminate();

  return 0;
}