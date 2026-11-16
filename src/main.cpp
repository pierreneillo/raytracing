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
  w = width;
  h = height;
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
  static Camera cam =
      Camera(real3(0.2, 0, 4), real3(0, 0, 0), real3(0, 1, 0), 45.f);

  camPtr = &cam;

  RCamera rcam = {cam.eye, cam.at, cam.up, cam.fov, cam.near, cam.far, aspect};

  /******************************
   *         Cornell box         *
   *******************************/

  // The Cornell box is composed of a red wall on the left, a green wall on the
  // right, and 3 white walls on bottom, top and back

  glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(-.2, -1, .1));
  transform =
      glm::rotate(transform, (float)glm::radians(70.), glm::vec3(0, 1., 0));
  transform = glm::scale(transform, glm::vec3(.1));

  Object smallCube = Object(basicColoredMat(real3(1, 1, 1)),
                            loadMesh("data/bunny_big.obj", transform));

  transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.5, -.6, -.3));
  transform =
      glm::rotate(transform, (float)glm::radians(18.), glm::vec3(0, 1., 0));
  transform = glm::scale(transform, glm::vec3(.2, .4, .2));

  Object bigCube = Object(basicColoredMat(real3(1, 1, 1)),
                          loadMesh("data/cube.obj", transform));

#define N_OBJ 7
  Object objects[N_OBJ] = {
      makePlane({-1, 0, 0}, 1, 0,
                basicColoredMat(real3(1, 0, 0))), // Left red plane
      makePlane({1, 0, 0}, 1, 0,
                basicColoredMat(real3(0, 1, 0))), // Right green plane
      makePlane({0, -1, 0}, 1, 1,
                basicColoredMat(real3(1, 1, 1))), // Bottom white plane
      makePlane({0, 1, 0}, 1, 1,
                basicColoredMat(real3(1, 1, 1))), // Top white plane
      makePlane({0, 0, -1}, 1, 2,
                basicColoredMat(real3(1, 1, 1))), // Background white plane
      smallCube,
      bigCube};

  // Temporary setup
#define N_LIGHTS 2
  LightSource lights[N_LIGHTS] = {
      {SUN, real3(0., -1, 0.), .1f, real3(1., 1., 1.)},
      {POINT, real3(0., -.7, 1.3), .3f, real3(1., .7, .2)},

  };

  GLuint vao;
  glCreateVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glfwGetFramebufferSize(window, &w, &h);
  aspect = (float)w / (float)h;

  recording = false;
  TARGET_FPS = 100.f;
  if (recording)
    startRecording(w, h);

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
      recordFrame(w, h);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  closeFetcher();
  end_raytracing();
  if (recording && ffmpegPipe)
    pclose(ffmpegPipe);
  glfwTerminate();

  return 0;
}