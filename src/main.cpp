#include "Camera.hpp"
#include "Error.hpp"
#include "Utils.hpp"

#include <cstddef>
#include <cstdio>
#include <driver_types.h>
#include <glm/fwd.hpp>
#include <glm/matrix.hpp>

Camera *camPtr = nullptr;

cudaGraphicsResource *current_pos_buffer;

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
}

void printHelp() {
  std::cout << "Help : " << std::endl
            << "\tEsc, W\tQuit the application" << std::endl;
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

int main() {
  setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 1);
  setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
  if (initGLX()) {
    return -1;
  };

  // Compile Shaders and create program
  program = createBasicProgram();

  static Camera cam =
      Camera(real3(0, 0, 1. / glm::atan(glm::radians(22.5))) + .5,
             real3(0, 0, 0), real3(0, 1, 0), 45.f);

  camPtr = &cam;

  RCamera rcam = {cam.eye, cam.at, cam.up, cam.fov, cam.near, cam.far, aspect};

  // Temporary setup
  LightSource lights[3] = {
      {real3(0., 2., 0.), 1.f, real3(1., 1., 1.), true},
      {real3(2., .5, -.2), .5f, real3(1., 1., .8), true},
      {real3(2., -1., 0.), .5f, real3(1., .8, .3), true},

  };

  // Sphere of radius r and center C
  Sphere s = {real3(), 1};
  // Red material for the sphere
  Material sphereMat = {
      real3(1, 0, 0), .1, real3(1, 0, 0), .96, real3(1, 1, 1), .04, 32.,
      PHONG_MATERIAL};
  Material planeMat = {
      real3(1., 1., 1.), .0, real3(1., 1., 1.), .5, real3(1, 1, 1), .7, 32.,
      PHONG_MATERIAL};

  Object st = {
      sphereMat,
      TRIANGLE,
      {.triangleMesh = genSphere(4)},
      8 * 4 * 4,
      {
          {-1, -1, -1},
          {1, 1, 1},
      },

  };
  Object so = {sphereMat, SPHERE, {&s}, 1, {{-1, -1, -1}, {1, 1, 1}}};
  float pS = 3;
  real3 planeVertices[4] = {
      {-pS, -1, -pS},
      {-pS, -1, pS},
      {pS, -1, pS},
      {pS, -1, -pS},
  };
  real3 planeNormals[4] = {
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0},
  };
  size_t planeTris[6] = {0, 1, 2, 0, 2, 3};
  Object plane = {planeMat,
                  TRIANGLE,
                  {.triangleMesh =
                       {
                           planeVertices,
                           planeNormals,
                           4,
                           planeTris,
                       }},
                  2,
                  {{-1, -1, -1}, {1, -1, 1}}};
  Object objects[2] = {so, plane};

  GLuint vao;
  glCreateVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_DEPTH_TEST);

  int iw, ih;
  glfwGetFramebufferSize(window, &iw, &ih);
  w = iw;
  h = ih;
  aspect = (float)w / (float)h;

  float4 *h_framebuffer = (float4 *)malloc(sizeof(float4) * w * h);
  GLuint texture;
  glCreateTextures(GL_TEXTURE_2D, 1, &texture);
  glTextureStorage2D(texture, 1, GL_RGBA32F, w, h);
  glUseProgram(program);
  glBindTextureUnit(0, texture);

  init_raytracing(rcam, lights, 3, objects, 2, h_framebuffer);

  while (!glfwWindowShouldClose(window)) {
    rcam.eye = cam.eye;
    rcam.aspect = aspect;
    updateCamera(rcam);
    std::cout << w << ", " << h << std::endl;
    float last_t = glfwGetTime();
    raytrace(h_framebuffer);
    float delta = glfwGetTime() - last_t;
    std::cout << "Raytracing time : " << delta << std::endl;

    glTextureSubImage2D(texture, 0, 0, 0, iw, ih, GL_RGBA, GL_FLOAT,
                        h_framebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program);
    glBindTextureUnit(0, texture);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  end_raytracing();
  glfwTerminate();

  return 0;
}