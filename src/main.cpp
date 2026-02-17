#include "Camera.hpp"
#include "Error.hpp"
#include "Utils.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../dep/tiny_obj_loader.hpp"
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
  size_t *indices = new size_t[6]{0, 1, 2, 0, 2, 3};

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

  Object plane = Object(planeMat, {verts, norms, 4, indices}, 2);

  return plane;
}

TriangleMesh loadMesh(std::string filename, int &nTriangles) {
  // Wrapper to the tinyobjloader
  tinyobj::ObjReaderConfig reader_config;
  tinyobj::ObjReader reader;

  if (!reader.ParseFromFile(filename, reader_config)) {
    std::cerr << "There was an error while loading mesh '" << filename << "'"
              << std::endl;
    // To prevent error upon mesh loading
    nTriangles = 0;
  };

  const tinyobj::attrib_t &attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();

  TriangleMesh t;
  t.nVerts = attrib.vertices.size() / 3;
  t.vertices = new real3[t.nVerts];
  t.normals = new real3[t.nVerts];
  for (int i = 0; i < t.nVerts; i++) {
    // We copy data
    t.vertices[i] = {
        attrib.vertices[3 * i],
        attrib.vertices[3 * i + 1],
        attrib.vertices[3 * i + 2],
    };
    // We set the normals at 0 as we will need to average over all the faces
    t.normals[i] = {0, 0, 0};
  }

  std::vector<size_t> indexes;
  for (auto &shape : shapes)
    for (auto &idx : shape.mesh.indices) {
      indexes.push_back(idx.vertex_index);
      // We add the normal if there is one
      if (idx.normal_index >= 0) {
        int i = idx.normal_index;
        real3 localNormal = {
            attrib.normals[3 * i],
            attrib.normals[3 * i + 1],
            attrib.normals[3 * i + 2],
        };
        t.normals[idx.vertex_index] += localNormal;
      }
    }

  // We normalize the normals
  for (int i = 0; i < t.nVerts; i++)
    t.normals[i] = normalized(t.normals[i]);

  // Lastly we copy the triangles in the designated buffer
  t.triangles = new size_t[indexes.size()];
  std::copy(indexes.begin(), indexes.end(), t.triangles);
  nTriangles = indexes.size() / 3;

  return t;
}

int main() {
  setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 1);
  setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
  if (initGLX()) {
    return -1;
  };

  // Compile Shaders and create program
  program = createBasicProgram();

  static Camera cam = Camera(real3(0, 1, 2. / glm::atan(glm::radians(22.5))),
                             real3(0, 0, 0), real3(0, 1, 0), 45.f);

  camPtr = &cam;

  RCamera rcam = {cam.eye, cam.at, cam.up, cam.fov, cam.near, cam.far, aspect};

  // Temporary setup
#define N_LIGHTS 2
  LightSource lights[N_LIGHTS] = {
      {real3(0., 3, 0.), 1.f, real3(1., 1., 1.), true},
      {real3(1.5, 1.5, 3.7), .5f, real3(1., 1., .8), true},
      // {real3(.8, -.5, 0.), .5f, real3(1., .8, .3), true},

  };

  Material planeMat = {
      real3(1., 1., 1.), .1, real3(1., 1., 1.), .9, real3(1, 1, 1), .1, 32.,
      PHONG_MATERIAL};

  Material gobeletMaterial = {
      real3(1., 1., 1.), .1, real3(1., 1., 1.), 1, real3(1, 1, 1), 0, 32.,
      PHONG_MATERIAL};

  // Sphere s = {real3(), 1.};
  // Object so = Object(planeMat, &s, 1);
  int nTriangles;
  TriangleMesh gobeletMesh = loadMesh("data/gobelet.obj", nTriangles);
  Object gobelet = Object(gobeletMaterial, gobeletMesh, nTriangles);
#define N_OBJ 2
  Object objects[N_OBJ] = {makePlane({0, 0, 0}, 4, 1, planeMat), gobelet};

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

  init_raytracing(rcam, lights, N_LIGHTS, objects, N_OBJ, h_framebuffer);

  while (!glfwWindowShouldClose(window)) {
    rcam.eye = cam.eye;
    if (windowHasChanged) {
      windowHasChanged = false;
      rcam.aspect = aspect;
      h_framebuffer = (float4 *)realloc(h_framebuffer, sizeof(float4) * w * h);
      glDeleteTextures(1, &texture);
      glCreateTextures(GL_TEXTURE_2D, 1, &texture);
      glTextureStorage2D(texture, 1, GL_RGBA32F, w, h);
      glBindTextureUnit(0, texture);
    }
    updateCamera(rcam);
    float last_t = glfwGetTime();
    raytrace(h_framebuffer);
    float delta = glfwGetTime() - last_t;
    printf("%.2f FPS       \r", 1. / delta);

    glTextureSubImage2D(texture, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT,
                        h_framebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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