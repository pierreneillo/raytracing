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
#include <unordered_map>

Camera *camPtr = nullptr;

cudaGraphicsResource *current_pos_buffer;

bool windowHasChanged = false;

// We scale the texture to allow anti-aliasing on one end and low res on the
// other
// Scaling factor corresponds to raytraced texture width / actual window width
float scaling_factor = 2;

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

  Object plane = Object(planeMat, {verts, norms, 4, indices, 2}, 2);

  return plane;
}

TriangleMesh loadMesh(std::string filename) {
  // Wrapper to the tinyobjloader
  tinyobj::ObjReaderConfig reader_config;
  reader_config.triangulate = true;
  tinyobj::ObjReader reader;
  TriangleMesh t;
  if (!reader.ParseFromFile(filename, reader_config)) {
    std::cerr << "There was an error while loading mesh '" << filename << "'"
              << std::endl;
    // To prevent error upon mesh loading
    t.nTriangles = 0;
  };

  const tinyobj::attrib_t &attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();

  std::vector<real3> vertices;
  std::vector<real3> normals;
  std::unordered_map<long long, index_t> index_pairs;
  std::vector<index_t> indexes;

  for (auto &shape : shapes)
    for (auto &idx : shape.mesh.indices) {
      // For each vertex of each face, we look at the (vertex, normal) pair and
      long long ids =
          ((long long)idx.normal_index << 32) | (unsigned int)idx.vertex_index;
      if (index_pairs.count(ids) == 0) {
        // We have not yet encountered this vertex
        // We setup the vertex id
        index_pairs[ids] = vertices.size();
        int i = idx.vertex_index;
        int j = idx.normal_index;

        vertices.push_back({
            attrib.vertices[3 * i],
            attrib.vertices[3 * i + 1],
            attrib.vertices[3 * i + 2],
        });
        if (j < 0) {
          normals.push_back({0, 1, 0});
        } else {
          normals.push_back({
              attrib.normals[3 * j],
              attrib.normals[3 * j + 1],
              attrib.normals[3 * j + 2],
          });
        }
      }
      indexes.push_back(index_pairs[ids]);
    }

  // Lastly we copy the data in the designated buffers
  t.nVerts = vertices.size();
  t.vertices = new real3[t.nVerts];
  std::copy(vertices.begin(), vertices.end(), t.vertices);
  t.normals = new real3[t.nVerts];
  std::copy(normals.begin(), normals.end(), t.normals);
  t.triangles = new index_t[indexes.size()];
  std::copy(indexes.begin(), indexes.end(), t.triangles);
  t.nTriangles = indexes.size() / 3;

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

  Material gobeletMaterial = {
      real3(1., 1., 1.), .1, real3(1., 1., 1.), 1, real3(1, 1, 1), 0, 32.,
      PHONG_MATERIAL};

  // Sphere s = {real3(), 1.};
  // Object so = Object(planeMat, &s, 1);
  int nTriangles;
  TriangleMesh gobeletMesh = loadMesh("data/monk_high.obj");
  Object gobelet = Object(gobeletMaterial, gobeletMesh, gobeletMesh.nTriangles);
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
  glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glUseProgram(program);
  glBindTextureUnit(0, texture);

  init_raytracing(rcam, lights, N_LIGHTS, objects, N_OBJ, h_framebuffer);

  while (!glfwWindowShouldClose(window)) {
    rcam.eye = cam.eye;
    if (windowHasChanged) {
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

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  end_raytracing();
  glfwTerminate();

  return 0;
}