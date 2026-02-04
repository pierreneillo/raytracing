#include "ShaderProgram.hpp"

#include <GL/glext.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <ios>

#include "Camera.hpp"
#include "Mesh.hpp"
#include "Scene.hpp"
#include "constants.hpp"
#include "glad/glad.h"

// Create a GPU program i.e., a graphics pipeline
ShaderProgram::ShaderProgram() : _id(glCreateProgram()) {}

ShaderProgram::~ShaderProgram() { glDeleteProgram(_id); }

std::string ShaderProgram::file2String(const std::string &filename) {
  std::ifstream input(filename.c_str());
  if (!input)
    throw std::ios_base::failure(
        "[Shader Program][file2String] Error: cannot open " + filename);
  std::stringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

void ShaderProgram::loadShader(GLenum type, const std::string &shaderFilename) {
  // Loads and compile a shader, before attaching it to a program
  GLuint shader =
      glCreateShader(type); // Create the shader, e.g., a vertex shader to be
                            // applied to every single vertex of a mesh
  std::string shaderSourceString = file2String(
      shaderFilename); // Loads the shader source from a file to a C++ string
  if (shaderSourceString.empty()) {
    std::cerr << "No content in shader " << shaderFilename << std::endl;
    glDeleteShader(shader);
    return;
  }
  const GLchar *shaderSource =
      (const GLchar *)shaderSourceString
          .c_str(); // Interface the C++ string through a C pointer
  glShaderSource(shader, 1, &shaderSource,
                 NULL);    // Load the vertex shader source code
  glCompileShader(shader); // THe GPU driver compile the shader
  GLint compiled;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLsizei len;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
    GLchar *log = new GLchar[len + 1];
    glGetShaderInfoLog(shader, len, &len, log);
    std::cerr << "Compilation error in shader " << shaderFilename << " : "
              << std::endl
              << log << std::endl;
    delete[] log;
    glDeleteShader(shader);
    throw std::runtime_error("Shader compilation failed for " + shaderFilename);
  }
  glAttachShader(_id, shader); // Set the vertex shader as the one ot be used
                               // with the program/pipeline
  glDeleteShader(shader);
}

void ShaderProgram::link() {
  glLinkProgram(_id);
  GLint linked;
  glGetProgramiv(_id, GL_LINK_STATUS, &linked);
  if (!linked) {
    GLsizei len;
    glGetProgramiv(_id, GL_INFO_LOG_LENGTH, &len);
    GLchar *log = new GLchar[len + 1];
    glGetProgramInfoLog(_id, len, &len, log);
    std::cerr << "Linking error in program " << _id << " : " << std::endl
              << log << std::endl;
    delete[] log;
    throw std::runtime_error("Program linking failed");
  }
}

std::shared_ptr<ShaderProgram> ShaderProgram::genBasicShaderProgram(
    const std::string &vertexShaderFilename,
    const std::string &fragmentShaderFilename) {
  std::shared_ptr<ShaderProgram> shaderProgramPtr =
      std::make_shared<ShaderProgram>();
  shaderProgramPtr->loadShader(GL_VERTEX_SHADER, vertexShaderFilename);
  shaderProgramPtr->loadShader(GL_FRAGMENT_SHADER, fragmentShaderFilename);
  shaderProgramPtr->link();
  shaderProgramPtr->use();
  return shaderProgramPtr;
}

std::shared_ptr<ShaderProgram> ShaderProgram::genComputeShaderProgram(
    const std::string &computeShaderFilename) {
  std::shared_ptr<ShaderProgram> shaderProgram =
      std::make_shared<ShaderProgram>();
  shaderProgram->loadShader(GL_COMPUTE_SHADER, computeShaderFilename);
  shaderProgram->link();
  shaderProgram->use();
  return shaderProgram;
}

std::shared_ptr<RayTracer>
RayTracer::genComputeShaderProgram(const std::string &computeShaderFilename) {
  std::shared_ptr<RayTracer> rayTracer = std::make_shared<RayTracer>();
  rayTracer->loadShader(GL_COMPUTE_SHADER, computeShaderFilename);
  rayTracer->link();
  rayTracer->use();
  return rayTracer;
}

GLuint RayTracer::genComputeTextureImage(GLuint w, GLuint h) {
  tex.w = w;
  tex.h = h;
  glGenTextures(1, &tex.id);
  glBindTexture(GL_TEXTURE_2D, tex.id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
  //                 GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
  // glGenerateMipmap(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex.id;
}

GLuint RayTracer::genSSBOs(const Scene scene) {
  struct RTVertex {
    glm::vec4 p;
    glm::vec4 n;
  };
  struct RTMat {
    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;
    glm::vec4 st;
  };
  // We have to bind the 6 SSBOs, but first tightly pack the data
  std::vector<RTVertex> vertexBuffer = {};
  std::vector<unsigned int> triangleBuffer = {};
  std::vector<int> meshBuffer = {};
  std::vector<RTMat> materialBuffer = {};
  std::vector<glm::ivec2> objectBuffer = {};
  std::vector<glm::vec4> lightBuffer;
  std::vector<AABB> AABBBuffer = {};
  std::vector<unsigned int> BVHBuffer = {};

  for (size_t i = 0; i < scene.objects.size(); i++) {
    const auto &obj = scene.objects[i];
    // 1 : Vertices
    std::vector<glm::vec3> vertices = obj->mesh->get_positions();
    std::vector<glm::vec3> normals = obj->mesh->get_normals();
    meshBuffer.push_back(vertices.size()); // nVertices
    // We compute vStart to be able to shift triangles indexes
    unsigned int vStart = vertexBuffer.size();
    meshBuffer.push_back(vStart); // vStart
    for (size_t i = 0; i < vertices.size(); i++) {
      glm::vec3 vertex = vertices[i];
      glm::vec3 normal = normals[i];
      vertexBuffer.push_back({glm::vec4(vertex, 0), glm::vec4(normal, 0)});
    }
    // 2 : Triangles
    std::vector<unsigned int> triangles = obj->mesh->get_triangles();
    meshBuffer.push_back(triangles.size() / 3);      // nTriangles
    meshBuffer.push_back(triangleBuffer.size() / 3); // tStart
    for (auto &idx : triangles) {
      triangleBuffer.push_back(idx + vStart);
    }
    // 7 : Bounding boxes
    AABBBuffer.insert(AABBBuffer.end(), obj->mesh->get_BB().begin(),
                      obj->mesh->get_BB().end());
    // 8 : BVH
    BVHBuffer.insert(BVHBuffer.end(), obj->mesh->get_BVH().begin(),
                     obj->mesh->get_BVH().end());
    // 3 : Meshes
    // Done above
    // 4 : Materials
    std::shared_ptr<PhongMaterial> pmat =
        std::dynamic_pointer_cast<PhongMaterial>(obj->mat);
    if (pmat) {
      materialBuffer.push_back({
          glm::vec4(pmat->ambientColor, pmat->kA),
          glm::vec4(pmat->diffuseColor, pmat->kD),
          glm::vec4(pmat->specularColor, pmat->kS),
          glm::vec4(pmat->shininess, PHONG_MATERIAL, 0, 0),
      });
    } else {
      materialBuffer.push_back({glm::vec4(0.), glm::vec4(0.), glm::vec4(0.),
                                glm::vec4(0.0f, UNDEFINED_MATERIAL, 0, 0)});
    }
    // 5 : Objects
    objectBuffer.insert(objectBuffer.end(), {(int)i, (int)i});
  }
  for (const auto &light : scene.lights) {
    // 6 : Lights
    lightBuffer.push_back(glm::vec4(light.position, light.intensity));
    lightBuffer.push_back(glm::vec4(light.color, light.isActive ? 1. : 0.));
  }
  // Create and bind SSBOs
  GLuint SSBOs[8];
  glGenBuffers(8, SSBOs);
  createBindSSBO(SSBOs[0], vertexBuffer, 1);
  createBindSSBO(SSBOs[1], triangleBuffer, 2);
  createBindSSBO(SSBOs[2], meshBuffer, 3);
  createBindSSBO(SSBOs[3], materialBuffer, 4);
  createBindSSBO(SSBOs[4], objectBuffer, 5);
  createBindSSBO(SSBOs[5], lightBuffer, 6);
  createBindSSBO(SSBOs[6], AABBBuffer, 7);
  createBindSSBO(SSBOs[7], BVHBuffer, 8);

  // Setup uniforms accordingly
  use();
  set("nLights", (int)scene.lights.size());
  set("nTriangles", (int)triangleBuffer.size() / 3);
  set("nObjects", (int)scene.objects.size());
  std::cout << "Number of triangles : " << triangleBuffer.size() / 3
            << std::endl;
  stop();
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  return 0;
}

void RayTracer::setUniforms(std::shared_ptr<Camera> cam) {
  // 1. Get the Camera Model Matrix (Camera -> World)
  // The View Matrix is World -> Camera, so we invert it.
  glm::mat4 viewMatrix = cam->computeViewMatrix();
  glm::mat4 camToWorld = glm::inverse(viewMatrix);

  // 2. Extract vectors directly from the columns of the matrix
  // Column 3 is the Position (Eye)
  glm::vec3 eye = glm::vec3(camToWorld[3]);

  // Column 2 is the "Backward" vector (+Z), so Forward is -Column 2
  glm::vec3 forward = -glm::vec3(camToWorld[2]);

  // Column 1 is the Up vector
  glm::vec3 up = glm::vec3(camToWorld[1]);

  // Column 0 is the Right vector (if you need it)
  glm::vec3 right = glm::vec3(camToWorld[0]);

  // 3. Send to Shader
  use();
  set("w", (int)tex.w);
  set("h", (int)tex.h);
  set("camera.eye", eye);
  set("camera.at", eye + forward); // "At" is Eye + Forward Direction
  set("camera.up", up);
  set("camera.fov", cam->getFov());
  set("camera.near", cam->getNear());
  set("camera.far", cam->getFar());

  stop();
}

void ShaderProgram::dispatchCompute(int access) {
  use();
  {
    glBindTexture(GL_TEXTURE_2D, tex.id);
    {
      glBindImageTexture(0, tex.id, 0, GL_FALSE, 0, access, GL_RGBA32F);
      {
        glDispatchCompute((tex.w + 15) / 16, (tex.h + 15) / 16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                        GL_TEXTURE_FETCH_BARRIER_BIT);
      }
      glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  stop();
}