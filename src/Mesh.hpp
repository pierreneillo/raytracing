#ifndef _MESH_H_
#define _MESH_H_

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ShaderProgram.hpp"
#include "constants.hpp"

class Mesh {
public:
  Mesh(std::vector<glm::vec3> positions,
       std::vector<unsigned int> triangle_indices)
      : m_vertexPositions(positions), m_triangleIndices(triangle_indices){};

  Mesh(std::vector<glm::vec3> positions, std::vector<glm::vec3> normals,
       std::vector<unsigned int> triangle_indices)
      : m_vertexPositions(positions), m_vertexNormals(normals),
        m_triangleIndices(triangle_indices){};

  Mesh(std::vector<glm::vec3> positions, std::vector<glm::vec3> normals,
       std::vector<glm::vec2> texCoords,
       std::vector<unsigned int> triangle_indices)
      : m_vertexPositions(positions), m_vertexNormals(normals),
        m_vertexTexCoords(texCoords), m_triangleIndices(triangle_indices){};

  ~Mesh();
  void init();
  static Mesh fromOFF(std::string filename);
  static Mesh genSphere(const unsigned int resolution = 32);
  static std::shared_ptr<Mesh> &genPlane() {
    static std::shared_ptr<Mesh> plane = std::make_shared<Mesh>(
        std::vector<glm::vec3>{
            {-1, 1, 0},
            {-1, -1, 0},
            {1, -1, 0},
            {1, 1, 0},
        },
        std::vector<glm::vec3>(4, glm::vec3(0, 0, 1)),
        std::vector<glm::vec2>({
            {0, 1},
            {0, 0},
            {1, 0},
            {1, 1},
        }),
        std::vector<unsigned int>{0, 1, 2, 0, 2, 3});
    return plane;
  };
  const std::vector<unsigned int> &get_triangles() const {
    return m_triangleIndices;
  };
  const std::vector<glm::vec3> &get_positions() const {
    return m_vertexPositions;
  };
  const std::vector<glm::vec3> &get_normals() const { return m_vertexNormals; };
  int get_triangles_size() const { return m_triangleIndices.size(); };
  int get_vertex_count() const { return m_vertexPositions.size(); }

  void computeBoundingSphere(glm::vec3 *center, float *radius);

private:
  std::vector<glm::vec3> m_vertexPositions;
  std::vector<glm::vec3> m_vertexNormals;
  std::vector<glm::vec2> m_vertexTexCoords;
  std::vector<unsigned int> m_triangleIndices;
  GLuint m_posVbo = 0;
  GLuint m_normalVbo = 0;
  GLuint m_texCoordVbo = 0;
  GLuint m_ibo = 0;
  // .
};

class Material {
public:
  virtual ~Material() = default;
  virtual void init(std::shared_ptr<ShaderProgram> shader) = 0;
  virtual void apply(std::shared_ptr<ShaderProgram> shader) = 0;
};

class EmptyMaterial : public Material {
public:
  EmptyMaterial(){};
  void init(std::shared_ptr<ShaderProgram> shader) {};
  void apply(std::shared_ptr<ShaderProgram> shader) {};
};

class PhongMaterial : public Material {
public:
  glm::vec3 ambientColor;
  glm::vec3 diffuseColor;
  glm::vec3 specularColor;

  float kA, kD, kS; // Coefficients
  float shininess;

  PhongMaterial(glm::vec3 color = {1.f, 1.f, 1.f}, float shiny = 32.0f)
      : ambientColor(color), diffuseColor(color),
        specularColor(1.0f, 1.0f, 1.0f), kA(0.2f), kD(0.8f), kS(0.01f),
        shininess(shiny) {}

  void init(std::shared_ptr<ShaderProgram> shader) override {};

  void apply(std::shared_ptr<ShaderProgram> shader) override {
    shader->set("material.type", PHONG_MATERIAL);
    shader->set("material.uses_texture", 0);
    shader->set("material.ambientColor", ambientColor);
    shader->set("material.diffuseColor", diffuseColor);
    shader->set("material.specularColor", specularColor);
    shader->set("material.kA", kA);
    shader->set("material.kD", kD);
    shader->set("material.kS", kS);
    shader->set("material.shininess", shininess);
  }
};

class Object {
public:
  Object(std::shared_ptr<Mesh> t, std::shared_ptr<Material> m)
      : mesh(t), mat(m){};

  // Transforms
  void rotate(float angle, glm::vec3 axis);
  void scale(glm::vec3 s);
  inline void scale(float s) { scale(glm::vec3(s)); };
  void translate(glm::vec3 t);
  inline void setTransform(glm::mat4 t) { transform = t; };
  inline glm::mat4 getTransform() { return transform; };
  inline void reinitTransform() { transform = glm::mat4(1.f); }
  inline void applyInitTransform() { transform *= initial_transform; }
  inline glm::vec3 getPos() {
    return getTransform() * glm::vec4(0., 0., 0., 1.);
  };

  // Graphics
  void init();
  void render(std::shared_ptr<ShaderProgram>);

  std::shared_ptr<Mesh> mesh;
  std::shared_ptr<Material> mat;

private:
  GLuint vao;

  glm::mat4 transform = glm::mat4(1.f);
  // Initial transform stores things such as scale and rotation around one's own
  // axis
  glm::mat4 initial_transform = glm::mat4(1.f);
};

#endif //_MESH_H_