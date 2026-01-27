#include "Mesh.hpp"

#include <fstream>

#include "glm/fwd.hpp"
#include "stb_image.h"

Mesh::~Mesh() {
  if (m_posVbo)
    glDeleteBuffers(1, &m_posVbo);
  if (m_normalVbo)
    glDeleteBuffers(1, &m_normalVbo);
  if (m_texCoordVbo)
    glDeleteBuffers(1, &m_texCoordVbo);
  if (m_ibo)
    glDeleteBuffers(1, &m_ibo);
};

void Mesh::init() { // should properly set up the geometry buffer
  // The creation of a vertex array is handled by the Object instance related to
  // this Mesh instance. We simply connect the vbos and ibo to the open vao
  // Generate a GPU buffer to store the positions of the vertices
  if (m_posVbo != 0) {
    // Buffers are already set, we need to rebuild all the bindings though
#ifdef _MY_OPENGL_IS_33_
    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, m_normalVbo);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
    glEnableVertexAttribArray(2);

    if (m_texCoordVbo != 0) {
      glBindBuffer(GL_ARRAY_BUFFER, m_texCoordVbo);
      glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0);
      glEnableVertexAttribArray(3);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
#else
    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, m_normalVbo);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(2);

    if (m_texCoordVbo != 0) {
      glBindBuffer(GL_ARRAY_BUFFER, m_texCoordVbo);
      glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);
      glEnableVertexAttribArray(3);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
#endif
    return;
  }
  size_t vertexBufferSize =
      sizeof(glm::vec3) *
      this->m_vertexPositions
          .size(); // Gather the size of the buffer from the CPU-side vector
#ifdef _MY_OPENGL_IS_33_
  glGenBuffers(1, &this->m_posVbo);
  glBindBuffer(GL_ARRAY_BUFFER, this->m_posVbo);
  glBufferData(GL_ARRAY_BUFFER, vertexBufferSize,
               this->m_vertexPositions.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
  glEnableVertexAttribArray(0);
#else
  glCreateBuffers(1, &this->m_posVbo);
  glBindBuffer(GL_ARRAY_BUFFER, this->m_posVbo);
  glNamedBufferStorage(this->m_posVbo, vertexBufferSize,
                       this->m_vertexPositions.data(), GL_DYNAMIC_STORAGE_BIT);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
  glEnableVertexAttribArray(0);
#endif

#ifdef _MY_OPENGL_IS_33_
  glGenBuffers(1, &this->m_normalVbo);
  glBindBuffer(GL_ARRAY_BUFFER, this->m_normalVbo);
  glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, this->m_vertexNormals.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
  glEnableVertexAttribArray(2);
#else
  glCreateBuffers(1, &this->m_normalVbo);
  glBindBuffer(GL_ARRAY_BUFFER, this->m_normalVbo);
  glNamedBufferStorage(
      this->m_normalVbo, vertexBufferSize, this->m_vertexNormals.data(),
      GL_DYNAMIC_STORAGE_BIT); // Create a data storage on the GPU and fill it
                               // from a CPU array
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
  glEnableVertexAttribArray(2);
#endif

  // Upload texcoords with correct size (vec2)
  size_t texBufferSize = sizeof(glm::vec2) * this->m_vertexTexCoords.size();
  if (texBufferSize > 0) {
#ifdef _MY_OPENGL_IS_33_
  glGenBuffers(1, &this->m_texCoordVbo);
  glBindBuffer(GL_ARRAY_BUFFER, this->m_texCoordVbo);
  glBufferData(GL_ARRAY_BUFFER, texBufferSize, this->m_vertexTexCoords.data(),
               GL_STATIC_DRAW);
  glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0);
  glEnableVertexAttribArray(3);
#else
  glCreateBuffers(1, &this->m_texCoordVbo);
  glBindBuffer(GL_ARRAY_BUFFER, this->m_texCoordVbo);
  glNamedBufferStorage(this->m_texCoordVbo, texBufferSize,
                       this->m_vertexTexCoords.data(), GL_DYNAMIC_STORAGE_BIT);
  glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);
  glEnableVertexAttribArray(3);
#endif
  }

  // Same for an index buffer object that stores the list of indices of the
  // triangles forming the mesh
  size_t indexBufferSize =
      sizeof(unsigned int) * this->m_triangleIndices.size();
#ifdef _MY_OPENGL_IS_33_
  glGenBuffers(1, &this->m_ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->m_ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBufferSize,
               this->m_triangleIndices.data(), GL_STATIC_DRAW);
#else
  glCreateBuffers(1, &this->m_ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->m_ibo);
  glNamedBufferStorage(this->m_ibo, indexBufferSize,
                       this->m_triangleIndices.data(), GL_DYNAMIC_STORAGE_BIT);
#endif

  // We do not deactivate the vao, because material still has to be bound
};

Mesh Mesh::fromOFF(const std::string filename) {
  std::cout << " > Start loading mesh <" << filename << ">" << std::endl;
  std::ifstream in(filename.c_str());
  if (!in)
    throw std::ios_base::failure("[Mesh Loader][loadOFF] Cannot open " +
                                 filename);
  std::string offString;
  unsigned int sizeV, sizeT, tmp;
  in >> offString >> sizeV >> sizeT >> tmp;
  std::vector<glm::vec3> vertices;
  std::vector<glm::uvec3> triangles;
  vertices.resize(sizeV);
  triangles.resize(sizeT);
  size_t tracker = (sizeV + sizeT) / 20;
  std::cout << " > [" << std::flush;
  for (unsigned int i = 0; i < sizeV; ++i) {
    if (i % tracker == 0)
      std::cout << "-" << std::flush;
    in >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
  }
  int s;
  for (unsigned int i = 0; i < sizeT; ++i) {
    if ((sizeV + i) % tracker == 0)
      std::cout << "-" << std::flush;
    in >> s;
    for (unsigned int j = 0; j < 3; ++j)
      in >> triangles[i][j];
  }
  std::cout << "]" << std::endl;
  in.close();
  std::vector<unsigned int> triangles_flattened;
  triangles_flattened.reserve(triangles.size() * 3);

  for (const glm::uvec3 &v : triangles) {
    triangles_flattened.insert(triangles_flattened.end(), {v.x, v.y, v.z});
  };

  std::vector<glm::vec3> normals(vertices.size(), glm::vec3(0.0));
  for (const auto &tri : triangles) {
    glm::vec3 v0 = vertices[tri.x];
    glm::vec3 v1 = vertices[tri.y];
    glm::vec3 v2 = vertices[tri.z];
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 n = glm::cross(edge1, edge2);
    normals[tri.x] += n;
    normals[tri.y] += n;
    normals[tri.z] += n;
  }
  for (auto &n : normals)
    n = glm::normalize(n);

  std::cout << " > Mesh <" << filename << "> loaded" << std::endl;
  return Mesh(vertices, normals, triangles_flattened);
}

Mesh Mesh::genSphere(
    const unsigned int resolution) // should generate a unit sphere
{
  // We interpret resolution as the number of edges on a half circumference
  std::vector<glm::vec3> vpos = {};
  std::vector<glm::vec3> vnorm = {};
  std::vector<glm::vec2> vtexcoords = {};
  std::vector<unsigned int> trInd = {};
  unsigned int latCount = resolution * 2 + 1;
  unsigned int lonCount = resolution * 2 + 1;

  for (unsigned int i = 0; i < latCount; ++i) {
    float phi = (float)i / (float)(latCount - 1) * glm::pi<float>() -
                glm::half_pi<float>();
    float z = glm::sin(phi); // height
    float r = glm::cos(phi); // radius of circle at this latitude
    float V = 1.f - (float)i / (float)(latCount - 1);
    for (unsigned int j = 0; j < lonCount; ++j) {
      float theta = (float)j / (float)(lonCount - 1) * glm::two_pi<float>();
      float x = r * glm::cos(theta);
      float y = r * glm::sin(theta);
      glm::vec3 v = {x, y, z};
      glm::vec3 n = glm::normalize(v);
      vpos.push_back(v);
      vnorm.push_back(n);

      float U = -(float)j / (float)(lonCount - 1);
      vtexcoords.push_back({U, V});
    }
  }

  // Build two triangles per square between latitude/longitude grid cells
  for (unsigned int i = 0; i + 1 < latCount; ++i) {
    for (unsigned int j = 0; j + 1 < lonCount; ++j) {
      unsigned int a = i * lonCount + j;
      unsigned int b = (i + 1) * lonCount + j;
      unsigned int c = i * lonCount + (j + 1);
      unsigned int d = (i + 1) * lonCount + (j + 1);

      trInd.push_back(a);
      trInd.push_back(b);
      trInd.push_back(c);

      trInd.push_back(c);
      trInd.push_back(b);
      trInd.push_back(d);
    }
  }
  return Mesh(vpos, vnorm, vtexcoords, trInd);
};

glm::vec3 min(glm::vec3 a, glm::vec3 b) {
  return glm::vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

glm::vec3 max(glm::vec3 a, glm::vec3 b) {
  return glm::vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

void Mesh::computeBoundingSphere(glm::vec3 *center, float *radius) {
  glm::vec3 mini{}, maxi{};
  for (const auto &p : m_vertexPositions) {
    mini = min(mini, p);
    maxi = max(maxi, p);
  }
  *center = (mini + maxi) / 2.f;
  glm::vec3 amplitude = maxi - mini;
  *radius = std::max(amplitude.x, std::max(amplitude.y, amplitude.z));
}

void Object::translate(glm::vec3 t) {
  transform = glm::translate(transform, t);
}

void Object::scale(glm::vec3 s) { transform = glm::scale(transform, s); };

void Object::rotate(float angle, glm::vec3 axis) {
  transform = glm::rotate(transform, angle, axis);
}

void Object::init() {
  // Create a single handle, vertex array object that contains attributes,
  // vertex buffer objects (e.g., vertex's position, normal, and color)
#ifdef _MY_OPENGL_IS_33_
  glGenVertexArrays(1,
                    &vao); // If your system doesn't support OpenGL 4.5, you
                           // should use this instead of glCreateVertexArrays.
#else
  glCreateVertexArrays(1, &vao);
#endif
  glBindVertexArray(vao);
  mesh->init();
}

void Object::render(std::shared_ptr<ShaderProgram> shader) {

  // Setting uniforms

  // Material uniforms
  mat->apply(shader);

  // Object uniforms
  shader->set("modelMat", transform);
  glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(transform)));
  shader->set("normMat", normalMat);

  glBindVertexArray(vao);
  // Call for rendering: stream the current GPU geometry through the current GPU
  // program
  glDrawElements(GL_TRIANGLES, mesh->get_triangles_size(), GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);
}