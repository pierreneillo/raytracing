#include "Mesh.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "glad/glad.h"
#include "glm/fwd.hpp"
#include "stb_image.h"

glm::vec3 cmin(glm::vec3 a, glm::vec3 b) {
  // Component-wise minimum
  return glm::vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

glm::vec3 cmax(glm::vec3 a, glm::vec3 b) {
  // Component-wise maximum
  return glm::vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

Mesh::~Mesh() {
  if (m_posVbo)
    glDeleteBuffers(1, &m_posVbo);
  if (m_normalVbo)
    glDeleteBuffers(1, &m_normalVbo);
  if (m_texCoordVbo)
    glDeleteBuffers(1, &m_texCoordVbo);
  if (m_ibo)
    glDeleteBuffers(1, &m_ibo);
  if (m_bvhbo)
    glDeleteBuffers(1, &m_bvhbo);
};

void Mesh::init(GLuint vao) {
  // should properly set up the geometry buffer
  // The creation of a vertex array is handled by the Object instance related to
  // this Mesh instance. We simply connect the vbos and ibo to the given vao
  // Generate a GPU buffer to store the positions of the vertices
  // We use the 4.5 DSA to prevent all confusion

  // ###################
  // # Buffer creation #
  // ###################

  glCreateBuffers(1, &m_posVbo);
  glNamedBufferData(m_posVbo, sizeof(glm::vec3) * m_vertexPositions.size(),
                    m_vertexPositions.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
  glEnableVertexAttribArray(0);

  glCreateBuffers(1, &m_normalVbo);
  glBindBuffer(GL_ARRAY_BUFFER, m_normalVbo);
  glNamedBufferStorage(
      m_normalVbo, m_vertexNormals.size(), m_vertexNormals.data(),
      GL_DYNAMIC_STORAGE_BIT); // Create a data storage on the GPU and fill it
                               // from a CPU array
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
  glEnableVertexAttribArray(2);

  // Upload texcoords with correct size (vec2)
  size_t texBufferSize = sizeof(glm::vec2) * m_vertexTexCoords.size();
  if (texBufferSize > 0) {
    glCreateBuffers(1, &m_texCoordVbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_texCoordVbo);
    glNamedBufferStorage(m_texCoordVbo, texBufferSize, m_vertexTexCoords.data(),
                         GL_DYNAMIC_STORAGE_BIT);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);
    glEnableVertexAttribArray(3);
  }

  // Upload the Bounding boxes buffer for the BVH
  size_t BBBufferSize = sizeof(glm::vec4) * m_boundingBoxes.size();
  if (BBBufferSize > 0) {
    glCreateBuffers(1, &m_BBVbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_BBVbo);
    glNamedBufferStorage(m_BBVbo, BBBufferSize, m_boundingBoxes.data(),
                         GL_DYNAMIC_STORAGE_BIT);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), 0);
    glEnableVertexAttribArray(4);
  }

  // Same for an index buffer object that stores the list of indices of the
  // triangles forming the mesh
  size_t indexBufferSize = sizeof(unsigned int) * m_triangleIndices.size();

  glCreateBuffers(1, &m_ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
  glNamedBufferStorage(m_ibo, indexBufferSize, m_triangleIndices.data(),
                       GL_DYNAMIC_STORAGE_BIT);

  // And finally, exactly the same for an index buffer object that stores the
  // list of indices of the triangles / bounding boxes forming the BVH
  size_t BVHBufferSize = sizeof(unsigned int) * m_BVH.size();

  glCreateBuffers(1, &m_bvhbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_bvhbo);
  glNamedBufferStorage(m_bvhbo, indexBufferSize, m_BVH.data(),
                       GL_DYNAMIC_STORAGE_BIT);

  // We do not deactivate the vao, because material still has to be bound
  std::cout << "Object::init vao=" << " mesh.posVbo=" << m_posVbo
            << " texVbo=" << m_texCoordVbo << std::endl;

  // Buffers are already set, we need to rebuild all the bindings though
  glVertexArrayVertexBuffer(vao, 0, m_posVbo, 0, sizeof(glm::vec3));
  glEnableVertexArrayAttrib(vao, 0);
  glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
  glVertexArrayAttribBinding(vao, 0, 0);

  glBindBuffer(GL_ARRAY_BUFFER, m_normalVbo);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
  glEnableVertexAttribArray(2);

  glBindBuffer(GL_ARRAY_BUFFER, m_texCoordVbo);
  glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);
  glEnableVertexAttribArray(3);

  glBindBuffer(GL_ARRAY_BUFFER, m_BBVbo);
  glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), 0);
  glEnableVertexAttribArray(4);

  glVertexArrayElementBuffer(vao, m_ibo);
  // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_bvhbo);
};

unsigned int indexTransform(unsigned int i, unsigned int f) { return i; }

void Mesh::genBVH() {
  // Generates the BVH. Should be called upon creation and at each geometry
  // change

  if (m_triangleIndices.empty())
    return;

  // Number of triangles (leafs)
  unsigned int f = m_triangleIndices.size() / 3;
  // Number of internal nodes
  unsigned int n = f - 1;
  // We compute the centroids for each triangle
  std::vector<glm::vec3> centroids(f, glm::vec3(0));
  std::vector<AABB> triangle_bounding_boxes(f, AABB());
  m_BVH.resize(f);
  for (int i = 0; i < f; i++) {
    m_BVH[i] = i;
    glm::vec3 v1 = m_vertexPositions[m_triangleIndices[3 * i]];
    glm::vec3 v2 = m_vertexPositions[m_triangleIndices[3 * i + 1]];
    glm::vec3 v3 = m_vertexPositions[m_triangleIndices[3 * i + 2]];
    triangle_bounding_boxes[i] = AABB(v1, v2, v3);
    centroids[i] = (v1 + v2 + v3) / 3.f;
  }
  m_boundingBoxes.resize(n + f);
  AABB bb = AABB(m_vertexPositions);
  unsigned int max_dim = bb.max_dim();
  // We do an extra simple bottom up approach by :
  // We trade optimisation for efficiency and simplicity, as computed this way,
  // the bounding boxes don't split on largest axis each time
  // 1. Sorting the array of indexes based on their centroids
  std::sort(m_BVH.begin(), m_BVH.end(),
            [&centroids, max_dim](unsigned int a, unsigned int b) {
              return centroids[a][max_dim] < centroids[b][max_dim];
            });
  // 2. Computing the bounding boxes bottom up
  for (int i = n + f - 1; i >= 0; i--) {
    // If i >= n, then we have a leaf and must calculate the bounding
    // box of the associated triangle, otherwise we have an internal node and
    // can compute the bb from children's bb which has necessarily been
    // calculated
    if (i >= n)
      // The index j of the leaf in the m_BVH is j = i - n
      m_boundingBoxes[i] = triangle_bounding_boxes[m_BVH[i - n]];
    else
      m_boundingBoxes[i] =
          AABB::merge(m_boundingBoxes[2 * i], m_boundingBoxes[2 * i + 1]);
  }
}

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

  glCreateVertexArrays(1, &vao);

  glBindVertexArray(vao);
  GLint enabled = 0;
  glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
  std::cout << "attrib0 enabled=" << enabled << "\n";
  glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
  std::cout << "attrib3 enabled=" << enabled << "\n";
  GLint elem = 0;
  glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &elem);
  std::cout << "element array bound = " << elem << "\n";

  mesh->init(vao);
}

void Object::render(std::shared_ptr<ShaderProgram> shader) {

  // Setting uniforms

  // Material uniforms
  // mat->apply(shader);

  // Object uniforms
  shader->set("modelMat", transform);
  glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(transform)));
  shader->set("normMat", normalMat);

  glBindVertexArray(vao);
  // Call for rendering: stream the current GPU geometry through the current GPU
  // program
  GLint currVAO = 0;
  glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currVAO);
  std::cout << "drawing vao=" << currVAO
            << " elementArrayBinding=" /*<< getBoundElementArrayBuffer()*/
            << std::endl;

  glDrawElements(GL_TRIANGLES, mesh->get_triangles_size(), GL_UNSIGNED_INT, 0);
  std::cout << "draw count = " << mesh->get_triangles_size() << std::endl;
  glBindVertexArray(0);
}