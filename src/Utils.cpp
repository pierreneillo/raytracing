#include "Utils.hpp"
#include <cstdio>

#define TINYOBJLOADER_IMPLEMENTATION
#include "../dep/tiny_obj_loader.hpp"

resize_policy policy = EXTEND;

float TARGET_FPS = 30.f;

bool moving = false;
bool playing = false;
bool recording = false;

glm::vec2 moveOrigin;

GLFWwindow *window;
GLuint program;

ind N_BUFFERS = 3;
ind current_read_buffer = 0;

FILE *ffmpegPipe = nullptr;

void startRecording(int width, int height) {
  // Command to record at 60fps, yuv420p format for compatibility
  std::string cmd =
      "ffmpeg  -y -f rawvideo -pix_fmt rgb24 -s " + std::to_string(width) +
      "x" + std::to_string(height) + " -r " + std::to_string(TARGET_FPS) +
      " -i - -vf \"vflip\" -c:v libx264 -preset fast -crf 18 -pix_fmt "
      "yuv420p output.mp4";

  ffmpegPipe = popen(cmd.c_str(), "w");
}

// Inside the main loop, after glClear and rendering particles
void recordFrame(int width, int height) {
  if (!ffmpegPipe)
    return;

  // Allocate a buffer for the RGB pixel
  static std::vector<unsigned char> pixels(3 * width * height);

  // Read pixels from the OpenGL back buffer
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

  // Write the pixel data to the FFmpeg pipe
  fwrite(pixels.data(), 1, pixels.size(), ffmpegPipe);
}

TriangleMesh loadMesh(std::string filename, glm::mat4 transform_matrix) {
  // Wrapper to the tinyobjloader
  tinyobj::ObjReaderConfig reader_config;
  reader_config.triangulate = true;
  tinyobj::ObjReader reader;
  TriangleMesh t;
  if (!reader.ParseFromFile(filename, reader_config)) {
    std::cerr << "There was an error while loading mesh '" << filename << "'"
              << std::endl;
    // To prevent error upon mesh loading
    t.nPrimitives = 0;
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

  // We transform the mesh if needed
  glm::mat4 normal_transform_matrix = glm::inverseTranspose(transform_matrix);
  for (int i = 0; i < vertices.size(); i++) {
    vertices[i] = real3(
        glm::vec3(transform_matrix *
                  glm::vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1.)));
    normals[i] = real3(
        glm::vec3(normal_transform_matrix *
                  glm::vec4(normals[i].x, normals[i].y, normals[i].z, 1.)));
  }

  // Lastly we copy the data in the designated buffers
  t.nVerts = vertices.size();
  t.vertices = new real3[t.nVerts];
  std::copy(vertices.begin(), vertices.end(), t.vertices);
  t.normals = new real3[t.nVerts];
  std::copy(normals.begin(), normals.end(), t.normals);
  t.triangles = new index_t[indexes.size()];
  std::copy(indexes.begin(), indexes.end(), t.triangles);
  t.nPrimitives = indexes.size() / 3;

  return t;
}

std::ifstream fetcher;
index_t N = 0;

index_t initFetcher(const std::string &filename) {
  fetcher.open(filename);
  if (!fetcher.is_open())
    return 0;

  // Read the total count from the top of the file
  if (!(fetcher >> N))
    N = 0;

  return N;
}

real3 *fetchAndTransformPositions(real scale, real3 translation) {
  // Safeguard
  if (!fetcher.is_open() || N == 0)
    return nullptr;

  real iScale = 1. / scale;

  // Allocate memory for N real3
  real3 *positions = new real3[N];

  for (index_t i = 0; i < N; ++i) {
    if (!(fetcher >> positions[i].x >> positions[i].y >> positions[i].z)) {
      std::cerr << "Error reading position n°" << i << "\n";
      break;
    }
    positions[i] = (positions[i] - translation) * iScale;
  }

  return positions;
}

bool fetchAndTransformPositionsInPlace(real3 *positions, real scale,
                                       real3 translation) {
  if (!fetcher.is_open() || N == 0 || positions == nullptr)
    return false;

  real iScale = 1. / scale;
  for (index_t i = 0; i < N; ++i) {
    if (!(fetcher >> positions[i].x >> positions[i].y >> positions[i].z)) {
      return false;
    }
    positions[i] = (positions[i] - translation) * iScale;
  }
  return true;
}

void closeFetcher() {
  if (fetcher.is_open())
    fetcher.close();
}

Material basicColoredMat(real3 color) {
  return {color, .1, color, .9, color, 0., 32., PHONG_MATERIAL};
};