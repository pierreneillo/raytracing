#include "utilities.cuh"
// Full HD
extern uint w;
extern uint h;
extern float aspect;

extern uint recursion_depth;

// Epsilon used in elemination criteria for parralel rays in triangle
// intersection
#define TRIANGLE_EPSILON .0000001

// Epsilon used in elimination criteria for low throughput contribution in
// recursive raytracing
#define THROUGHPUT_EPSILON .001

// Maw tree depth for BVH traversal
#define MAX_TREE_DEPTH 32

inline __device__ void imageStore(float4 *image, uint w, uint2 coords,
                                  float4 value) {
  image[coords.x + coords.y * w] = value;
}

struct TriangleMesh {
  real3 *vertices;
  real3 *normals;
  index_t nVerts;
  index_t *triangles;
  index_t nTriangles;
};

struct Sphere {
  real3 C;
  float r;
};

struct Triangle {
  real3 v1, v2, v3;
  real3 n1, n2, n3;
};

struct AABB {
  real3 m;
  real3 M;
  __host__ __device__ real3 normal(real3 p) {
    // Calculates the normal of the vector on a face of the AABB
    // There are only 6 possible values for the normal
    // We calculate distance to faces and choose the normal of the closest face
    real3 dm = p - m;
    real3 dM = M - p;

    float minD =
        fminf(fminf(fminf(dm.x, dm.y), fminf(dm.z, dM.x)), fminf(dM.y, dM.z));

    // We check which has the minimun dist to the point
    real3 n = real3((minD == dm.x)   ? -1.0
                    : (minD == dM.x) ? 1.0
                                     : 0.0,
                    (minD == dm.y)   ? -1.0
                    : (minD == dM.y) ? 1.0
                                     : 0.0,
                    (minD == dm.z)   ? -1.0
                    : (minD == dM.z) ? 1.0
                                     : 0.0);

    return n;
  }

  __host__ __device__ void add(real3 v) {
    m = vmin(m, v);
    M = vmax(M, v);
  }

  __host__ __device__ float area() {
    real3 e = M - m;
    return dot(e, e);
  }
};

/* BVH Implementation
 * This BVH Implementation was done following this tuto (thanks to the author)
 * https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
 */

struct BVHNode {
  AABB bb;
  index_t left;
  index_t firstPrimitive, nPrimitives;
};

// Structure made to stock the objects from the BVH, without moving the
// original triangle array around
struct BVHObj {
  index_t index;
  real3 centroid;
};

struct BVH {
  BVHNode *nodes;
  BVHObj *objects;
  index_t nodes_used = 0;
};

void updateNodeBounds(BVH &b, TriangleMesh &tm, index_t nodeIdx);

void subdivide(BVH &b, TriangleMesh &tm, index_t nodeIdx);

BVH buildBVH(TriangleMesh tm);

enum MaterialType { UNDEF_MATERIAL, PHONG_MATERIAL, PBR_MATERIAL };

struct Material {
  real3 ambient;
  real kA;
  real3 diffuse;
  real kD;
  real3 specular;
  real kS;
  real shininess;
  bool metallic;
  MaterialType type;
};

enum PrimitiveType { TRIANGLE, SPHERE };

struct Object {
  Material mat;
  PrimitiveType type;
  union {
    Sphere *spheres;
    TriangleMesh triangleMesh;
  } data;
  index_t nPrimitives;
  BVH b;

  __host__ __device__ Object(Material mat, Sphere *spheres, index_t nPrimitives)
      : mat(mat), data({.spheres = spheres}), nPrimitives(nPrimitives) {
    real3 mini = spheres[0].C;
    real3 maxi = spheres[0].C;
    for (int i = 0; i < nPrimitives; i++) {
      Sphere s = spheres[i];
      mini = vmin(mini, s.C - s.r);
      maxi = vmax(maxi, s.C + s.r);
    }
    b = {nullptr, nullptr, 0};
    type = SPHERE;
  }

  __host__ Object(Material mat, TriangleMesh tM, index_t nPrimitives)
      : mat(mat), data({.triangleMesh = tM}), nPrimitives(nPrimitives) {
    real3 mini = tM.vertices[0];
    real3 maxi = tM.vertices[0];
    for (int i = 0; i < tM.nVerts; i++) {
      mini = vmin(mini, tM.vertices[i]);
      maxi = vmax(maxi, tM.vertices[i]);
    }
    b = buildBVH(tM);
    type = TRIANGLE;
  }

  __device__ Triangle buildTriangle(index_t k) {
    if (k < nPrimitives) {
      index_t j = 3 * k;
      real3 *v = data.triangleMesh.vertices;
      real3 *n = data.triangleMesh.normals;
      index_t *t = data.triangleMesh.triangles;
      return {v[t[j]], v[t[j + 1]], v[t[j + 2]],
              n[t[j]], n[t[j + 1]], n[t[j + 2]]};
    }
    return {real3(), real3(), real3(), real3(), real3(), real3()};
  }
};

struct LightSource {
  real3 pos;
  float intensity;
  real3 color;
  bool isActive;
};

struct RCamera {
  real3 eye;
  real3 at;
  real3 up;
  float fov;
  float near;
  float far;
  float aspect;

  glm::mat4x4 matrix();
};

TriangleMesh genSphere(const unsigned int resolution);

void updateCamera(RCamera);

void reallocDeviceFramebuffer();

void init_raytracing(RCamera cam, LightSource *lights, int nLights,
                     Object *objects, int nObjects, float4 *framebuffer);

void raytrace(float4 *frambuffer);
void end_raytracing();