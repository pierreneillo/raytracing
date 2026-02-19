#include "utilities.cuh"

struct AABB {
  real3 m;
  real3 M;

  __host__ __device__ real3 normal(real3 p) {
    // Calculates the normal of the vector on a face of the AABB
    // There are only 6 possible values for the normal
    // We calculate distance to faces and choose the normal of the closest
    // face
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

struct TriangleMesh {
  real3 *vertices;
  real3 *normals;
  index_t nVerts;
  index_t *triangles;

  index_t nPrimitives;

  __host__ __device__ real3 centroid(index_t i) {
    assert(i < nPrimitives);
    index_t j = 3 * i;
    return (vertices[triangles[j]] + vertices[triangles[j + 1]] +
            vertices[triangles[j + 2]]) /
           3;
  }

  __host__ __device__ void add_to_bb(index_t i, AABB &bb) {
    assert(i < nPrimitives);
    index_t j = 3 * i;
    bb.add(vertices[triangles[j]]);
    bb.add(vertices[triangles[j + 1]]);
    bb.add(vertices[triangles[j + 2]]);
  }
};

struct Triangle {
  real3 v1, v2, v3;
  real3 n1, n2, n3;
};

struct SphereMesh {
  real3 *C;
  float *r;

  index_t nPrimitives;

  __host__ __device__ real3 centroid(index_t i) {
    assert(i < nPrimitives);
    return C[i];
  }

  __host__ __device__ void add_to_bb(index_t i, AABB &bb) {
    assert(i < nPrimitives);
    bb.add(C[i] - r[i]);
    bb.add(C[i] + r[i]);
  }
};

struct Sphere {
  real3 C;
  float r;
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
// original primitive array around
struct BVHObj {
  index_t index;
  real3 centroid;
};

struct BVH {
  BVHNode *nodes;
  BVHObj *objects;
  index_t nodes_used = 0;
};

template <typename Mesh>
__host__ __device__ void updateNodeBounds(BVH &b, Mesh &m, index_t nodeIdx);

template <typename Mesh>
__host__ __device__ void subdivide(BVH &b, Mesh &m, index_t nodeIdx);

template <typename Mesh> __host__ __device__ BVH buildBVH(Mesh m);