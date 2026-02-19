#include "utilities.cuh"

#include "bvh.cuh"

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
    SphereMesh sphereMesh;
    TriangleMesh triangleMesh;
  } data;
  index_t nPrimitives;
  BVH b;

  __host__ __device__ Object(Material mat, SphereMesh sM)
      : mat(mat), data({.sphereMesh = sM}), nPrimitives(sM.nPrimitives) {
    real3 mini = sM.C[0];
    real3 maxi = sM.C[0];
    for (int i = 0; i < nPrimitives; i++) {
      mini = vmin(mini, sM.C[i] - sM.r[i]);
      maxi = vmax(maxi, sM.C[i] - sM.r[i]);
    }
    b = buildBVH(sM);
    type = SPHERE;
  }

  __host__ Object(Material mat, TriangleMesh tM)
      : mat(mat), data({.triangleMesh = tM}), nPrimitives(tM.nPrimitives) {
    real3 mini = tM.vertices[0];
    real3 maxi = tM.vertices[0];
    for (int i = 0; i < tM.nVerts; i++) {
      mini = vmin(mini, tM.vertices[i]);
      maxi = vmax(maxi, tM.vertices[i]);
    }
    b = buildBVH(tM);
    type = TRIANGLE;
  }

  __host__ __device__ Triangle buildTriangle(index_t k) {
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

// #############
// #    Hit    #
// #############

// used to store hits
struct Hit {
  real3 p;         // Position
  real3 n;         // Normal
  float t;         // Parameter of the ray
  bool front_face; // Tells whether the ray has hit the object from the outside
  int kO;          // Index (or hash) of the hit object
  int kP;          // Index (or hash) of the hit primitive
};

// ##############
// #  Interval  #
// ##############

struct Interval {
  __host__ __device__ Interval(float s, float e) : s(s), e(e) {};
  float s, e;
  __host__ __device__ static Interval make(float a, float b) {
    return (a < b) ? Interval(a, b) : Interval(b, a);
  };
  __host__ __device__ bool isEmpty() { return s > e; }
  __host__ __device__ __forceinline__ bool contains(float t) {
    return s <= t && t <= e;
  }
  __host__ __device__ bool overlaps(Interval other) {
    return s <= other.e && other.s <= e;
  }
  __host__ __device__ static Interval intersection(Interval a, Interval b) {
    if (a.overlaps(b)) {
      Interval c = Interval(fmaxf(a.s, b.s), fminf(a.e, b.e));
      return c;
    }
    return Interval(1, 0);
  };
};

struct Ray {
  __host__ __device__ Ray(real3 eye, real3 dir) : eye(eye), dir(dir) {};
  __host__ __device__ real3 at(real t) { return eye + t * dir; };
  __host__ __device__ bool hits(Sphere s, Interval i, Hit &hit) {
    // Computation of the intersection with the sphere
    // Given P(t) = at(r,t), we have an intersection along the ray iff
    // there is a t > 0 such as (C - P(t))·(C - P(t)) == r²
    // This can be written (C - eye)·(C - eye) - 2 t (C - eye)·direction + t²
    // direction·direction = r² It is a 2nd degree equation, we solve for t, and
    // deduce there is one or more intersections iff delta = 4 ((C -
    // eye)·direction)² - 4 (|C - eye|² - r²) >= 0

    // C - eye
    real3 cme = s.C - eye;
    // h = -2b
    float h = dot(cme, dir);
    float c = dot(cme, cme) - s.r * s.r;
    float delta = h * h - c;

    if (delta > 0) {
      float t = h - sqrt(delta);
      if (i.contains(t)) {
        hit.p = at(t);
        // The normal is simply the center to point vector, normalized
        hit.n = normalized(hit.p - s.C);
        hit.t = t;
        // We check whether the ray has hit the sphere from the inside
        hit.front_face = dot(dir, hit.n) < 0;
        // If it is the case, we change the normal to point the other way
        // hit.n = hit.front_face ? hit.n : -hit.n;
        return true;
      } else
        return false;
    } else
      return false;
  }

  __host__ __device__ bool hits(Triangle tri, Interval i, Hit &hit) {
    // Implementation of the Möller-Trumbore algorithm
    real3 e1 = tri.v2 - tri.v1;
    real3 e2 = tri.v3 - tri.v1;
    real3 rxe2 = cross(dir, e2);
    float det = dot(e1, rxe2);

    // We consider the ray is parallel
    if (abs(det) < TRIANGLE_EPSILON)
      return false;

    float idet = 1.0 / det;
    real3 s = eye - tri.v1;
    float u = idet * dot(s, rxe2);

    if (u < 0.0 || u > 1.0)
      return false;

    real3 sxe1 = cross(s, e1);
    float v = idet * dot(dir, sxe1);

    if (v < 0.0 || u + v > 1.0)
      return false;

    // At this stage we can compute t to find out where the intersection point
    // is on the line.
    float t = idet * dot(e2, sxe1);

    if (i.contains(t)) {
      // Yeepee, there is an intersection
      hit.t = t;
      hit.p = eye + dir * t;

      // Barycentric interpolation for the normal, with our calculated weights
      float w = 1.0 - u - v;
      hit.n = normalized(w * tri.n1 + u * tri.n2 + v * tri.n3);
      // We check whether the ray has hit the triangle from the front or back
      // face
      hit.front_face = dot(dir, hit.n) < 0;
      // If it is the case, we change the normal to point the other way
      hit.n = hit.front_face ? hit.n : -hit.n;
      return true;
    }

    return false;
  }

  __host__ __device__ bool hits(AABB bb, Interval i, Hit &hit) {
    // We first eliminate on a simple criteria
    real3 d = dir;
    real3 o = eye;
    if ((d.x == 0 && (bb.m.x > o.x || bb.M.x < o.x)) ||
        (d.y == 0 && (bb.m.y > o.y || bb.M.y < o.y)) ||
        (d.z == 0 && (bb.m.z > o.z || bb.M.z < o.z)))
      return false;
    // We calculate t intervals in which the ray intersects with each of the
    // axis
    Interval j = i;
    for (int dim = 0; dim < 3; dim++) {
      // The ray has equation O + tD and the planes are at [dim] = m and [dim] =
      // M Therefore the two hit times respect O + tD = m => t = (m - O)/D, and
      // idem t = (M - O)/D
      float t1 = (bb.M[dim] - o[dim]) / d[dim];
      float t2 = (bb.m[dim] - o[dim]) / d[dim];
      if (!j.overlaps(Interval::make(t1, t2)) || j.isEmpty())
        return false;
      j = Interval::intersection(j, Interval::make(t1, t2));
    }
    if (j.isEmpty())
      return false;
    else {
      hit.t = j.s;
      hit.p = at(j.s);
      hit.n = bb.normal(hit.p);
      hit.front_face = dot(hit.n, d) < 0;
      hit.n = hit.front_face ? hit.n : -hit.n;
      return true;
    }
  }

  __device__ bool hits(Object o, Interval i, Hit &hit, const int last_kP) {
    bool did_hit = false;
    float closest_now = i.e;
    // BVH Traversal
    index_t stack[MAX_TREE_DEPTH];
    index_t stackPtr = 0;
    stack[stackPtr++] = 0;

    Hit idc_hit;
    while (stackPtr > 0) {
      if (stackPtr >= MAX_TREE_DEPTH) {
        printf("Stack depth exceeded\n");
        break;
      }
      int nodeIdx = stack[--stackPtr];
      // int depth = depthStack[stackPtr];
      BVHNode &node = o.b.nodes[nodeIdx];
      // If we didn't hit the box, we directly pass to the next box on the
      // stack
      if (!hits(node.bb, Interval(i.s, closest_now), idc_hit))
        continue;

      // If we did hit we branch depending if the node is a leaf
      if (node.nPrimitives > 0) {
        // Leaf node
        for (int k = node.firstPrimitive;
             k < node.nPrimitives + node.firstPrimitive; k++) {
          index_t j = o.b.objects[k].index;
          switch (o.type) {
          case TRIANGLE: {
            Triangle t = o.buildTriangle(j);
            if (j != last_kP && hits(t, Interval(i.s, closest_now), hit)) {
              did_hit = true;
              closest_now = hit.t;
              hit.kP = j;
            }
            break;
          }
          case SPHERE: {
            Sphere s = {o.data.sphereMesh.C[j], o.data.sphereMesh.r[j]};
            if (j != last_kP && hits(s, Interval(i.s, closest_now), hit)) {
              did_hit = true;
              closest_now = hit.t;
              hit.kP = j;
            }
            break;
          }
          }
        }
      } else {
        // Internal node, we add children to stack
        // We order the two children nodes to improve performances
        Hit left_hit;
        Hit right_hit;
        if (hits(o.b.nodes[node.left].bb, Interval(i.s, closest_now),
                 left_hit)) {
          if (hits(o.b.nodes[node.left + 1].bb, Interval(i.s, closest_now),
                   right_hit)) {
            // Both nodes have a hit, we put the closest up on the stack
            if (left_hit.t < right_hit.t) {
              stack[stackPtr++] = node.left + 1;
              stack[stackPtr++] = node.left;
            } else {
              stack[stackPtr++] = node.left;
              stack[stackPtr++] = node.left + 1;
            }
          } else {
            // Right node has not hit, we only put the left one on the stack
            stack[stackPtr++] = node.left;
          }
        } else {
          // Left node has not hit, we only put the right one on the stack
          stack[stackPtr++] = node.left + 1;
        }
      }
    }
    return did_hit;
  }
  __host__ __device__ index_t hitCount(Object o, Interval i) {
    index_t nHits = 0;

    index_t stack[MAX_TREE_DEPTH];
    index_t stackPtr = 0;
    stack[stackPtr++] = 0;

    Hit idc_hit;
    while (stackPtr > 0) {
      if (stackPtr >= MAX_TREE_DEPTH) {
        printf("Stack depth exceeded\n");
        break;
      }
      int nodeIdx = stack[--stackPtr];
      // int depth = depthStack[stackPtr];
      BVHNode &node = o.b.nodes[nodeIdx];
      // If we didn't hit the box, we directly pass to the next box on the
      // stack
      if (!hits(node.bb, Interval(i.s, i.e), idc_hit))
        continue;

      // If we did hit we branch depending if the node is a leaf
      if (node.nPrimitives > 0) {
        // Leaf node
        for (int k = node.firstPrimitive;
             k < node.nPrimitives + node.firstPrimitive; k++) {
          index_t j = o.b.objects[k].index;
          switch (o.type) {
          case TRIANGLE: {
            Triangle t = o.buildTriangle(j);
            if (hits(t, Interval(i.s, i.e), idc_hit))
              nHits++;

            break;
          }
          case SPHERE: {
            Sphere s = {o.data.sphereMesh.C[j], o.data.sphereMesh.r[j]};
            if (hits(s, Interval(i.s, i.e), idc_hit))
              nHits++;
            break;
          }
          }
        }
      } else {
        // Internal node, we add children to stack
        // We order the two children nodes to improve performances
        stack[stackPtr++] = node.left;
        stack[stackPtr++] = node.left + 1;
      }
    }
    return nHits;
  }

  __device__ bool nearestHit(Object *objects, int nObjects, Interval i,
                             Hit &hit, Hit last_hit = {.kO = -1}) {

    bool did_hit = false;
    float closest_now = i.e;
    for (int k = 0; k < nObjects; k++) {
      if (hits(objects[k], Interval(i.s, closest_now), hit,
               (last_hit.kO == k) ? last_hit.kP : -1)) {
        did_hit = true;
        hit.kO = k;
        closest_now = hit.t;
      }
    }
    return did_hit;
  }

  real3 eye;
  real3 dir;
};

TriangleMesh genSphere(const unsigned int resolution);

void updateCamera(RCamera);

void reallocDeviceFramebuffer();

Object *init_raytracing(RCamera cam, LightSource *lights, int nLights,
                        Object *objects, int nObjects, float4 *framebuffer);

void syncSphereObjectAndRebuildBVH(index_t objectIdx,
                                   const real3 *d_particlePositions,
                                   index_t nParticles);

void updateSphereObjectAndRebuildBVH(index_t objectIdx,
                                     const real3 *h_particlePositions,
                                     index_t nParticles);

void replaceObjectWithSphere(index_t objectIdx, const Object &cpu_obj);

void raytrace(float4 *frambuffer);
void end_raytracing();