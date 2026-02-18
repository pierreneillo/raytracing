#include "../include/raytracer.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>

/*Raytracer in cuda*/
uint w = 1920;
uint h = 1080;
float aspect = (float)w / (float)h;
uint recursion_depth = 2;

glm::mat4x4 RCamera::matrix() {
  return glm::perspective(glm::radians(fov), aspect, near, far) *
         glm::lookAt(glm::vec3(eye), glm::vec3(at), glm::vec3(up));
}

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
      Interval c = Interval(max(a.s, b.s), min(a.e, b.e));
      return c;
    }
    return Interval(1, 0);
  };
};

struct Ray {
  __device__ Ray(real3 eye, real3 dir) : eye(eye), dir(dir) {};
  __device__ real3 at(real t) { return eye + t * dir; };
  __device__ bool hits(Sphere s, Interval i, Hit &hit) {
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

  __device__ bool hits(Triangle tri, Interval i, Hit &hit) {
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

  __device__ bool hits(AABB bb, Interval i, Hit &hit) {
    // We first eliminate on a simple criteria
    real3 d = dir;
    real3 o = eye;
    if ((d.x == 0 && (bb.m.x > d.x || bb.M.x < d.x)) ||
        (d.y == 0 && (bb.m.y > d.y || bb.M.y < d.y)) ||
        (d.z == 0 && (bb.m.z > d.z || bb.M.z < d.z)))
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
    switch (o.type) {
    case TRIANGLE: {
      // BVH Traversal
      index_t stack[MAX_TREE_DEPTH];
      index_t stackPtr = 0;
      stack[stackPtr++] = 0;

      Hit idc_hit;
      while (stackPtr > 0) {
        if (stackPtr >= MAX_TREE_DEPTH)
          printf("Stack depth exceeded\n");
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
            Triangle t = o.buildTriangle(o.b.objects[k].index);
            if (k != last_kP && hits(t, Interval(i.s, closest_now), hit)) {
              did_hit = true;
              closest_now = hit.t;
              hit.kP = k;
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
      break;
    }
    case SPHERE: {
      for (int k = 0; k < o.nPrimitives; k++) {
        Sphere s = o.data.spheres[k];
        if (hits(s, Interval(i.s, closest_now), hit)) {
          did_hit = true;
          closest_now = hit.t;
          hit.kP = k;
        }
      }
      break;
    }
    }
    return did_hit;
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

__device__ real3 defaultShade(Material material) {
  return material.ambient * material.kA;
}

__device__ real3 shade(Ray r, LightSource *lights, uint nLights,
                       Object *objects, uint nObjects,
                       uint recursion_depth = 0) {

  real3 color = real3();
  real throughput = 1;
  Ray current_ray = r;
  Hit last_hit = {.kO = -1};
  for (int depth = 0; depth <= recursion_depth; depth++) {
    Hit hit;
    if (!current_ray.nearestHit(objects, nObjects, Interval(0, 1. / 0.), hit,
                                last_hit))
      break;
    last_hit = hit;
    Material material = objects[hit.kO].mat;
    // We create as many rays as there are lights and look for intersections
    // along the way

    real3 v = -current_ray.dir;

    // Shadow ray
    for (int i = 0; i < nLights; i++) {
      LightSource light = lights[i];
      // Shadow ray
      real3 L = real3(light.pos) - hit.p;
      Ray sr = {hit.p, normalized(L)};
      Hit obstacleHit;
      if (!sr.nearestHit(objects, nObjects, Interval(0, length(L)), obstacleHit,
                         last_hit)) {
        // There is light (yepee)
        // We shade with the material
        real3 L = light.pos - hit.p;
        float dist = length(L);
        real3 lv = L / dist;
        real3 h = normalized(lv + v); // Halfway vector for Blinn-Phong

        // Distance Attenuation (in 1 / d²) to make it look realistic,
        // incorporating intensity A small epsilon is added to prevent
        // division by 0
        float attenuation = light.intensity / (dist * dist + .001);

        // We calculate diffuse and specular components
        // Diffuse Component
        float diffCoeff = max(dot(hit.n, lv), 0.f);
        real3 diffuse =
            light.color * material.diffuse * diffCoeff * material.kD;

        // Specular lighting
        float specCoeff = pow(max(dot(hit.n, h), 0.), material.shininess);
        real3 specular =
            light.color * material.specular * specCoeff * material.kS;

        // Attenuate
        color += (diffuse + specular) * attenuation * throughput;
      }
    }
    // Ambient color
    color += defaultShade(material) * throughput;

    if (depth < recursion_depth) {
      // We trace reflected ray and refracted ray if needed
      // Refraction dir is the symmetric of ray dir relative to normal:
      real3 reflected_dir =
          current_ray.dir - 2 * hit.n * dot(hit.n, current_ray.dir);
      current_ray = Ray(hit.p, reflected_dir);
      throughput *= material.kS;
      if (throughput < THROUGHPUT_EPSILON)
        break;
    }
  }
  return color;
}

__global__ void raytraceKernel(RCamera cam, LightSource *lights, uint nLights,
                               Object *objects, uint nObjects,
                               const float *__restrict__ invCameraMat,
                               float4 *framebuffer, uint w, uint h,
                               uint recursion_depth) {
  uint2 screen_coords = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y);
  if (screen_coords.x >= w || screen_coords.y >= h)
    return;

  // WORLD POS CALCULATION
  // We first transform screen coords into NDC at near plane
  float4 clip_space_pos = make_float4(
      2 * ((float)(screen_coords.x + .5f) / (float)w) - 1.f,
      2 * ((float)(screen_coords.y + .5f) / (float)h) - 1.f, -1.0f, 1.0f);
  // We then transform to world pos by using the inverse camera matrix
  float4 worldPosH = mul(invCameraMat, clip_space_pos);
  real3 worldPos = real3(worldPosH.x, worldPosH.y, worldPosH.z);
  if (worldPosH.w != 0)
    worldPos = worldPos / worldPosH.w;

  // The ray R(t) = eye_pos + t*ray_direction
  real3 eye = cam.eye;
  Ray r = Ray(eye, normalized(worldPos - eye));

  // Default pixel value
  real3 pixel = shade(r, lights, nLights, objects, nObjects, recursion_depth);
  imageStore(framebuffer, w, screen_coords, {pixel.x, pixel.y, pixel.z, 1.0f});
}

void imageWrite(float4 *framebuffer, const char *fn, int w, int h) {
  FILE *out = fopen(fn, "w");
  if (!out) {
    fprintf(stderr, "Cannot open file %s", fn);
    return;
  }
  fprintf(out, "P6\n%d %d\n255\n", w, h);
  for (int y = h - 1; y >= 0; --y) {
    for (int x = 0; x < w; ++x) {
      int i = (y * w + x);
      float a = framebuffer[i].w;
      unsigned char r = static_cast<unsigned char>(
          clamp(framebuffer[i].x * a, 0.0f, 1.0f) * 255.0f);
      unsigned char g = static_cast<unsigned char>(
          clamp(framebuffer[i].y * a, 0.0f, 1.0f) * 255.0f);
      unsigned char b = static_cast<unsigned char>(
          clamp(framebuffer[i].z * a, 0.0f, 1.0f) * 255.0f);
      fwrite(&r, sizeof(char), 1, out);
      fwrite(&g, sizeof(char), 1, out);
      fwrite(&b, sizeof(char), 1, out);
    }
  }
  fclose(out);
  printf("Framebuffer saved to %s", fn);
}
struct RScene {
  RCamera cam;
  int nLights;
  float *d_invCamMat;
  float4 *d_framebuffer;
  LightSource *d_lightSource;
  Object *d_objects;
  int nObjects;
};

RScene scene;

void updateCamera(RCamera rcam) { scene.cam = rcam; }

void reallocDeviceFramebuffer() {
  cudaFree(scene.d_framebuffer);
  cudaMalloc(&scene.d_framebuffer, sizeof(float4) * w * h);
}

void init_raytracing(RCamera cam, LightSource *lights, int nLights,
                     Object *objects, int nObjects, float4 *framebuffer) {
  // Calculation of camera matrix
  scene.cam = cam;
  scene.nLights = nLights;
  scene.nObjects = nObjects;
  glm::mat4x4 camMat = cam.matrix();
  glm::mat4x4 invCamMat = glm::inverse(camMat);
  float h_invCamMat[16];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      h_invCamMat[i * 4 + j] = invCamMat[i][j];
    }
  }

  gpuErrchk(cudaMalloc(&scene.d_invCamMat, 16 * sizeof(float)));
  gpuErrchk(cudaMemcpy(scene.d_invCamMat, h_invCamMat, 16 * sizeof(float),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&scene.d_framebuffer, sizeof(float4) * w * h));

  gpuErrchk(cudaMalloc(&scene.d_lightSource, sizeof(LightSource) * nLights));
  gpuErrchk(cudaMemcpy(scene.d_lightSource, lights,
                       sizeof(LightSource) * nLights, cudaMemcpyHostToDevice));

  // Host-allocated memory pointing to device-allocated memory
  Object *dh_objects = (Object *)malloc(sizeof(Object) * nObjects);

  // We copy memory for each object, changing only the pointers
  for (int i = 0; i < nObjects; i++) {
    dh_objects[i] = objects[i];
    const Object &cpu_obj = objects[i];
    Object &gpu_obj = dh_objects[i];
    // If there is a BVH, we need to reallocate it
    if (cpu_obj.b.nodes != nullptr) {
      BVHNode *d_nodes;
      gpuErrchk(cudaMalloc(&d_nodes,
                           sizeof(BVHNode) * (2 * cpu_obj.nPrimitives + 1)));
      gpuErrchk(cudaMemcpy(d_nodes, cpu_obj.b.nodes,
                           sizeof(BVHNode) * (2 * cpu_obj.nPrimitives + 1),
                           cudaMemcpyHostToDevice));
      gpu_obj.b.nodes = d_nodes;
    }
    if (cpu_obj.b.objects != nullptr) {
      BVHObj *d_objects;
      gpuErrchk(cudaMalloc(&d_objects, sizeof(BVHObj) * cpu_obj.nPrimitives));
      gpuErrchk(cudaMemcpy(d_objects, cpu_obj.b.objects,
                           sizeof(BVHObj) * cpu_obj.nPrimitives,
                           cudaMemcpyHostToDevice));
      gpu_obj.b.objects = d_objects;
    }

    switch (cpu_obj.type) {
    case SPHERE: {
      // We only need to reallocate the Sphere*
      Sphere *d_spheres;
      gpuErrchk(cudaMalloc(&d_spheres, sizeof(Sphere) * cpu_obj.nPrimitives));
      gpuErrchk(cudaMemcpy(d_spheres, cpu_obj.data.spheres,
                           sizeof(Sphere) * cpu_obj.nPrimitives,
                           cudaMemcpyHostToDevice));
      dh_objects[i].data.spheres = d_spheres;
      break;
    }
    case TRIANGLE: {
      real3 *d_verts;
      gpuErrchk(cudaMalloc(&d_verts,
                           sizeof(real3) * cpu_obj.data.triangleMesh.nVerts));
      gpuErrchk(cudaMemcpy(d_verts, cpu_obj.data.triangleMesh.vertices,
                           sizeof(real3) * cpu_obj.data.triangleMesh.nVerts,
                           cudaMemcpyHostToDevice));
      real3 *d_norm;
      gpuErrchk(cudaMalloc(&d_norm,
                           sizeof(real3) * cpu_obj.data.triangleMesh.nVerts));
      gpuErrchk(cudaMemcpy(d_norm, cpu_obj.data.triangleMesh.normals,
                           sizeof(real3) * cpu_obj.data.triangleMesh.nVerts,
                           cudaMemcpyHostToDevice));
      index_t *d_tris;
      gpuErrchk(cudaMalloc(&d_tris, sizeof(index_t) * cpu_obj.nPrimitives * 3));
      gpuErrchk(cudaMemcpy(d_tris, cpu_obj.data.triangleMesh.triangles,
                           sizeof(index_t) * cpu_obj.nPrimitives * 3,
                           cudaMemcpyHostToDevice));
      dh_objects[i].data.triangleMesh = {
          d_verts, d_norm, cpu_obj.data.triangleMesh.nVerts, d_tris,
          cpu_obj.data.triangleMesh.nTriangles};

      break;
    }
    }
  }

  // Device-allocated memory pointing to device-allocated memory
  gpuErrchk(cudaMalloc(&scene.d_objects, sizeof(Object) * nObjects));
  gpuErrchk(cudaMemcpy(scene.d_objects, dh_objects, sizeof(Object) * nObjects,
                       cudaMemcpyHostToDevice));
  free(dh_objects);
}

void raytrace(float4 *framebuffer) {
  scene.cam.aspect = aspect;

  glm::mat4x4 camMat = scene.cam.matrix();
  glm::mat4x4 invCamMat = glm::inverse(camMat);
  float h_invCamMat[16];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      h_invCamMat[i * 4 + j] = invCamMat[i][j];
    }
  }

  gpuErrchk(cudaMemcpy(scene.d_invCamMat, h_invCamMat, 16 * sizeof(float),
                       cudaMemcpyHostToDevice));

  uint b_size = sqrt(MAX_BLOCK_SIZE);
  dim3 block_size = {b_size, b_size, 1};
  dim3 grid_size = {(uint)ceil((float)w / (float)b_size),
                    (uint)ceil((float)h / (float)b_size), 1};

  raytraceKernel<<<grid_size, block_size>>>(
      scene.cam, scene.d_lightSource, scene.nLights, scene.d_objects,
      scene.nObjects, scene.d_invCamMat, scene.d_framebuffer, w, h,
      recursion_depth);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(framebuffer, scene.d_framebuffer, sizeof(float4) * w * h,
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaDeviceSynchronize());
}

void end_raytracing() {
  gpuErrchk(cudaFree(scene.d_lightSource));
  gpuErrchk(cudaFree(scene.d_invCamMat));
  gpuErrchk(cudaFree(scene.d_framebuffer));
}

TriangleMesh
genSphere(const unsigned int resolution) // should generate a unit sphere
{
  // We interpret resolution as the number of edges on a half circumference
  unsigned int latCount = resolution * 2 + 1;
  unsigned int lonCount = resolution * 2 + 1;
  real3 *vpos = (real3 *)malloc(sizeof(real3) * latCount * lonCount);
  real3 *vnorm = (real3 *)malloc(sizeof(real3) * latCount * lonCount);
  float2 *vtexcoords = (float2 *)malloc(sizeof(float2) * latCount * lonCount);
  index_t *trInd =
      (index_t *)malloc(sizeof(index_t) * 24 * resolution * resolution);

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
      vpos[j + lonCount * i] = {v.x, v.y, v.z};
      vnorm[j + lonCount * i] = {n.x, n.y, n.z};

      float U = -(float)j / (float)(lonCount - 1);
      vtexcoords[j + lonCount * i] = {U, V};
    }
  }

  // Build two triangles per square between latitude/longitude grid cells
  for (unsigned int i = 0; i + 1 < latCount; i++) {
    for (unsigned int j = 0; j + 1 < lonCount; j++) {
      unsigned int a = i * lonCount + j;
      unsigned int b = (i + 1) * lonCount + j;
      unsigned int c = i * lonCount + (j + 1);
      unsigned int d = (i + 1) * lonCount + (j + 1);

      trInd[6 * (j + i * (lonCount - 1))] = a;
      trInd[6 * (j + i * (lonCount - 1)) + 1] = b;
      trInd[6 * (j + i * (lonCount - 1)) + 2] = c;
      trInd[6 * (j + i * (lonCount - 1)) + 3] = c;
      trInd[6 * (j + i * (lonCount - 1)) + 4] = b;
      trInd[6 * (j + i * (lonCount - 1)) + 5] = d;
    }
  }
  return TriangleMesh({vpos, vnorm, latCount * lonCount, trInd,
                       2 * (lonCount - 1) * (latCount - 1)});
};
