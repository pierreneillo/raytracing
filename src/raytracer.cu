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
uint recursion_depth = 1;

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
  int k;           // Index (or hash) of the hit object
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
    if (abs(det) < 0.0001)
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

  __device__ bool hits(Object o, Interval i, Hit &hit) {
    Hit idc_hit;
    if (!hits(o.bb, i, idc_hit))
      return false;
    bool did_hit = false;
    float closest_now = i.e;
    switch (o.type) {
    case TRIANGLE: {
      for (int k = 0; k < o.nPrimitives; k++) {
        Triangle t = o.buildTriangle(k);
        if (hits(t, Interval(i.s, closest_now), hit)) {
          did_hit = true;
          closest_now = hit.t;
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
        }
      }
      break;
    }
    }
    return did_hit;
  }

  __device__ bool nearestHit(Object *objects, int nObjects, Interval i,
                             Hit &hit) {

    bool did_hit = false;
    float closest_now = i.e;
    for (int k = 0; k < nObjects; k++) {
      if (hits(objects[k], Interval(i.s, closest_now), hit)) {
        did_hit = true;
        hit.k = k; // TODO : Change this; it prevents self shadowing
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

  Hit hit;
  if (r.nearestHit(objects, nObjects, Interval(0., 1. / 0.), hit)) {
    Material material = objects[hit.k].mat;
    // We create as many rays as there are lights and look for intersections
    // along the way

    real3 v = r.dir;
    real3 color = defaultShade(material);

    // Shadow ray
    for (int i = 0; i < nLights; i++) {
      LightSource light = lights[i];
      // Shadow ray
      real3 L = real3(light.pos) - hit.p;
      Ray sr = {hit.p, normalized(L)};
      Hit obstacleHit;
      if (!sr.nearestHit(objects, nObjects, Interval(EPSILON, length(L)),
                         obstacleHit)) {
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
        color += (diffuse + specular) * attenuation;
      }
    }

    if (recursion_depth > 0) {
      // We trace reflected ray and refracted ray if needed
      // Refraction dir is the symmetric of ray dir relative to normal:
      real3 reflected_dir = r.dir - 2 * hit.n * dot(hit.n, r.dir);
      Ray reflected = Ray(hit.p + EPSILON * reflected_dir, reflected_dir);
      color += material.kS * shade(reflected, lights, nLights, objects,
                                   nObjects, recursion_depth - 1);
    }
    return color;
  }
  return real3();
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
    Object o = objects[i];
    dh_objects[i] = o;
    switch (o.type) {
    case SPHERE: {
      // We only need to reallocate the Sphere*
      Sphere *d_spheres;
      gpuErrchk(cudaMalloc(&d_spheres, sizeof(Sphere) * o.nPrimitives));
      gpuErrchk(cudaMemcpy(d_spheres, o.data.spheres,
                           sizeof(Sphere) * o.nPrimitives,
                           cudaMemcpyHostToDevice));
      dh_objects[i].data.spheres = d_spheres;
      break;
    }
    case TRIANGLE: {
      real3 *d_verts;
      gpuErrchk(
          cudaMalloc(&d_verts, sizeof(real3) * o.data.triangleMesh.nVerts));
      gpuErrchk(cudaMemcpy(d_verts, o.data.triangleMesh.vertices,
                           sizeof(real3) * o.data.triangleMesh.nVerts,
                           cudaMemcpyHostToDevice));
      real3 *d_norm;
      gpuErrchk(
          cudaMalloc(&d_norm, sizeof(real3) * o.data.triangleMesh.nVerts));
      gpuErrchk(cudaMemcpy(d_norm, o.data.triangleMesh.normals,
                           sizeof(real3) * o.data.triangleMesh.nVerts,
                           cudaMemcpyHostToDevice));
      size_t *d_tris;
      gpuErrchk(cudaMalloc(&d_tris, sizeof(size_t) * o.nPrimitives * 3));
      gpuErrchk(cudaMemcpy(d_tris, o.data.triangleMesh.triangles,
                           sizeof(size_t) * o.nPrimitives * 3,
                           cudaMemcpyHostToDevice));
      dh_objects[i].data.triangleMesh = {d_verts, d_norm,
                                         o.data.triangleMesh.nVerts, d_tris};

      break;
    }
    }
  }

  // Device-allocated memory pointing to device-allocated memory
  gpuErrchk(cudaMalloc(&scene.d_objects, sizeof(Object) * nObjects));
  gpuErrchk(cudaMemcpy(scene.d_objects, dh_objects, sizeof(Object) * nObjects,
                       cudaMemcpyHostToDevice));
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
  size_t *trInd =
      (size_t *)malloc(sizeof(size_t) * 24 * resolution * resolution);

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
  return TriangleMesh({vpos, vnorm, latCount * lonCount, trInd});
};

int _main() {
  // We intialise camera

  RCamera cam = {real3(0, 0, 1. / glm::atan(glm::radians(22.5))) + .5,
                 real3(0, 0, 0),
                 real3(0, 1, 0),
                 45.f,
                 .1f,
                 1000.f,
                 aspect};

  LightSource lights[3] = {
      {real3(0., 2., 1.) * .8f, .5f, real3(1., 1., 1.), true},
      {real3(.3, 1.5, 1.4), .5f, real3(1., 1., .8), true},
      {real3(2., .5, -.2) * .8f, .5f, real3(1., 1., .8), true},
  };

  // Sphere of radius r and center C
  Sphere s = {real3(), 1};
  // Red material for the sphere
  Material sphereMat = {
      real3(1, 0, 0), .2, real3(1, 0, 0), .7, real3(1, 1, 1), .01, 32.,
      PHONG_MATERIAL};
  Material planeMat = {
      real3(.5, .5, .5), .2, real3(.5, .5, .5), .7, real3(1, 1, 1), .01, 32.,
      PHONG_MATERIAL};

  Object st = {
      sphereMat,
      TRIANGLE,
      {.triangleMesh = genSphere(16)},
      8 * 16 * 16,
      {
          {-1, -1, -1},
          {1, 1, 1},
      },

  };
  Object so = {sphereMat, SPHERE, {&s}, 1, {{-1, -1, -1}, {1, 1, 1}}};
  real3 planeVertices[4] = {
      {-1, -1, -1},
      {-1, -1, 1},
      {1, -1, 1},
      {1, -1, -1},
  };
  real3 planeNormals[4] = {
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0},
  };
  size_t planeTris[6] = {0, 1, 2, 0, 2, 3};
  Object plane = {planeMat,
                  TRIANGLE,
                  {.triangleMesh =
                       {
                           planeVertices,
                           planeNormals,
                           4,
                           planeTris,
                       }},
                  2,
                  {{-1, -1, -1}, {1, -1, 1}}};
  Object objects[2] = {st, plane};

  float4 *h_framebuffer = (float4 *)malloc(sizeof(float4) * w * h);
  init_raytracing(cam, lights, 3, objects, 2, h_framebuffer);
  for (int i = 0; i < 10; i++)
    raytrace(h_framebuffer);
  end_raytracing();

  printf("Finished raytracing\n");
  imageWrite(h_framebuffer, "test.ppm", w, h);

  free(h_framebuffer);
  return 0;
}