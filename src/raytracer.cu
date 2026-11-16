#include "../include/raytracer.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>

/*Raytracer in cuda*/
int w = 1920;
int h = 1080;
uint sampling_factor = 3;
__managed__ uint samples_per_pixel;
float aspect = (float)w / (float)h;
uint recursion_depth = 2;

glm::mat4x4 RCamera::matrix() {
  return glm::perspective(glm::radians(fov), aspect, near, far) *
         glm::lookAt(glm::vec3(eye), glm::vec3(at), glm::vec3(up));
}

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

    for (int i = 0; i < nLights; i++) {
      LightSource light = lights[i];
      // Shadow ray
      real3 L;
      if (light.t == POINT)
        L = light.pos - hit.p;
      else
        L = -light.dir;
      Ray sr = {hit.p, normalized(L)};
      Hit obstacleHit;
      if (!sr.nearestHit(objects, nObjects, Interval(0, length(L)), obstacleHit,
                         last_hit)) {
        // There is light (yepee)
        // We shade with the material
        real3 L;
        if (light.t == POINT)
          L = light.pos - hit.p;
        else
          L = -light.dir;
        float dist = length(L);
        real3 lv = L / dist;
        real3 h = normalized(lv + v); // Halfway vector for Blinn-Phong

        float attenuation = light.intensity;
        if (light.t == POINT) {
          // Distance Attenuation (with 3 factors: constant, linear and
          // quadratic, which are fixed for now)
          // TODO : allow tweaking ?
          float kc = .001; // Avoid division by zero at close distances
          float kl = 0.;
          float kq = 1.;
          attenuation /= (kc + kl * dist + kq * dist * dist);
        }
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
      // Reflection dir is the symmetric of ray dir relative to normal:
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
                               uint recursion_depth,
                               real2 *__restrict__ sample_offset) {

  __shared__ real3 pixels[MAX_BLOCK_SIZE];
  if (threadIdx.z == 0)
    pixels[threadIdx.x + blockDim.y * threadIdx.y] = real3();

  __syncthreads();
  uint2 screen_coords = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y);

  bool valid = screen_coords.x < w && screen_coords.y < h;
  if (valid) {
    // WORLD POS CALCULATION
    // We first transform screen coords into NDC at near plane
    float4 clip_space_pos = make_float4(
        2 * ((float)(screen_coords.x + .5f + sample_offset[threadIdx.z].x) /
             (float)w) -
            1.f,
        2 * ((float)(screen_coords.y + .5f + sample_offset[threadIdx.z].y) /
             (float)h) -
            1.f,
        -1.0f, 1.0f);
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
    uint idx = threadIdx.x + blockDim.y * threadIdx.y;
    atomicAdd(&pixels[idx].x, pixel.x);
    atomicAdd(&pixels[idx].y, pixel.y);
    atomicAdd(&pixels[idx].z, pixel.z);
  }
  __syncthreads();
  if (threadIdx.z == 0 && valid) {
    uint idx = threadIdx.x + blockDim.y * threadIdx.y;
    real3 sum = pixels[idx] / float(samples_per_pixel);
    imageStore(framebuffer, w, screen_coords, {sum.x, sum.y, sum.z, 1.0f});
  }
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

Object *init_raytracing(RCamera cam, LightSource *lights, int nLights,
                        Object *objects, int nObjects, float4 *framebuffer) {
  // Calculation of camera matrix
  samples_per_pixel = sampling_factor * sampling_factor;
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
                       cudaMemcpyDefault));
  gpuErrchk(cudaMalloc(&scene.d_framebuffer, sizeof(float4) * w * h));

  gpuErrchk(cudaMalloc(&scene.d_lightSource, sizeof(LightSource) * nLights));
  gpuErrchk(cudaMemcpy(scene.d_lightSource, lights,
                       sizeof(LightSource) * nLights, cudaMemcpyDefault));

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
                           cudaMemcpyDefault));
      gpu_obj.b.nodes = d_nodes;
    }
    if (cpu_obj.b.objects != nullptr) {
      BVHObj *d_objects;
      gpuErrchk(cudaMalloc(&d_objects, sizeof(BVHObj) * cpu_obj.nPrimitives));
      gpuErrchk(cudaMemcpy(d_objects, cpu_obj.b.objects,
                           sizeof(BVHObj) * cpu_obj.nPrimitives,
                           cudaMemcpyDefault));
      gpu_obj.b.objects = d_objects;
    }

    switch (cpu_obj.type) {
    case SPHERE: {
      // We only need to reallocate the SphereMesh
      real3 *d_centers;
      size_t cSize = sizeof(real3) * cpu_obj.data.sphereMesh.nPrimitives;
      gpuErrchk(cudaMalloc(&d_centers, cSize));
      gpuErrchk(cudaMemcpy(d_centers, cpu_obj.data.sphereMesh.C, cSize,
                           cudaMemcpyDefault));
      gpu_obj.data.sphereMesh.C = d_centers;
      float *d_radius;
      size_t rSize = sizeof(float) * cpu_obj.data.sphereMesh.nPrimitives;
      gpuErrchk(cudaMalloc(&d_radius, rSize));
      gpuErrchk(cudaMemcpy(d_radius, cpu_obj.data.sphereMesh.r, rSize,
                           cudaMemcpyDefault));
      gpu_obj.data.sphereMesh.r = d_radius;
      break;
    }
    case TRIANGLE: {
      real3 *d_verts;
      gpuErrchk(cudaMalloc(&d_verts,
                           sizeof(real3) * cpu_obj.data.triangleMesh.nVerts));
      gpuErrchk(cudaMemcpy(d_verts, cpu_obj.data.triangleMesh.vertices,
                           sizeof(real3) * cpu_obj.data.triangleMesh.nVerts,
                           cudaMemcpyDefault));
      real3 *d_norm;
      gpuErrchk(cudaMalloc(&d_norm,
                           sizeof(real3) * cpu_obj.data.triangleMesh.nVerts));
      gpuErrchk(cudaMemcpy(d_norm, cpu_obj.data.triangleMesh.normals,
                           sizeof(real3) * cpu_obj.data.triangleMesh.nVerts,
                           cudaMemcpyDefault));
      index_t *d_tris;
      gpuErrchk(cudaMalloc(&d_tris, sizeof(index_t) * cpu_obj.nPrimitives * 3));
      gpuErrchk(cudaMemcpy(d_tris, cpu_obj.data.triangleMesh.triangles,
                           sizeof(index_t) * cpu_obj.nPrimitives * 3,
                           cudaMemcpyDefault));
      gpu_obj.data.triangleMesh = {d_verts, d_norm,
                                   cpu_obj.data.triangleMesh.nVerts, d_tris,
                                   cpu_obj.data.triangleMesh.nPrimitives};

      break;
    }
    }
  }

  // Device-allocated memory pointing to device-allocated memory
  gpuErrchk(cudaMalloc(&scene.d_objects, sizeof(Object) * nObjects));
  gpuErrchk(cudaMemcpy(scene.d_objects, dh_objects, sizeof(Object) * nObjects,
                       cudaMemcpyDefault));
  return dh_objects;
}

// Sync raytracing and particle sim
void syncSphereObjectAndRebuildBVH(index_t objectIdx,
                                   const real3 *d_particlePositions,
                                   index_t nParticles) {
  if (objectIdx >= scene.nObjects || d_particlePositions == nullptr ||
      nParticles == 0)
    return;

  Object *h_obj_ptr = (Object *)malloc(sizeof(Object));
  gpuErrchk(cudaMemcpy(h_obj_ptr, scene.d_objects + objectIdx, sizeof(Object),
                       cudaMemcpyDefault));
  Object &h_obj = *h_obj_ptr;

  if (h_obj.type != SPHERE) {
    free(h_obj_ptr);
    return;
  }

  index_t nSphere = h_obj.data.sphereMesh.nPrimitives;
  if (nSphere != nParticles) {
    fprintf(
        stderr,
        "syncSphereObjectAndRebuildBVH: particle count mismatch (%u vs %u)\n",
        nParticles, nSphere);
    fflush(stderr);
    free(h_obj_ptr);
    return;
  }

  // We unfortunately need to copy a lot of data CPU side to rebuild BVH
  real3 *h_centers = new real3[nParticles];
  float *h_radii = new float[nParticles];

  gpuErrchk(cudaMemcpy(h_centers, d_particlePositions,
                       sizeof(real3) * nParticles, cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(h_radii, h_obj.data.sphereMesh.r,
                       sizeof(float) * nParticles, cudaMemcpyDefault));

  gpuErrchk(cudaMemcpy(h_obj.data.sphereMesh.C, h_centers,
                       sizeof(real3) * nParticles, cudaMemcpyDefault));

  SphereMesh h_mesh = {h_centers, h_radii, nParticles};
  BVH h_bvh = buildBVH(h_mesh);

  gpuErrchk(cudaMemcpy(h_obj.b.nodes, h_bvh.nodes,
                       sizeof(BVHNode) * (2 * nParticles + 1),
                       cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(h_obj.b.objects, h_bvh.objects,
                       sizeof(BVHObj) * nParticles, cudaMemcpyDefault));

  delete[] h_bvh.nodes;
  delete[] h_bvh.objects;
  delete[] h_centers;
  delete[] h_radii;
  free(h_obj_ptr);
}

void updateSphereObjectAndRebuildBVH(index_t objectIdx,
                                     const real3 *h_particlePositions,
                                     index_t nParticles) {
  if (objectIdx >= scene.nObjects || h_particlePositions == nullptr ||
      nParticles == 0)
    return;

  Object *h_obj_ptr = (Object *)malloc(sizeof(Object));
  gpuErrchk(cudaMemcpy(h_obj_ptr, scene.d_objects + objectIdx, sizeof(Object),
                       cudaMemcpyDefault));
  Object &h_obj = *h_obj_ptr;

  if (h_obj.type != SPHERE) {
    free(h_obj_ptr);
    return;
  }

  index_t nSphere = h_obj.data.sphereMesh.nPrimitives;
  if (nSphere != nParticles) {
    fprintf(stderr,
            "updateSphereObjectAndRebuildBVH: particle count mismatch (%u vs "
            "%u)\n",
            nParticles, nSphere);
    fflush(stderr);
    free(h_obj_ptr);
    return;
  }

  gpuErrchk(cudaMemcpy(h_obj.data.sphereMesh.C, h_particlePositions,
                       sizeof(real3) * nParticles, cudaMemcpyDefault));

  real3 *h_centers = new real3[nParticles];
  float *h_radii = new float[nParticles];
  for (index_t i = 0; i < nParticles; i++) {
    h_centers[i] = h_particlePositions[i];
  }

  gpuErrchk(cudaMemcpy(h_radii, h_obj.data.sphereMesh.r,
                       sizeof(float) * nParticles, cudaMemcpyDefault));

  SphereMesh h_mesh = {h_centers, h_radii, nParticles};
  BVH h_bvh = buildBVH(h_mesh);

  gpuErrchk(cudaMemcpy(h_obj.b.nodes, h_bvh.nodes,
                       sizeof(BVHNode) * (2 * nParticles + 1),
                       cudaMemcpyDefault));
  gpuErrchk(cudaMemcpy(h_obj.b.objects, h_bvh.objects,
                       sizeof(BVHObj) * nParticles, cudaMemcpyDefault));

  delete[] h_bvh.nodes;
  delete[] h_bvh.objects;
  delete[] h_centers;
  delete[] h_radii;
  free(h_obj_ptr);
}

void replaceObjectWithSphere(index_t objectIdx, const Object &cpu_obj) {
  if (objectIdx >= scene.nObjects || cpu_obj.type != SPHERE)
    return;

  Object gpu_obj = cpu_obj;

  real3 *d_centers = nullptr;
  size_t cSize = sizeof(real3) * cpu_obj.data.sphereMesh.nPrimitives;
  gpuErrchk(cudaMalloc(&d_centers, cSize));
  gpuErrchk(cudaMemcpy(d_centers, cpu_obj.data.sphereMesh.C, cSize,
                       cudaMemcpyDefault));
  gpu_obj.data.sphereMesh.C = d_centers;

  float *d_radius = nullptr;
  size_t rSize = sizeof(float) * cpu_obj.data.sphereMesh.nPrimitives;
  gpuErrchk(cudaMalloc(&d_radius, rSize));
  gpuErrchk(cudaMemcpy(d_radius, cpu_obj.data.sphereMesh.r, rSize,
                       cudaMemcpyDefault));
  gpu_obj.data.sphereMesh.r = d_radius;

  BVHNode *d_nodes = nullptr;
  gpuErrchk(
      cudaMalloc(&d_nodes, sizeof(BVHNode) * (2 * cpu_obj.nPrimitives + 1)));
  gpuErrchk(cudaMemcpy(d_nodes, cpu_obj.b.nodes,
                       sizeof(BVHNode) * (2 * cpu_obj.nPrimitives + 1),
                       cudaMemcpyDefault));
  gpu_obj.b.nodes = d_nodes;

  BVHObj *d_objects = nullptr;
  gpuErrchk(cudaMalloc(&d_objects, sizeof(BVHObj) * cpu_obj.nPrimitives));
  gpuErrchk(cudaMemcpy(d_objects, cpu_obj.b.objects,
                       sizeof(BVHObj) * cpu_obj.nPrimitives,
                       cudaMemcpyDefault));
  gpu_obj.b.objects = d_objects;

  gpuErrchk(cudaMemcpy(scene.d_objects + objectIdx, &gpu_obj, sizeof(Object),
                       cudaMemcpyDefault));
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
                       cudaMemcpyDefault));

  // We need to calculate sample offset, to store in an array and send to the
  // raytracer, using a grid sampling
  // TODO : Ideally, we should use rotated and scaled grid
  // (https://en.wikipedia.org/wiki/Supersampling#Rotated_grid)
  real2 *h_sample_offset = (real2 *)malloc(sizeof(real2) * samples_per_pixel);
  for (int i = 0; i < sampling_factor; i++)
    for (int j = 0; j < sampling_factor; j++)
      h_sample_offset[i + sampling_factor * j] =
          real2((float(i) + .5) / float(sampling_factor) - .5,
                (float(j) + .5) / float(sampling_factor) - .5);

  real2 *d_sample_offset;
  gpuErrchk(cudaMalloc(&d_sample_offset, sizeof(real2) * samples_per_pixel));
  gpuErrchk(cudaMemcpy(d_sample_offset, h_sample_offset,
                       samples_per_pixel * sizeof(real2),
                       cudaMemcpyHostToDevice));

  uint b_size = sqrt(int(MAX_BLOCK_SIZE / samples_per_pixel));
  dim3 block_size = {b_size, b_size, samples_per_pixel};
  dim3 grid_size = {(uint)ceil((float)w / (float)b_size),
                    (uint)ceil((float)h / (float)b_size), 1};

  raytraceKernel<<<grid_size, block_size>>>(
      scene.cam, scene.d_lightSource, scene.nLights, scene.d_objects,
      scene.nObjects, scene.d_invCamMat, scene.d_framebuffer, w, h,
      recursion_depth, d_sample_offset);
  gpuErrchk(cudaGetLastError());
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
