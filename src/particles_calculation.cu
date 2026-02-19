/* This programs aims to calculate positions of fluid particles parallely on the
 * GPU */

#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/iterator>
#include <cuda_runtime.h>

#include "cub/device/device_radix_sort.cuh"

#include "../include/particles_calculation.cuh"
#include "../include/raytracer.cuh"

// *************
// * CONSTANTS *
// *************

#define MIN_LENGTH 0.0000001

// Allocated memory for CUB
void *d_cub_mem = nullptr;
size_t d_cub_mem_size = 0;

// Simulation scale
int scale = 1;

// Cursor variables for interactive exp
real2 cursorPos = real2({10., 20.});
float cursorRadius = 10.;
float cursorMagnitude = 0;

void setCursorPos(real2 p) { cursorPos = p; }
void setCursorMagnitude(real m) { cursorMagnitude = m; }

// *************************
// * Helper math functions *
// *************************

__device__ __forceinline__ real clamp(real v, real vmin, real vmax) {
  return max(min(v, vmax), vmin);
}

__device__ __forceinline__ ind clamp(ind v, ind vmin, ind vmax) {
  return max(min(v, vmax), vmin);
}

__device__ __forceinline__ real dot(real2 v1, real2 v2) {
  return v1.x * v2.x + v1.y * v2.y;
}

__device__ __forceinline__ real length(real2 v) { return sqrt(dot(v, v)); }

// SPH Kernel function: cubic spline
// Credit : Kiwon Um (kiwon.um@telecom-paris.fr)
class CubicSpline {
public:
  explicit CubicSpline(const real h = 1, const unsigned int dim = 2)
      : _dim(dim) {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const real h) {
    const real h2 = square(h), h3 = h2 * h;
    _h = h;
    _sr = 2.f * h;
    _c[0] = 2.f / (3.f * h);
    _c[1] = 10.0f / (7.f * M_PI * h2);
    _c[2] = 1.f / (M_PI * h3);
    _gc[0] = _c[0] / h;
    _gc[1] = _c[1] / h;
    _gc[2] = _c[2] / h;
  }
  __device__ real smoothingLen() const { return _h; }
  __device__ real supportRadius() const { return _sr; }

  __device__ real f(const real l) const {
    const real q = l / _h;
    if (q >= 2.f)
      return 0;
    else if (q >= 1.f)
      return _c[_dim - 1] * (0.25 * cube(2.f - q));
    else
      return _c[_dim - 1] * (1.f - 1.5 * square(q) + 0.75 * cube(q));
  }
  __device__ real derivative_f(const real l) const {
    const real q = l / _h;
    if (q >= 2.f)
      return 0;
    else if (q > 1.f)
      return -_gc[_dim - 1] * 0.75 * square(2.f - q);
    else
      return _gc[_dim - 1] * (-3.f * q + 2.25 * square(q));
  }

  template <typename T> __device__ real w(const T &rij) const {
    return f(length(rij));
  }

  template <typename T> __device__ T grad_w(const T rij, const real len) const {
    if (len < MIN_LENGTH)
      return T();
    return derivative_f(len) / len * rij;
  }

  template <typename T> __device__ T grad_w(const T rij) const {
    return grad_w(rij, length(rij));
  }

private:
  unsigned int _dim;
  real _h, _sr, _c[3], _gc[3];
};

// GRID
struct Grid {
  Grid(ind nCells, real cellSize)
      : nCells(nCells), cellSize(cellSize), iCellSize(1. / cellSize) {};
  // Particle position, turned into 1D
  ind *d_particle1dPosIn;
  ind *d_particle1dPosOut;
  // Flattened grid, containing particle IDs
  ind *d_flattenedGridIn;
  ind *d_flattenedGridOut;

  // Grid cell start coordinates
  ind *d_cellStart;
  // Grid cell end coordinates
  ind *d_cellEnd;

  ind nCells;
  real cellSize;
  real iCellSize;
};

// *****************************
// *       INIT GLOBALS        *
// *****************************

// Particle parameters
// For now, we initialise with an array of size 1000
ind nParticles = 0;
ind posBufferSize = 1000;
// We make boundaries with particles, we store them in the first _nStatic
// slots of our vectors
ind nStatic;

// Particle arrays
// Host position
real3 *h_pos;
// Position
real3 *d_pos;
// Speed
real3 *d_vel;
// Acceleration
real3 *d_acc;
// Pressure
real *d_p;
// Density
real *d_d;

cudaStream_t stream;

SPH s = SPH();

Grid g = Grid(s.resX * s.resY * s.resZ * s.iCellSize, s.cellSize);

CubicSpline kernel = CubicSpline(s.h, 3);

// ****************************
// * GENERAL HELPER FUNCTIONS *
// ****************************

void fillRectangle(real3 startPos, real3 endPos, real3 *&pos, ind &nParticles,
                   ind &maxBufferSize, real h) {
  for (float x = startPos.x; x < endPos.x; x += h)
    for (float y = startPos.y; y < endPos.y; y += h)
      for (float z = startPos.z; z < endPos.z; z += h) {
        if (maxBufferSize <= nParticles) {
          // Realloc
          real3 *newptr =
              (real3 *)realloc(pos, sizeof(real3) * 2 * maxBufferSize);
          if (!newptr) {
            fprintf(stderr, "Realloc of size %lu bytes failed\n",
                    sizeof(real3) * 2 * maxBufferSize);
            fflush(stderr);
            exit(1);
          }
          pos = newptr;
          maxBufferSize *= 2;
        }
        pos[nParticles] = {x, y, z};
        nParticles++;
      }
}

void fillRectangle(float startPosX, float startPosY, float startPosZ,
                   float endPosX, float endPosY, float endPosZ, real3 *&posptr,
                   ind &nParticles, ind &maxBufferSize, real h) {
  fillRectangle(real3(startPosX, startPosY, startPosZ) + h / 2.f,
                real3(endPosX, endPosY, endPosZ) + h / 2.f, posptr, nParticles,
                maxBufferSize, h);
}

void fillMesh(const Object &o, real3 *&pos, ind &nParticles, ind &maxBufferSize,
              float h) {
  /*Fills the provided object with particles, adding their pos to the passed pos
   * array, and increasing the particle count*/
  AABB bb = o.b.nodes[0].bb;
  real3 ray_dirs[3] = {real3(0.123f, 0.456f, 0.789f),
                       real3(0.901f, 0.272f, 0.338f),
                       real3(0.511f, 0.731f, 0.451f)};
  constexpr float ray_origin_epsilon = 1e-4f;
  int turn = 0;
  real3 extent = bb.M - bb.m;
  int total = extent.x * extent.y * extent.z / (h * h * h);
  for (float x = bb.m.x; x <= bb.M.x; x += h)
    for (float y = bb.m.y; y <= bb.M.y; y += h)
      for (float z = bb.m.z; z <= bb.M.z; z += h) {
        printf("Turn %d/%d       \r", turn, total);
        turn++;
        real3 jitter = real3((float)rand() / (float)RAND_MAX,
                             (float)rand() / (float)RAND_MAX,
                             (float)rand() / (float)RAND_MAX) *
                       .01;
        real3 possible_pos = real3(x, y, z) + jitter;
        int inside_votes = 0;
        for (int k = 0; k < 3; k++) {
          real3 ray_dir = ray_dirs[k];
          Ray r = {possible_pos + ray_origin_epsilon * ray_dir, ray_dir};
          inside_votes += (r.hitCount(o, Interval(0., 1. / 0.)) % 2 == 1);
        }
        // We test if the position is inside or outside the mesh following the
        // even-odd rule, with a majority vote over 3 rays to avoid
        // weird things.
        if (inside_votes >= 2) {
          // The point is inside the mesh, we add the pos to pos list
          if (maxBufferSize <= nParticles) {
            // Realloc
            real3 *newptr =
                (real3 *)realloc(pos, sizeof(real3) * 2 * maxBufferSize);
            if (!newptr) {
              fprintf(stderr, "Realloc of size %lu bytes failed\n",
                      sizeof(real3) * 2 * maxBufferSize);
              fflush(stderr);
              exit(1);
            }
            pos = newptr;
            maxBufferSize *= 2;
          }
          pos[nParticles] = possible_pos;
          nParticles++;
        }
      }
}

// ***********************************
// * GRID MANAGMENT HELPER FUNCTIONS *
// ***********************************

__host__ __device__ __forceinline__ ind toHash(uint3 pos, ind nCells) {
  return (pos.x * 73856093 + pos.y * 19349669 + pos.z * 83492791) % nCells;
}

__device__ uint3 getNDCoords(real3 pos, float iCellSize) {
  ind gridX = (ind)(pos.x * iCellSize);
  ind gridY = (ind)(pos.y * iCellSize);
  ind gridZ = (ind)(pos.z * iCellSize);
  return {gridX, gridY, gridZ};
}

__device__ ind getHash(real3 pos, float iCellSize, ind nCells) {
  return toHash(getNDCoords(pos, iCellSize), nCells);
}

// ******************************
// * MAIN CALCULATION FUNCTIONS *
// ******************************

__global__ void calculateGridPos(Grid &g, const real3 *__restrict__ pos,
                                 ind nParticles) {
  ind i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < nParticles) {
    g.d_particle1dPosIn[i] = getHash(pos[i], g.iCellSize, g.nCells);
    g.d_flattenedGridIn[i] = i;
  }
}

__global__ void calculateGridCellIndexes(Grid &g, const real3 *__restrict__ pos,
                                         ind nParticles) {
  ind i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < nParticles) {
    // We look at the particles before and after in the `particle1DPos` array :
    // if they have different 1DPos values, this particle is a cell boundary
    ind cell = g.d_particle1dPosOut[i];

    if (i == 0 || g.d_particle1dPosOut[i - 1] != cell)
      g.d_cellStart[cell] = i;

    if (i == nParticles - 1 || g.d_particle1dPosOut[i + 1] != cell)
      g.d_cellEnd[cell] = i + 1;
  }
}

void computeGrid(Grid &g, const real3 *__restrict__ pos, int nParticles) {
  dim3 block_size = {MAX_BLOCK_SIZE, 1, 1};
  dim3 grid_size = {(nParticles + block_size.x - 1) / block_size.x, 1, 1};

  cudaMemset(g.d_cellStart, -1, sizeof(ind) * g.nCells);
  cudaMemset(g.d_cellEnd, -1, sizeof(ind) * g.nCells);

  calculateGridPos<<<grid_size, block_size, 0, stream>>>(g, pos, nParticles);

  // ...then sort the indexes array based on "hash"...
  cub::DeviceRadixSort::SortPairs(d_cub_mem, d_cub_mem_size,
                                  g.d_particle1dPosIn, g.d_particle1dPosOut,
                                  g.d_flattenedGridIn, g.d_flattenedGridOut,
                                  nParticles, 0, sizeof(ind) * 8, stream);
  // ...and finally update the cell indexes.
  calculateGridCellIndexes<<<grid_size, block_size, 0, stream>>>(g, pos,
                                                                 nParticles);
}

// Check wether the point at coords `pos` is within the rectangle
// [0, bounds.x[ * [0, bounds.y[ * [0, bounds.z[
__device__ __forceinline__ bool inBounds(int3 pos, uint3 bounds) {
  return 0 <= pos.x && 0 <= pos.y && 0 <= pos.z && pos.x < bounds.x &&
         pos.y < bounds.y && pos.z < bounds.z;
}

// Check wether the point at coords `pos` is within the rectangle
// [0, bounds.x[ * [0, bounds.y[ * [0, bounds.z[
__device__ __forceinline__ bool inBounds(uint3 pos, uint3 bounds) {
  return pos.x < bounds.x && pos.y < bounds.y && pos.z < bounds.z;
}

template <typename Func>
__device__ __forceinline__ void forEachNeighbour(const Grid &g, const real3 pos,
                                                 Func action) {
  int3 gridPos = {(int)(pos.x * g.iCellSize), (int)(pos.y * g.iCellSize),
                  (int)(pos.z * g.iCellSize)};
  for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
      for (int x = -1; x <= 1; x++) {
        int cx = gridPos.x + x;
        int cy = gridPos.y + y;
        int cz = gridPos.z + z;
        ind hash = toHash({(ind)cx, (ind)cy, (ind)cz}, g.nCells);
        ind start = g.d_cellStart[hash];
        // Special value for non initialized / empty cell
        if (start != 0xFFFFFFFF) {
          ind end = g.d_cellEnd[hash];
          // Our cell is non empty, we iterate
          for (int k = start; k < end; k++) {
            // We do our action on j
            action(g.d_flattenedGridOut[k]);
          }
        }
      }
}

__device__ real equationOfState(real d, real d0, real k, real gamma) {
  real drp;
  real dr = d / d0;
  if (gamma == 7) {
    float dr2 = dr * dr;
    float dr4 = dr2 * dr2;
    drp = dr4 * dr2 * dr;
  } else {
    printf("Slooooow\n");
    drp = cuda::std::pow(dr, gamma);
  }
  return max(k * (drp - 1), 0.0f);
}

__global__ void computeDensityAndPressure(const Grid &g, const SPH &s,
                                          const CubicSpline &kernel, real *d_d,
                                          real *d_p,
                                          const real3 *__restrict__ d_pos,
                                          ind nParticles) {
  ind i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < nParticles) {
    // Calculate the density of the i-th particle
    d_d[i] = 0;
    real3 pos_i = d_pos[i];
    real d_i = d_d[i];
    auto densityComputationFn = [=, &d_i](ind j) {
      real3 rij = pos_i - d_pos[j];
      d_i += s.m0 * kernel.w(rij);
    };
    forEachNeighbour(g, pos_i, densityComputationFn);
    d_d[i] = d_i;

    // calculate pressure of the i-th particle with the equation of state
    d_p[i] = equationOfState(d_i, s.d0, s.k, s.gamma);
  }
}

__global__ void
applyForces(const Grid &g, const SPH &s, const CubicSpline &kernel,
            const real *__restrict__ d_d, const real *__restrict__ d_p,
            real3 *d_acc, const real3 *__restrict__ d_vel, real3 *d_vel_write,
            const real3 *__restrict__ d_pos, real3 *d_pos_write, ind nParticles,
            ind nStatic, float dt, real2 cP = cursorPos, real cR = cursorRadius,
            real cM = cursorMagnitude) {
  ind i = threadIdx.x + blockDim.x * blockIdx.x + nStatic;
  if (i < nParticles) {
    // Calculate the forces applying on the i-th particle
    real3 acc_force = s.g;
    real3 pos_i = d_pos[i];
    real3 vel_i = d_vel[i];
    real d_i = d_d[i];
    real p_over_rho2_i = d_p[i] / square(d_i);
    real two_nu_m0_over_d_i = 2 * s.nu * s.m0 / d_i;
    auto forcesComputationFunction = [=, &acc_force](ind j) {
      // For viscosity and pressure forces, we skip oneself
      if (i != j) {
        real3 rij = pos_i - d_pos[j];
        real sqlen = dot(rij, rij);
        real len = sqrt(sqlen);
        len = max(len, MIN_LENGTH);
        if (len <= kernel.supportRadius()) {
          real3 uij = vel_i - d_vel[j];

          // Calculate pressure force and apply it
          real pdterm = p_over_rho2_i + d_p[j] / square(d_d[j]);
          acc_force -= s.m0 * pdterm * kernel.grad_w(rij, len);

          // Calculate viscosity force
          real gradterm = dot(rij, kernel.grad_w(rij, len)) / (sqlen + s.eh2);
          acc_force += two_nu_m0_over_d_i * gradterm * uij;
        }
      }
    };

    forEachNeighbour(g, d_pos[i], forcesComputationFunction);

    real3 cP3 = real3(cP.x, cP.y, pos_i.z);
    float distToCur = length(pos_i - cP3);
    if (distToCur < cR) {
      acc_force += cM / cR * (pos_i - cP3);
    }

    d_acc[i] = acc_force;
    d_vel_write[i] = vel_i + dt * d_acc[i];
    d_pos_write[i] = pos_i + dt * d_vel_write[i];
  }
}

// ************************
// *    Main functions    *
// ************************

ind getParticleCount() { return nParticles; }
ind getStaticParticleCount() { return nStatic; }

float getdt() { return s.dt; }

uint getResX() { return s.resX; }
uint getResY() { return s.resY; }
uint getResZ() { return s.resZ; }

void initSimulation(real3 *h_pos, bool initPosNVel) {

  // We malloc the other buffers, which will have exactly the right size, and
  // set them to zero, except for `d_pos` which receives h_pos values.
  size_t partR3Size = sizeof(real3) * nParticles;
  size_t partIndSize = sizeof(ind) * nParticles;

  if (initPosNVel) {
    cudaMalloc(&d_pos, partR3Size);
    cudaMemcpy(d_pos, h_pos, partR3Size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_vel, partR3Size);
    cudaMemset(d_vel, 0, partR3Size);
  }

  cudaMalloc(&d_acc, partR3Size);
  cudaMemset(d_acc, 0, partR3Size);

  cudaMalloc(&d_p, sizeof(real) * nParticles);
  cudaMemset(d_p, 0, sizeof(real) * nParticles);

  cudaMalloc(&d_d, sizeof(real) * nParticles);
  cudaMemset(d_d, 0, sizeof(real) * nParticles);

  cudaMalloc(&g.d_particle1dPosIn, partIndSize);
  cudaMalloc(&g.d_flattenedGridIn, partIndSize);

  cudaMalloc(&g.d_particle1dPosOut, partIndSize);
  cudaMalloc(&g.d_flattenedGridOut, partIndSize);

  cudaMalloc(&g.d_cellStart, sizeof(ind) * g.nCells);
  cudaMalloc(&g.d_cellEnd, sizeof(ind) * g.nCells);

  cudaMemset(g.d_cellStart, -1, sizeof(ind) * g.nCells);
  cudaMemset(g.d_cellEnd, -1, sizeof(ind) * g.nCells);

  cudaStreamCreate(&stream);

  // We initialize the CUB memory that will be used for sorts indefinitely
  // afterwards
  cub::DeviceRadixSort::SortPairs(
      d_cub_mem, d_cub_mem_size, (const ind *)g.d_particle1dPosIn,
      g.d_particle1dPosOut, (const ind *)g.d_flattenedGridIn,
      g.d_flattenedGridOut, nParticles);

  cudaMalloc(&d_cub_mem, d_cub_mem_size);
}

void stepSimulation(const real3 *__restrict__ d_pos_read, real3 *d_pos_write,
                    const real3 *__restrict__ d_vel_read, real3 *d_vel_write,
                    ind n_steps) {
  dim3 block_size = {MAX_BLOCK_SIZE, 1, 1};
  dim3 grid_size = {(nParticles + block_size.x - 1) / block_size.x};
  dim3 dyn_grid_size = {(nParticles - nStatic + block_size.x - 1) /
                        block_size.x};
  for (int i = 0; i < n_steps; i++) {
    // Simulation step
    const real3 *__restrict__ pos_read_ptr =
        (i == 0) ? d_pos_read : d_pos_write;
    const real3 *__restrict__ vel_read_ptr =
        (i == 0) ? d_vel_read : d_vel_write;
    computeGrid(g, pos_read_ptr, nParticles);
    computeDensityAndPressure<<<grid_size, block_size, 0, stream>>>(
        g, s, kernel, d_d, d_p, pos_read_ptr, nParticles);
    applyForces<<<dyn_grid_size, block_size, 0, stream>>>(
        g, s, kernel, d_d, d_p, d_acc, vel_read_ptr, d_vel_write, pos_read_ptr,
        d_pos_write, nParticles, nStatic, s.dt);
  }
}

void stepSimulation(ind n_steps) {
  stepSimulation(d_pos, d_pos, d_vel, d_vel, n_steps);
}

const real3 *getSimulationDevicePositions() { return d_pos; }

void endSimulation() {
  // Free alocated arrays
  free(h_pos);
  cudaFree(d_acc);
  cudaFree(d_d);
  cudaFree(d_p);
  cudaFree(g.d_particle1dPosIn);
  cudaFree(g.d_flattenedGridIn);
  cudaFree(g.d_particle1dPosOut);
  cudaFree(g.d_flattenedGridOut);
  cudaFree(g.d_cellStart);
  cudaFree(g.d_cellEnd);
  cudaFree(d_cub_mem);
}