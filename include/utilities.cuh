#ifndef UTILITIES_H
#define UTILITIES_H
#include <cstdio>
#include <glm/glm.hpp>

#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

/***********\
* Constants *
\***********/

extern int w;
extern int h;
extern float aspect;

extern uint recursion_depth;

#define MAX_BLOCK_SIZE 256

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

/*********************************************\
* Vector type definition and helper functions *
\*********************************************/

typedef float real;
typedef uint32_t index_t;

typedef unsigned int ind;

struct real2 {
  real x, y;

  __host__ __device__ real2() {
    x = 0.f;
    y = 0.f;
  };

  __host__ __device__ real2(real x, real y) : x(x), y(y) {}

  // Adapted from code of Kiwon Um (kiwon.um@telecom-paris.fr)
  __host__ __device__ real2 &operator+=(const real2 &r) {
    x += r.x;
    y += r.y;
    return *this;
  }
  __host__ __device__ real2 &operator-=(const real2 &r) {
    x -= r.x;
    y -= r.y;
    return *this;
  }
  __host__ __device__ real2 &operator*=(const real2 &r) {
    x *= r.x;
    y *= r.y;
    return *this;
  }
  __host__ __device__ real2 &operator/=(const real2 &r) {
    x /= r.x;
    y /= r.y;
    return *this;
  }

  __host__ __device__ real2 &operator+=(const real s) {
    x += s;
    y += s;
    return *this;
  }
  __host__ __device__ real2 &operator-=(const real s) {
    x -= s;
    y -= s;
    return *this;
  }
  __host__ __device__ real2 &operator*=(const real s) {
    x *= s;
    y *= s;
    return *this;
  }
  __host__ __device__ real2 &operator/=(const real s) {
    const real d = static_cast<real>(1) / s;
    return operator*=(d);
  }

  // unary operators
  __host__ __device__ real2 operator+() const { return *this; }
  __host__ __device__ real2 operator-() const { return {-x, -y}; }

  // binary operators
  __host__ __device__ real2 operator+(const real2 &r) const {
    return real2(*this) += r;
  }
  __host__ __device__ real2 operator-(const real2 &r) const {
    return real2(*this) -= r;
  }
  __host__ __device__ real2 operator*(const real2 &r) const {
    return real2(*this) *= r;
  }
  __host__ __device__ real2 operator/(const real2 &r) const {
    return real2(*this) /= r;
  }

  __host__ __device__ real2 operator+(const real s) const {
    return real2(*this) += s;
  }
  __host__ __device__ real2 operator-(const real s) const {
    return real2(*this) -= s;
  }
  __host__ __device__ real2 operator*(const real s) const {
    return real2(*this) *= s;
  }
  __host__ __device__ real2 operator/(const real s) const {
    return real2(*this) /= s;
  }

  __host__ __device__ friend real2 operator+(const real s, const real2 v) {
    return v + s;
  }
  __host__ __device__ friend real2 operator-(const real s, const real2 v) {
    return v - s;
  }
  __host__ __device__ friend real2 operator*(const real s, const real2 v) {
    return v * s;
  }
  __host__ __device__ friend real2 operator/(const real s, const real2 v) {
    return v / s;
  }

  // comparison operators
  __host__ __device__ bool operator==(const real2 &r) const {
    return ((x == r.x) && (y == r.y));
  }
  __host__ __device__ bool operator!=(const real2 &r) const {
    return !((*this) == r);
  }
  __host__ __device__ bool operator<(const real2 &r) const {
    return (x != r.x) ? x < r.x : y < r.y;
  }
};

struct real3 {
  real x, y, z;

  __host__ __device__ real3() {
    x = 0.f;
    y = 0.f;
    z = 0.f;
  };

  __host__ real3(glm::vec3 v) {
    x = v.x;
    y = v.y;
    z = v.z;
  }

  __host__ __device__ real3(real x, real y, real z) : x(x), y(y), z(z) {}

  __host__ __device__ real operator[](int k) {
    switch (k) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    return x;
  };

  // Adapted from code of Kiwon Um (kiwon.um@telecom-paris.fr)
  __host__ __device__ real3 &operator+=(const real3 &r) {
    x += r.x;
    y += r.y;
    z += r.z;
    return *this;
  }
  __host__ __device__ real3 &operator-=(const real3 &r) {
    x -= r.x;
    y -= r.y;
    z -= r.z;
    return *this;
  }
  __host__ __device__ real3 &operator*=(const real3 &r) {
    x *= r.x;
    y *= r.y;
    z *= r.z;
    return *this;
  }
  __host__ __device__ real3 &operator/=(const real3 &r) {
    x /= r.x;
    y /= r.y;
    z /= r.z;
    return *this;
  }

  __host__ __device__ real3 &operator+=(const real s) {
    x += s;
    y += s;
    z += s;
    return *this;
  }
  __host__ __device__ real3 &operator-=(const real s) {
    x -= s;
    y -= s;
    z -= s;
    return *this;
  }
  __host__ __device__ real3 &operator*=(const real s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  __host__ __device__ real3 &operator/=(const real s) {
    const real d = static_cast<real>(1) / s;
    return operator*=(d);
  }

  // unary operators
  __host__ __device__ real3 operator+() const { return *this; }
  __host__ __device__ real3 operator-() const { return {-x, -y, -z}; }

  // binary operators
  __host__ __device__ real3 operator+(const real3 &r) const {
    return real3(*this) += r;
  }
  __host__ __device__ real3 operator-(const real3 &r) const {
    return real3(*this) -= r;
  }
  __host__ __device__ real3 operator*(const real3 &r) const {
    return real3(*this) *= r;
  }
  __host__ __device__ real3 operator/(const real3 &r) const {
    return real3(*this) /= r;
  }

  __host__ __device__ real3 operator+(const real s) const {
    return real3(*this) += s;
  }
  __host__ __device__ real3 operator-(const real s) const {
    return real3(*this) -= s;
  }
  __host__ __device__ real3 operator*(const real s) const {
    return real3(*this) *= s;
  }
  __host__ __device__ real3 operator/(const real s) const {
    return real3(*this) /= s;
  }

  __host__ __device__ friend real3 operator+(const real s, const real3 v) {
    return v + s;
  }
  __host__ __device__ friend real3 operator-(const real s, const real3 v) {
    return v - s;
  }
  __host__ __device__ friend real3 operator*(const real s, const real3 v) {
    return v * s;
  }
  __host__ __device__ friend real3 operator/(const real s, const real3 v) {
    return v / s;
  }

  // comparison operators
  __host__ __device__ bool operator==(const real3 &r) const {
    return ((x == r.x) && (y == r.y));
  }
  __host__ __device__ bool operator!=(const real3 &r) const {
    return !((*this) == r);
  }
  __host__ __device__ bool operator<(const real3 &r) const {
    return (x != r.x) ? x < r.x : y < r.y;
  }

  __host__ __device__ operator glm::vec3() const { return glm::vec3(x, y, z); }
};

__host__ __device__ __forceinline__ real dot(real3 a, real3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ real length(real3 v) {
  return sqrt(dot(v, v));
}

__host__ __device__ __forceinline__ real3 cross(real3 a, real3 b) {
  return real3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x);
}

__host__ __device__ __forceinline__ real3 normalized(real3 v) {
  return v / length(v);
}

__host__ __device__ __forceinline__ real square(real x) { return x * x; }

__host__ __device__ __forceinline__ real cube(real x) { return x * x * x; }

template <typename T>
__device__ __host__ __forceinline__ T clamp(T v, T vmin, T vmax) {
  return max(min(v, vmax), vmin);
}

__device__ __host__ __forceinline__ real3 vmin(real3 a, real3 b) {
  return real3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __host__ __forceinline__ real3 vmax(real3 a, real3 b) {
  return real3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __device__ float4 mul(const float *mat, float4 v) {
  // Matrix multiplication function
  float4 result;
  result.x = mat[0] * v.x + mat[4] * v.y + mat[8] * v.z + mat[12] * v.w;
  result.y = mat[1] * v.x + mat[5] * v.y + mat[9] * v.z + mat[13] * v.w;
  result.z = mat[2] * v.x + mat[6] * v.y + mat[10] * v.z + mat[14] * v.w;
  result.w = mat[3] * v.x + mat[7] * v.y + mat[11] * v.z + mat[15] * v.w;
  return result;
}

#endif // UTILITIES_H