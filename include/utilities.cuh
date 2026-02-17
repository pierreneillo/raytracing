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

#define MAX_BLOCK_SIZE 256

/*********************************************\
* Vector type definition and helper functions *
\*********************************************/

typedef float real;

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
    z /= r.y;
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

template <typename T>
__device__ __host__ __forceinline__ T clamp(T v, T vmin, T vmax) {
  return max(min(v, vmax), vmin);
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