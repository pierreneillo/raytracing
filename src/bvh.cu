#include "../include/raytracer.cuh"

// #define SAH

template <typename Mesh> __host__ __device__ BVH buildBVH(Mesh m) {
  BVH b;
  b.objects = new BVHObj[m.nPrimitives];
  b.nodes = new BVHNode[m.nPrimitives * 2 + 1];
  for (index_t i = 0; i < m.nPrimitives; i++) {
    b.objects[i] = {i, m.centroid(i)};
  }
  BVHNode &root = b.nodes[0];
  root.firstPrimitive = 0;
  root.nPrimitives = m.nPrimitives;
  printf("Starting building BVH for %u primitives...\n", m.nPrimitives);
  updateNodeBounds(b, m, 0);
  b.nodes_used++;
  printf("Now recursing...\n");
  subdivide(b, m, 0);
  return b;
}

// We force template construction, otherwise it doesn't work
template __host__ __device__ BVH buildBVH<TriangleMesh>(TriangleMesh m);
template __host__ __device__ BVH buildBVH<SphereMesh>(SphereMesh m);

template <typename Mesh>
__host__ __device__ void updateNodeBounds(BVH &b, Mesh &m, index_t nodeIdx) {
  BVHNode &node = b.nodes[nodeIdx];
  if (node.nPrimitives == 0)
    printf("FAULTY NODE : N°%u\n", nodeIdx);
  assert(node.nPrimitives > 0);
  AABB bb = AABB({real3(10e30, 10e30, 10e30), real3(-10e30, -10e30, -10e30)});
  for (int i = node.firstPrimitive; i < node.firstPrimitive + node.nPrimitives;
       i++) {
    index_t j = b.objects[i].index;
    m.add_to_bb(j, bb);
  }
  node.bb = bb;
}

template <typename T> __host__ __device__ void swap(T *arr, int i, int j) {
  T temp = arr[i];
  arr[i] = arr[j];
  arr[j] = temp;
}

template <typename Mesh>
__host__ __device__ float evaluateSAH(BVH &b, Mesh &m, BVHNode &node, int axis,
                                      float splitPos) {
  AABB left = AABB({real3(10e30, 10e30, 10e30), real3(-10e30, -10e30, -10e30)});
  AABB right =
      AABB({real3(10e30, 10e30, 10e30), real3(-10e30, -10e30, -10e30)});
  index_t leftCount = 0;
  for (int i = node.firstPrimitive; i < node.firstPrimitive + node.nPrimitives;
       i++) {
    index_t j = b.objects[i].index;
    if (b.objects[i].centroid[axis] < splitPos) {
      leftCount++;
      m.add_to_bb(j, left);
    } else {
      m.add_to_bb(j, right);
    }
  }
  float cost =
      leftCount * left.area() + (node.nPrimitives - leftCount) * right.area();
  return (cost > 0) ? cost : 10e30f;
}

template <typename Mesh>
__host__ __device__ void subdivide(BVH &b, Mesh &m, index_t nodeIdx) {
  BVHNode &node = b.nodes[nodeIdx];

// We split using the SAH criteria if it is defined
#ifdef SAH
  int axis = -1;
  real splitPos = 0, cost = 10e30f;
  // We iterate over each direction and each primitive and calculate the SAH
  for (int a = 0; a < 3; a++)
    for (int i = 0; i < node.nPrimitives; i++) {
      BVHObj o = b.objects[i + node.firstPrimitive];
      float candidateSplit = o.centroid[a];
      float candidateCost = evaluateSAH(b, tm, node, a, candidateSplit);
      if (candidateCost < cost) {
        axis = a;
        splitPos = candidateSplit;
        cost = candidateCost;
      }
    }

#else
  if (node.nPrimitives <= 2) {
    // If there are one or two primitives left : this is considered a leaf
    return;
  }
  // For now we split in half
  real3 size = node.bb.M - node.bb.m;
  int axis = (size.y > size.x) ? ((size.z > size.y) ? 2 : 1)
                               : ((size.x > size.z) ? 0 : 2);
  real splitPos = node.bb.m[axis] + size[axis] * .5f;
#endif

  // We sort the primitive list in place
  int mini = node.firstPrimitive;         // First valid index
  int maxi = mini + node.nPrimitives - 1; // Last valid index
  while (mini <= maxi) {
    if (b.objects[mini].centroid[axis] < splitPos)
      // This primitive can stay here
      mini++;
    else {
      // We swap the primitive with the last non inspected one
      swap(b.objects, mini, maxi--);
    }
  }
  index_t leftCount = mini - node.firstPrimitive;
  // This case could happen
  if (leftCount == 0 || leftCount == node.nPrimitives) {
    return;
  }

  // We can now initialise left and right children nodes and recurse
  node.left = b.nodes_used;
  b.nodes_used += 2;
  BVHNode &leftNode = b.nodes[node.left];
  BVHNode &rightNode = b.nodes[node.left + 1];
  leftNode.firstPrimitive = node.firstPrimitive;
  leftNode.nPrimitives = leftCount;
  rightNode.firstPrimitive = mini;
  rightNode.nPrimitives = node.nPrimitives - leftCount;
  node.nPrimitives = 0;

  // We now update node bounds
  updateNodeBounds(b, m, node.left);
  updateNodeBounds(b, m, node.left + 1);
  subdivide(b, m, node.left);
  subdivide(b, m, node.left + 1);
}