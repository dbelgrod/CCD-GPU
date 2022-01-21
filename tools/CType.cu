#include <ccdgpu/CType.cuh>

namespace ccd {

#ifdef GPUTI_USE_DOUBLE_PRECISION
__host__ __device__ Scalar3 make_Scalar3(const Scalar &a, const Scalar &b,
                                         const Scalar &c) {
  return make_double3(a, b, c);
}
__host__ __device__ Scalar2 make_Scalar2(const Scalar &a, const Scalar &b) {
  return make_double2(a, b);
}
#warning Using Double
#else
#warning Using Float
__host__ __device__ Scalar3 make_Scalar3(const Scalar &a, const Scalar &b,
                                         const Scalar &c) {
  return make_float3(a, b, c);
}
__host__ __device__ Scalar2 make_Scalar2(const Scalar &a, const Scalar &b) {
  return make_float2(a, b);
}
#endif

} // namespace ccd