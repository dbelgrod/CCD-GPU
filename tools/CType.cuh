#pragma once
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

namespace ccd
{
#ifdef GPUTI_USE_DOUBLE_PRECISION
	typedef double3 Scalar3;
	typedef double2 Scalar2;
	typedef double Scalar;
	__host__ __device__ Scalar3 make_Scalar3(const Scalar &a, const Scalar &b,
											 const Scalar &c);
	__host__ __device__ Scalar2 make_Scalar2(const Scalar &a, const Scalar &b);
#warning Using Double
#define SCALAR_LIMIT DBL_MAX;
#else
	typedef float3 Scalar3;
	typedef float2 Scalar2;
	typedef float Scalar;
#warning Using Float
	__host__ __device__ Scalar3 make_Scalar3(const Scalar &a, const Scalar &b,
											 const Scalar &c);
	__host__ __device__ Scalar2 make_Scalar2(const Scalar &a, const Scalar &b);
#define SCALAR_LIMIT INT_MAX;
#endif
} // namespace ccd