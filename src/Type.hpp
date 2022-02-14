#pragma once
#include <array>
#include <assert.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
// #include <gputi/read_rational_csv.cuh>
#include <ccdgpu/CType.cuh>
#include <cuda/semaphore>
#include <limits>
#include <utility>

namespace ccd {
///////////////////////////////
// here are the parameters for the memory pool
static const int MAX_OVERLAP_SIZE = 1e7;
static const int MAX_CHECKS = 1e6;

///////////////////////////////

// THE FOLLOWING VALUES ARE JUST FOR DEBUGGING
// #define GPUTI_GO_DEAP_HEAP
// static const int TESTING_ID = 219064;
// static const int TEST_SIZE = 1;

// TODO next when spliting time intervals, check if overlaps the current toi,
// then decide if we push it into the heap the reason of considerting it is that
// the limited heap size. token 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

// overflow instructions
static const int NO_OVERFLOW = 0;
static const int BISECTION_OVERFLOW = 1;
static const int HEAP_OVERFLOW = 2;
static const int ITERATION_OVERFLOW = 3;

class Singleinterval {
public:
  __device__ Singleinterval(){};
  __device__ Singleinterval(const Scalar &f, const Scalar &s);
  Scalar first;
  Scalar second;
  __device__ Singleinterval &operator=(const Singleinterval &x) {
    if (this == &x)
      return *this;
    first = x.first;
    second = x.second;
    return *this;
  }
};

class interval_pair {
public:
  __device__ interval_pair(const Singleinterval &itv);
  __device__ interval_pair(){};
  Singleinterval first;
  Singleinterval second;
};

void print_vector(Scalar *v, int size);
void print_vector(int *v, int size);

// the initialized error input, solve tolerance, time interval upper bound, etc.
class CCDConfig {
public:
  // Scalar err_in[3];           // the input error bound calculate from the
  // AABB of the whole mesh
  Scalar co_domain_tolerance; // tolerance of the co-domain
  // Scalar max_t;               // the upper bound of the time interval
  unsigned int mp_start;
  unsigned int mp_end;
  int mp_remaining;
  long unit_size;
  Scalar toi;
  cuda::binary_semaphore<cuda::thread_scope_device> mutex;
  bool use_ms;
  bool allow_zero_toi;
  int max_iter;
  int overflow_flag;
};

// this is to record the interval related info

class MP_unit {
public:
  Singleinterval itv[3];

  int query_id;
  // Scalar true_tol;
  // bool box_in;

  __device__ __host__ void init(int i) {
    itv[0].first = 0;
    itv[0].second = 1;
    itv[1].first = 0;
    itv[1].second = 1;
    itv[2].first = 0;
    itv[2].second = 1;
    query_id = i;
    // box_in = true; // same result if true or false
  }
  __device__ MP_unit &operator=(const MP_unit &x) {
    if (this == &x)
      return *this;
    itv[0] = x.itv[0];
    itv[1] = x.itv[1];
    itv[2] = x.itv[2];
    query_id = x.query_id;
    // box_in = x.box_in;
    // true_tol = x.true_tol;
    return *this;
  }
};

class CCDData {
public:
  __host__ __device__ CCDData(){};
  // CCDData(const std::array<std::array<Scalar,3>,8>&input);
  Scalar v0s[3];
  Scalar v1s[3];
  Scalar v2s[3];
  Scalar v3s[3];
  Scalar v0e[3];
  Scalar v1e[3];
  Scalar v2e[3];
  Scalar v3e[3];
  Scalar ms;     // minimum separation
  Scalar err[3]; // error bound of each query, calculated from each scene
  Scalar tol[3]; // domain tolerance to help decide which dimension to split
#ifdef CCD_TOI_PER_QUERY
  Scalar toi;
  int aid;
  int bid;
#endif
  int nbr_checks = 0;

  __device__ __host__ CCDData &operator=(const CCDData &x) {
    if (this == &x)
      return *this;
    for (int i = 0; i < 3; i++) {
      v0s[i] = x.v0s[i];
      v1s[i] = x.v1s[i];
      v2s[i] = x.v2s[i];
      v3s[i] = x.v3s[i];
      v0e[i] = x.v0e[i];
      v1e[i] = x.v1e[i];
      v2e[i] = x.v2e[i];
      v3e[i] = x.v3e[i];
      err[i] = x.err[i];
      tol[i] = x.tol[i];
    }
    ms = x.ms;
    return *this;
  }
};
// this is to calculate the vertices of the inclusion function
class BoxPrimatives {
public:
  bool b[3];
  int dim;
  Scalar t;
  Scalar u;
  Scalar v;
  __device__ void calculate_tuv(const MP_unit &unit);
};
CCDData array_to_ccd(const std::array<std::array<Scalar, 3>, 8> &a);
__device__ void single_test_wrapper(CCDData *vfdata, bool &result);
__device__ Scalar calculate_ee(const CCDData &data_in, const BoxPrimatives &bp);
} // namespace ccd