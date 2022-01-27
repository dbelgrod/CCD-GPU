#include <array>
#include <ccdgpu/rational.hpp>
#include <ccdgpu/record.hpp>
#include <ccdgpu/root_finder.cuh>
#include <float.h>
#include <vector>

#include <cuda/semaphore>

#include <spdlog/spdlog.h>

using namespace std;

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    spdlog::error("GPUassert: {} {} {:d}\n", cudaGetErrorString(code), file,
                  line);
    if (abort)
      exit(code);
  }
}

namespace ccd {

__device__ Singleinterval::Singleinterval(const Scalar &f, const Scalar &s) {
  first = f;
  second = s;
}

// this function do the bisection
__device__ interval_pair::interval_pair(const Singleinterval &itv) {
  Scalar c = (itv.first + itv.second) / 2;
  first.first = itv.first;
  first.second = c;
  second.first = c;
  second.second = itv.second;
}

__device__ bool sum_no_larger_1(const Scalar &num1, const Scalar &num2) {
#ifdef GPUTI_USE_DOUBLE_PRECISION
  if (num1 + num2 > 1 / (1 - DBL_EPSILON)) {
    return false;
  }
#else
  if (num1 + num2 > 1 / (1 - FLT_EPSILON)) {
    return false;
  }
#endif
  return true;
}
__device__ void
compute_face_vertex_tolerance_memory_pool(CCDdata &data_in,
                                          const CCDConfig &config) {
  Scalar p000[3], p001[3], p011[3], p010[3], p100[3], p101[3], p111[3], p110[3];
  for (int i = 0; i < 3; i++) {
    p000[i] = data_in.v0s[i] - data_in.v1s[i];
    p001[i] = data_in.v0s[i] - data_in.v3s[i];
    p011[i] =
        data_in.v0s[i] - (data_in.v2s[i] + data_in.v3s[i] - data_in.v1s[i]);
    p010[i] = data_in.v0s[i] - data_in.v2s[i];
    p100[i] = data_in.v0e[i] - data_in.v1e[i];
    p101[i] = data_in.v0e[i] - data_in.v3e[i];
    p111[i] =
        data_in.v0e[i] - (data_in.v2e[i] + data_in.v3e[i] - data_in.v1e[i]);
    p110[i] = data_in.v0e[i] - data_in.v2e[i];
  }
  Scalar dl = 0;
  for (int i = 0; i < 3; i++) {
    dl = max(dl, fabs(p100[i] - p000[i]));
    dl = max(dl, fabs(p101[i] - p001[i]));
    dl = max(dl, fabs(p111[i] - p011[i]));
    dl = max(dl, fabs(p110[i] - p010[i]));
  }
  dl *= 3;
  data_in.tol[0] = config.co_domain_tolerance / dl;

  dl = 0;
  for (int i = 0; i < 3; i++) {
    dl = max(dl, fabs(p010[i] - p000[i]));
    dl = max(dl, fabs(p110[i] - p100[i]));
    dl = max(dl, fabs(p111[i] - p101[i]));
    dl = max(dl, fabs(p011[i] - p001[i]));
  }
  dl *= 3;
  data_in.tol[1] = config.co_domain_tolerance / dl;

  dl = 0;
  for (int i = 0; i < 3; i++) {
    dl = max(dl, fabs(p001[i] - p000[i]));
    dl = max(dl, fabs(p101[i] - p100[i]));
    dl = max(dl, fabs(p111[i] - p110[i]));
    dl = max(dl, fabs(p011[i] - p010[i]));
  }
  dl *= 3;
  data_in.tol[2] = config.co_domain_tolerance / dl;
}
__device__ void
compute_edge_edge_tolerance_memory_pool(CCDdata &data_in,
                                        const CCDConfig &config) {
  Scalar p000[3], p001[3], p011[3], p010[3], p100[3], p101[3], p111[3], p110[3];
  for (int i = 0; i < 3; i++) {
    p000[i] = data_in.v0s[i] - data_in.v2s[i];
    p001[i] = data_in.v0s[i] - data_in.v3s[i];
    p011[i] = data_in.v1s[i] - data_in.v3s[i];
    p010[i] = data_in.v1s[i] - data_in.v2s[i];
    p100[i] = data_in.v0e[i] - data_in.v2e[i];
    p101[i] = data_in.v0e[i] - data_in.v3e[i];
    p111[i] = data_in.v1e[i] - data_in.v3e[i];
    p110[i] = data_in.v1e[i] - data_in.v2e[i];
  }
  Scalar dl = 0;
  for (int i = 0; i < 3; i++) {
    dl = max(dl, fabs(p100[i] - p000[i]));
    dl = max(dl, fabs(p101[i] - p001[i]));
    dl = max(dl, fabs(p111[i] - p011[i]));
    dl = max(dl, fabs(p110[i] - p010[i]));
  }
  dl *= 3;
  data_in.tol[0] = config.co_domain_tolerance / dl;

  dl = 0;
  for (int i = 0; i < 3; i++) {
    dl = max(dl, fabs(p010[i] - p000[i]));
    dl = max(dl, fabs(p110[i] - p100[i]));
    dl = max(dl, fabs(p111[i] - p101[i]));
    dl = max(dl, fabs(p011[i] - p001[i]));
  }
  dl *= 3;
  data_in.tol[1] = config.co_domain_tolerance / dl;

  dl = 0;
  for (int i = 0; i < 3; i++) {
    dl = max(dl, fabs(p001[i] - p000[i]));
    dl = max(dl, fabs(p101[i] - p100[i]));
    dl = max(dl, fabs(p111[i] - p110[i]));
    dl = max(dl, fabs(p011[i] - p010[i]));
  }
  dl *= 3;
  data_in.tol[2] = config.co_domain_tolerance / dl;
}

__device__ __host__ void get_numerical_error_vf_memory_pool(CCDdata &data_in,
                                                            bool use_ms) {
  Scalar vffilter;
  //   bool use_ms = false;
  if (!use_ms) {
#ifdef GPUTI_USE_DOUBLE_PRECISION
    vffilter = 6.661338147750939e-15;
#else
    vffilter = 3.576279e-06;
#endif
  } else {
#ifdef GPUTI_USE_DOUBLE_PRECISION
    vffilter = 7.549516567451064e-15;
#else
    vffilter = 4.053116e-06;
#endif
  }
  Scalar xmax = fabs(data_in.v0s[0]);
  Scalar ymax = fabs(data_in.v0s[1]);
  Scalar zmax = fabs(data_in.v0s[2]);

  xmax = max(xmax, fabs(data_in.v1s[0]));
  ymax = max(ymax, fabs(data_in.v1s[1]));
  zmax = max(zmax, fabs(data_in.v1s[2]));

  xmax = max(xmax, fabs(data_in.v2s[0]));
  ymax = max(ymax, fabs(data_in.v2s[1]));
  zmax = max(zmax, fabs(data_in.v2s[2]));

  xmax = max(xmax, fabs(data_in.v3s[0]));
  ymax = max(ymax, fabs(data_in.v3s[1]));
  zmax = max(zmax, fabs(data_in.v3s[2]));

  xmax = max(xmax, fabs(data_in.v0e[0]));
  ymax = max(ymax, fabs(data_in.v0e[1]));
  zmax = max(zmax, fabs(data_in.v0e[2]));

  xmax = max(xmax, fabs(data_in.v1e[0]));
  ymax = max(ymax, fabs(data_in.v1e[1]));
  zmax = max(zmax, fabs(data_in.v1e[2]));

  xmax = max(xmax, fabs(data_in.v2e[0]));
  ymax = max(ymax, fabs(data_in.v2e[1]));
  zmax = max(zmax, fabs(data_in.v2e[2]));

  xmax = max(xmax, fabs(data_in.v3e[0]));
  ymax = max(ymax, fabs(data_in.v3e[1]));
  zmax = max(zmax, fabs(data_in.v3e[2]));

  xmax = max(xmax, Scalar(1));
  ymax = max(ymax, Scalar(1));
  zmax = max(zmax, Scalar(1));

  data_in.err[0] = xmax * xmax * xmax * vffilter;
  data_in.err[1] = ymax * ymax * ymax * vffilter;
  data_in.err[2] = zmax * zmax * zmax * vffilter;
  return;
}
__device__ __host__ void get_numerical_error_ee_memory_pool(CCDdata &data_in,
                                                            bool use_ms) {
  Scalar vffilter;
  //   bool use_ms = false;
  if (!use_ms) {

#ifdef GPUTI_USE_DOUBLE_PRECISION
    vffilter = 6.217248937900877e-15;
#else
    vffilter = 3.337861e-06;
#endif
  } else {
#ifdef GPUTI_USE_DOUBLE_PRECISION
    vffilter = 7.105427357601002e-15;
#else
    vffilter = 3.814698e-06;
#endif
  }
  Scalar xmax = fabs(data_in.v0s[0]);
  Scalar ymax = fabs(data_in.v0s[1]);
  Scalar zmax = fabs(data_in.v0s[2]);

  xmax = max(xmax, fabs(data_in.v1s[0]));
  ymax = max(ymax, fabs(data_in.v1s[1]));
  zmax = max(zmax, fabs(data_in.v1s[2]));

  xmax = max(xmax, fabs(data_in.v2s[0]));
  ymax = max(ymax, fabs(data_in.v2s[1]));
  zmax = max(zmax, fabs(data_in.v2s[2]));

  xmax = max(xmax, fabs(data_in.v3s[0]));
  ymax = max(ymax, fabs(data_in.v3s[1]));
  zmax = max(zmax, fabs(data_in.v3s[2]));

  xmax = max(xmax, fabs(data_in.v0e[0]));
  ymax = max(ymax, fabs(data_in.v0e[1]));
  zmax = max(zmax, fabs(data_in.v0e[2]));

  xmax = max(xmax, fabs(data_in.v1e[0]));
  ymax = max(ymax, fabs(data_in.v1e[1]));
  zmax = max(zmax, fabs(data_in.v1e[2]));

  xmax = max(xmax, fabs(data_in.v2e[0]));
  ymax = max(ymax, fabs(data_in.v2e[1]));
  zmax = max(zmax, fabs(data_in.v2e[2]));

  xmax = max(xmax, fabs(data_in.v3e[0]));
  ymax = max(ymax, fabs(data_in.v3e[1]));
  zmax = max(zmax, fabs(data_in.v3e[2]));

  xmax = max(xmax, Scalar(1));
  ymax = max(ymax, Scalar(1));
  zmax = max(zmax, Scalar(1));

  data_in.err[0] = xmax * xmax * xmax * vffilter;
  data_in.err[1] = ymax * ymax * ymax * vffilter;
  data_in.err[2] = zmax * zmax * zmax * vffilter;
  return;
}

__device__ void BoxPrimatives::calculate_tuv(const MP_unit &unit) {
  if (b[0] == 0) { // t0
    t = unit.itv[0].first;
  } else { // t1
    t = unit.itv[0].second;
  }

  if (b[1] == 0) { // u0
    u = unit.itv[1].first;
  } else { // u1
    u = unit.itv[1].second;
  }

  if (b[2] == 0) { // v0
    v = unit.itv[2].first;
  } else { // v1
    v = unit.itv[2].second;
  }
}
__device__ Scalar calculate_vf(const CCDdata &data_in,
                               const BoxPrimatives &bp) {
  Scalar v, pt, t0, t1, t2;
  v = (data_in.v0e[bp.dim] - data_in.v0s[bp.dim]) * bp.t + data_in.v0s[bp.dim];
  t0 = (data_in.v1e[bp.dim] - data_in.v1s[bp.dim]) * bp.t + data_in.v1s[bp.dim];
  t1 = (data_in.v2e[bp.dim] - data_in.v2s[bp.dim]) * bp.t + data_in.v2s[bp.dim];
  t2 = (data_in.v3e[bp.dim] - data_in.v3s[bp.dim]) * bp.t + data_in.v3s[bp.dim];
  pt = (t1 - t0) * bp.u + (t2 - t0) * bp.v + t0;
  return (v - pt);
}
__device__ Scalar calculate_ee(const CCDdata &data_in,
                               const BoxPrimatives &bp) {
  Scalar edge0_vertex0 =
      (data_in.v0e[bp.dim] - data_in.v0s[bp.dim]) * bp.t + data_in.v0s[bp.dim];
  Scalar edge0_vertex1 =
      (data_in.v1e[bp.dim] - data_in.v1s[bp.dim]) * bp.t + data_in.v1s[bp.dim];
  Scalar edge1_vertex0 =
      (data_in.v2e[bp.dim] - data_in.v2s[bp.dim]) * bp.t + data_in.v2s[bp.dim];
  Scalar edge1_vertex1 =
      (data_in.v3e[bp.dim] - data_in.v3s[bp.dim]) * bp.t + data_in.v3s[bp.dim];
  Scalar result = ((edge0_vertex1 - edge0_vertex0) * bp.u + edge0_vertex0) -
                  ((edge1_vertex1 - edge1_vertex0) * bp.v + edge1_vertex0);

  return result;
}

inline __device__ bool Origin_in_vf_inclusion_function_memory_pool(
    const CCDdata &data_in, MP_unit &unit, Scalar &true_tol, bool &box_in) {
  box_in = true;
  true_tol = 0.0;
  BoxPrimatives bp;
  Scalar vmin = SCALAR_LIMIT;
  Scalar vmax = -SCALAR_LIMIT;
  Scalar value;
  for (bp.dim = 0; bp.dim < 3; bp.dim++) {
    vmin = SCALAR_LIMIT;
    vmax = -SCALAR_LIMIT;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          bp.b[0] = i;
          bp.b[1] = j;
          bp.b[2] = k; // 100
          bp.calculate_tuv(unit);
          value = calculate_vf(data_in, bp);
          vmin = min(vmin, value);
          vmax = max(vmax, value);
        }
      }
    }

    // get the min and max in one dimension
    true_tol = max(true_tol, vmax - vmin);

    if (vmin - data_in.ms > data_in.err[bp.dim] ||
        vmax + data_in.ms < -data_in.err[bp.dim]) {
      return false;
    }

    if (vmin + data_in.ms < -data_in.err[bp.dim] ||
        vmax - data_in.ms > data_in.err[bp.dim]) {
      box_in = false;
    }
  }
  return true;
}
inline __device__ bool Origin_in_ee_inclusion_function_memory_pool(
    const CCDdata &data_in, MP_unit &unit, Scalar &true_tol, bool &box_in) {
  box_in = true;
  true_tol = 0.0;
  BoxPrimatives bp;
  Scalar vmin = SCALAR_LIMIT;
  Scalar vmax = -SCALAR_LIMIT;
  Scalar value;
  for (bp.dim = 0; bp.dim < 3; bp.dim++) {
    vmin = SCALAR_LIMIT;
    vmax = -SCALAR_LIMIT;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          bp.b[0] = i;
          bp.b[1] = j;
          bp.b[2] = k; // 100
          bp.calculate_tuv(unit);
          value = calculate_ee(data_in, bp);
          vmin = min(vmin, value);
          vmax = max(vmax, value);
        }
      }
    }

    // get the min and max in one dimension
    true_tol = max(true_tol, vmax - vmin);

    if (vmin - data_in.ms > data_in.err[bp.dim] ||
        vmax + data_in.ms < -data_in.err[bp.dim]) {
      return false;
    }

    if (vmin + data_in.ms < -data_in.err[bp.dim] ||
        vmax - data_in.ms > data_in.err[bp.dim]) {
      box_in = false;
    }
  }
  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the memory pool method
__global__ void compute_vf_tolerance_memory_pool(CCDdata *data,
                                                 CCDConfig *config,
                                                 const int query_size) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tx >= query_size)
    return;

  // release the mutex here before real calculations
  config[0].mutex.release();

  compute_face_vertex_tolerance_memory_pool(data[tx], config[0]);

  data[tx].nbr_checks = 0;
  get_numerical_error_vf_memory_pool(data[tx], config[0].use_ms);
}
__global__ void compute_ee_tolerance_memory_pool(CCDdata *data,
                                                 CCDConfig *config,
                                                 const int query_size) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tx >= query_size)
    return;

  // release the mutex here before real calculations
  config[0].mutex.release();

  compute_edge_edge_tolerance_memory_pool(data[tx], config[0]);

  data[tx].nbr_checks = 0;
  get_numerical_error_ee_memory_pool(data[tx], config[0].use_ms);
}

__global__ void initialize_memory_pool(MP_unit *units, int query_size) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tx >= query_size)
    return;
  units[tx].init(tx);
}
__device__ int
split_dimension_memory_pool(const CCDdata &data,
                            Scalar width[3]) { // clarified in queue.h
  int split = 0;
  Scalar res[3];
  res[0] = width[0] / data.tol[0];
  res[1] = width[1] / data.tol[1];
  res[2] = width[2] / data.tol[2];
  if (res[0] >= res[1] && res[0] >= res[2]) {
    split = 0;
  }
  if (res[1] >= res[0] && res[1] >= res[2]) {
    split = 1;
  }
  if (res[2] >= res[1] && res[2] >= res[0]) {
    split = 2;
  }
  return split;
}

inline __device__ bool bisect_vf_memory_pool(const MP_unit &unit, int split,
                                             CCDConfig *config,
#ifdef CCD_TOI_PER_QUERY
                                             Scalar data_toi,
#endif
                                             MP_unit *out) {
  interval_pair halves(unit.itv[split]); // bisected

  if (halves.first.first >= halves.first.second) {
    // valid_nbr = 0;
    return true;
  }
  if (halves.second.first >= halves.second.second) {
    // valid_nbr = 0;
    return true;
  }
  // bisected[0] = unit;
  // bisected[1] = unit;
  // valid_nbr = 1;

  int unit_id = atomicInc(&config[0].mp_end, config[0].unit_size - 1);
  out[unit_id] = unit;
  out[unit_id].itv[split] = halves.first;

  if (split == 0) {
#ifndef CCD_TOI_PER_QUERY
    if (halves.second.first <= config[0].toi) {
#else
    if (halves.second.first <= data_toi) {
#endif
      unit_id = atomicInc(&config[0].mp_end, config[0].unit_size - 1);
      out[unit_id] = unit;
      out[unit_id].itv[split] = halves.second;
    }
  } else if (split == 1) {
    if (sum_no_larger_1(halves.second.first,
                        unit.itv[2].first)) // check if u+v<=1
    {
      unit_id = atomicInc(&config[0].mp_end, config[0].unit_size - 1);
      out[unit_id] = unit;
      out[unit_id].itv[1] = halves.second;
      // valid_nbr = 2;
    }
  } else if (split == 2) {
    if (sum_no_larger_1(halves.second.first,
                        unit.itv[1].first)) // check if u+v<=1
    {
      unit_id = atomicInc(&config[0].mp_end, config[0].unit_size - 1);
      out[unit_id] = unit;
      out[unit_id].itv[2] = halves.second;
      // valid_nbr = 2;
    }
  }
  return false;
}
inline __device__ bool bisect_ee_memory_pool(const MP_unit &unit, int split,
                                             CCDConfig *config,
#ifdef CCD_TOI_PER_QUERY
                                             Scalar data_toi,
#endif
                                             MP_unit *out) {
  interval_pair halves(unit.itv[split]); // bisected

  if (halves.first.first >= halves.first.second) {
    // valid_nbr = 0;
    return true;
  }
  if (halves.second.first >= halves.second.second) {
    // valid_nbr = 0;
    return true;
  }
  // bisected[0] = unit;
  // bisected[1] = unit;
  // valid_nbr = 1;

  int unit_id = atomicInc(&config[0].mp_end, config[0].unit_size - 1);
  out[unit_id] = unit;
  out[unit_id].itv[split] = halves.first;

  if (split == 0) // split the time interval
  {
#ifndef CCD_TOI_PER_QUERY
    if (halves.second.first <= config[0].toi) {
#else
    if (halves.second.first <= data_toi) {
#endif
      unit_id = atomicInc(&config[0].mp_end, config[0].unit_size - 1);
      out[unit_id] = unit;
      out[unit_id].itv[split] = halves.second;
    }
  } else {

    unit_id = atomicInc(&config[0].mp_end, config[0].unit_size - 1);
    out[unit_id] = unit;
    out[unit_id].itv[split] = halves.second;
    // valid_nbr = 2;
  }

  return false;
}

inline __device__ void
mutex_update_min(cuda::binary_semaphore<cuda::thread_scope_device> &mutex,
                 Scalar &value, const Scalar &compare) {
  mutex.acquire();
  value = compare < value ? compare : value; // if compare is smaller, update it
  mutex.release();
}

__global__ void vf_ccd_memory_pool(MP_unit *units, int query_size,
                                   CCDdata *data, CCDConfig *config) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tx >= config[0].mp_remaining)
    return;

  //   bool allow_zero_toi = true;
  int qid = (tx + config[0].mp_start) % config[0].unit_size;

  Scalar widths[3];
  bool condition;
  // int split;

  MP_unit units_in = units[qid];
  int box_id = units_in.query_id;
  CCDdata data_in = data[box_id];

  atomicAdd(&data[box_id].nbr_checks, 1);

  const Scalar time_left = units_in.itv[0].first; // the time of this unit

// if the time is larger than toi, return
#ifndef CCD_TOI_PER_QUERY
  if (time_left >= config[0].toi) {
    return;
  }
#else
  if (time_left >= data_in.toi)
    return;
#endif
  // if (results[box_id] > 0)
  // { // if it is sure that have root, then no need to check
  // 	return;
  // }
  if (config[0].max_iter >= 0 &&
      data_in.nbr_checks > config[0].max_iter) // max checks
  {
    // if (!config[0].overflow_flag)
    //   atomicAdd(&config[0].overflow_flag, 1);
    return;
  } else if (config[0].mp_remaining > config[0].unit_size / 2) // overflow
  {
    if (!config[0].overflow_flag)
      atomicAdd(&config[0].overflow_flag, 1);
    return;
  }

  Scalar true_tol = 0;
  bool box_in;

  const bool zero_in = Origin_in_vf_inclusion_function_memory_pool(
      data_in, units_in, true_tol, box_in);
  if (zero_in) {
    widths[0] = units_in.itv[0].second - units_in.itv[0].first;
    widths[1] = units_in.itv[1].second - units_in.itv[1].first;
    widths[2] = units_in.itv[2].second - units_in.itv[2].first;

    // Condition 1
    condition = widths[0] <= data_in.tol[0] && widths[1] <= data_in.tol[1] &&
                widths[2] <= data_in.tol[2];
    if (condition) {
      mutex_update_min(config[0].mutex, config[0].toi, time_left);
      // results[box_id] = 1;

#ifdef CCD_TOI_PER_QUERY
      mutex_update_min(config[0].mutex, data[box_id].toi, time_left);
#endif
      return;
    }
    // Condition 2, the box is inside the epsilon box, have a root, return true;
    // condition = units_in.box_in;

    if (box_in && (config[0].allow_zero_toi || time_left > 0)) {
      mutex_update_min(config[0].mutex, config[0].toi, time_left);
      // results[box_id] = 1;

#ifdef CCD_TOI_PER_QUERY
      mutex_update_min(config[0].mutex, data[box_id].toi, time_left);
#endif
      return;
    }

    // Condition 3, real tolerance is smaller than the input tolerance, return
    // true
    condition = true_tol <= config->co_domain_tolerance;
    if (condition && (config[0].allow_zero_toi || time_left > 0)) {
      mutex_update_min(config[0].mutex, config[0].toi, time_left);
      // results[box_id] = 1;

#ifdef CCD_TOI_PER_QUERY
      mutex_update_min(config[0].mutex, data[box_id].toi, time_left);
#endif
      return;
    }
    const int split = split_dimension_memory_pool(data_in, widths);

#ifndef CCD_TOI_PER_QUERY
    const bool sure_in = bisect_vf_memory_pool(units_in, split, config, units);
#else
    const bool sure_in =
        bisect_vf_memory_pool(units_in, split, config, data_in.toi, units);
#endif

    if (sure_in) // in this case, the interval is too small that overflow
                 // happens. it should be rare to happen
    {
      mutex_update_min(config[0].mutex, config[0].toi, time_left);
      // results[box_id] = 1;

#ifdef CCD_TOI_PER_QUERY
      mutex_update_min(config[0].mutex, data[box_id].toi, time_left);
#endif
      return;
    }
  }
}
__global__ void ee_ccd_memory_pool(MP_unit *units, int query_size,
                                   CCDdata *data, CCDConfig *config) {
  //   bool allow_zero_toi = true;

  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tx >= config[0].mp_remaining)
    return;

  int qid = (tx + config[0].mp_start) % config[0].unit_size;

  Scalar widths[3];
  bool condition;

  MP_unit units_in = units[qid];
  int box_id = units_in.query_id;
  CCDdata data_in = data[box_id];

  atomicAdd(&data[box_id].nbr_checks, 1);

  const Scalar time_left = units_in.itv[0].first; // the time of this unit

// if the time is larger than toi, return
#ifndef CCD_TOI_PER_QUERY
  if (time_left >= config[0].toi) {
    return;
  }
#else
  if (time_left >= data_in.toi)
    return;
#endif
  // if (results[box_id] > 0)
  // { // if it is sure that have root, then no need to check
  // 	return;
  // }
  if (config[0].max_iter >= 0 &&
      data_in.nbr_checks > config[0].max_iter) // max checks
  {
    // if (!config[0].overflow_flag)
    //   atomicAdd(&config[0].overflow_flag, 1);
    return;
  } else if (config[0].mp_remaining > config[0].unit_size / 2) // overflow
  {
    if (!config[0].overflow_flag)
      atomicAdd(&config[0].overflow_flag, 1);
    return;
  }

  Scalar true_tol = 0;
  bool box_in;

  const bool zero_in = Origin_in_ee_inclusion_function_memory_pool(
      data_in, units_in, true_tol, box_in);
  if (zero_in) {
    widths[0] = units_in.itv[0].second - units_in.itv[0].first;
    widths[1] = units_in.itv[1].second - units_in.itv[1].first;
    widths[2] = units_in.itv[2].second - units_in.itv[2].first;

    // Condition 1
    condition = widths[0] <= data_in.tol[0] && widths[1] <= data_in.tol[1] &&
                widths[2] <= data_in.tol[2];
    if (condition) {
      mutex_update_min(config[0].mutex, config[0].toi, time_left);
      // results[box_id] = 1;

#ifdef CCD_TOI_PER_QUERY
      mutex_update_min(config[0].mutex, data[box_id].toi, time_left);
#endif
      return;
    }
    // Condition 2, the box is inside the epsilon box, have a root, return true;
    if (box_in && (config[0].allow_zero_toi || time_left > 0)) {
      mutex_update_min(config[0].mutex, config[0].toi, time_left);
      // results[box_id] = 1;

#ifdef CCD_TOI_PER_QUERY
      mutex_update_min(config[0].mutex, data[box_id].toi, time_left);
#endif
      return;
    }

    // Condition 3, real tolerance is smaller than the input tolerance, return
    // true
    condition = true_tol <= config->co_domain_tolerance;
    if (condition && (config[0].allow_zero_toi || time_left > 0)) {
      mutex_update_min(config[0].mutex, config[0].toi, time_left);
      // results[box_id] = 1;

#ifdef CCD_TOI_PER_QUERY
      mutex_update_min(config[0].mutex, data[box_id].toi, time_left);
#endif
      return;
    }
    const int split = split_dimension_memory_pool(data_in, widths);

#ifndef CCD_TOI_PER_QUERY
    const bool sure_in = bisect_ee_memory_pool(units_in, split, config, units);
#else
    const bool sure_in =
        bisect_ee_memory_pool(units_in, split, config, data_in.toi, units);
#endif

    if (sure_in) // in this case, the interval is too small that overflow
                 // happens. it should be rare to happen
    {
      mutex_update_min(config[0].mutex, config[0].toi, time_left);
      // results[box_id] = 1;

#ifdef CCD_TOI_PER_QUERY
      mutex_update_min(config[0].mutex, data[box_id].toi, time_left);
#endif

      return;
    }
  }
}

__global__ void shift_queue_pointers(CCDConfig *config) {
  config[0].mp_start += config[0].mp_remaining;
  config[0].mp_start = config[0].mp_start % config[0].unit_size;
  config[0].mp_remaining = (config[0].mp_end - config[0].mp_start);
  config[0].mp_remaining =
      config[0].mp_remaining < 0
          ? config[0].mp_end + config[0].unit_size - config[0].mp_start
          : config[0].mp_remaining;
}

void run_memory_pool_ccd(CCDdata *d_data_list, int tmp_nbr, bool is_edge,
                         std::vector<int> &result_list, int parallel_nbr,
                         int max_iter, ccd::Scalar tol, bool use_ms,
                         bool allow_zero_toi, ccd::Scalar &toi,
                         ccdgpu::Record &r) {
  int nbr = tmp_nbr;
  spdlog::trace("tmp_nbr {}", tmp_nbr);
  // int *res = new int[nbr];
  CCDConfig *config = new CCDConfig[1];
  // config[0].err_in[0] =
  //     -1; // the input error bound calculate from the AABB of the whole
  //     mesh
  config[0].co_domain_tolerance = tol; // tolerance of the co-domain
  // config[0].max_t = 1;                  // the upper bound of the time

  // interval
  config[0].toi = toi;
  config[0].mp_end = nbr;
  config[0].mp_start = 0;
  config[0].mp_remaining = nbr;
  config[0].overflow_flag = 0;
  config[0].unit_size = nbr * 16; // 2.0 * nbr;
  config[0].use_ms = use_ms;
  config[0].allow_zero_toi = allow_zero_toi;
  config[0].max_iter = max_iter;
  spdlog::trace("unit_size : {:d}", config[0].unit_size);

  // int *d_res;
  MP_unit *d_units;
  CCDConfig *d_config;

  size_t unit_size = sizeof(MP_unit) * config[0].unit_size; // arbitrary #

  // size_t result_size = sizeof(int) * nbr;

  // cudaMalloc(&d_res, result_size);
  cudaMalloc(&d_units, unit_size);
  cudaMalloc(&d_config, sizeof(CCDConfig));
  cudaMemcpy(d_config, config, sizeof(CCDConfig), cudaMemcpyHostToDevice);
  gpuErrchk(cudaGetLastError());
  // ccd::Timer timer;
  // timer.start();
  spdlog::trace("nbr: {:d}, parallel_nbr {:d}", nbr, parallel_nbr);
  initialize_memory_pool<<<nbr / parallel_nbr + 1, parallel_nbr>>>(d_units,
                                                                   nbr);

  cudaDeviceSynchronize();
  if (is_edge) {
    compute_ee_tolerance_memory_pool<<<nbr / parallel_nbr + 1, parallel_nbr>>>(
        d_data_list, d_config, nbr);
  } else {
    compute_vf_tolerance_memory_pool<<<nbr / parallel_nbr + 1, parallel_nbr>>>(
        d_data_list, d_config, nbr);
  }
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  spdlog::trace("MAX_OVERLAP_SIZE: {:d}", MAX_OVERLAP_SIZE);
  spdlog::trace("sizeof(Scalar) {:d}", sizeof(ccd::Scalar));

  int nbr_per_loop = nbr;
  int start;
  int end;

  spdlog::trace("Queue size t0: {:d}", nbr_per_loop);
  while (nbr_per_loop > 0) {
    if (is_edge) {
      ee_ccd_memory_pool<<<nbr_per_loop / parallel_nbr + 1, parallel_nbr>>>(
          d_units, nbr, d_data_list, d_config);
    } else {
      vf_ccd_memory_pool<<<nbr_per_loop / parallel_nbr + 1, parallel_nbr>>>(
          d_units, nbr, d_data_list, d_config);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    shift_queue_pointers<<<1, 1>>>(d_config);
    cudaDeviceSynchronize();
    cudaMemcpy(&nbr_per_loop, &d_config[0].mp_remaining, sizeof(int),
               cudaMemcpyDeviceToHost);
    // cudaMemcpy(&start, &d_config[0].mp_start, sizeof(int),
    //            cudaMemcpyDeviceToHost);
    // cudaMemcpy(&end, &d_config[0].mp_end, sizeof(int),
    // cudaMemcpyDeviceToHost); cudaMemcpy(&toi, &d_config[0].toi,
    // sizeof(ccd::Scalar),
    //            cudaMemcpyDeviceToHost);
    // spdlog::trace("toi {}", toi);
    // spdlog::trace("toi {:.4f}",  toi);
    // spdlog::trace("Start {:d}, End {:d}, Queue size: {:d}",  start, end,
    // nbr_per_loop);
    gpuErrchk(cudaGetLastError());
    spdlog::trace("Queue size: {:d}", nbr_per_loop);
  }
  cudaDeviceSynchronize();
  // double tt = timer.getElapsedTimeInMicroSec();
  // run_time += tt / 1000.0f;
  gpuErrchk(cudaGetLastError());

  // cudaMemcpy(res, d_res, result_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&toi, &d_config[0].toi, sizeof(ccd::Scalar),
             cudaMemcpyDeviceToHost);
  int overflow;
  cudaMemcpy(&overflow, &d_config[0].overflow_flag, sizeof(int),
             cudaMemcpyDeviceToHost);
  if (overflow) {
    spdlog::error("OVERFLOW!!!!");
    abort();
  }

  gpuErrchk(cudaFree(d_units));
  gpuErrchk(cudaFree(d_config));

  // for (size_t i = 0; i < nbr; i++) {
  //   result_list[i] = res[i];
  // }

  // delete[] res;
  delete[] config;
  cudaError_t ct = cudaGetLastError();
  spdlog::trace("\n******************\n{}\n******************",
                cudaGetErrorString(ct));

#ifdef CCD_TOI_PER_QUERY
  CCDdata *data_list = new CCDdata[tmp_nbr];
  // CCDConfig *config = new CCDConfig[1];
  cudaMemcpy(data_list, d_data_list, sizeof(CCDdata) * tmp_nbr,
             cudaMemcpyDeviceToHost);
  // std::vector<std::pair<std::string, std::string>> symbolic_tois;
  int tpq_cnt = 0;
  for (size_t i = 0; i < tmp_nbr; i++) {
    ccdgpu::Rational ra(data_list[i].toi);
    if (data_list[i].toi > 1)
      continue;
    tpq_cnt++;
    // symbolic_tois.emplace_back(ra.get_numerator_str(),
    //                            ra.get_denominator_str());
    // auto pair = make_pair(ra.get_numerator_str(),
    // ra.get_denominator_str());
    std::string triple[4] = {std::to_string(data_list[i].aid),
                             std::to_string(data_list[i].bid),
                             ra.get_numerator_str(), ra.get_denominator_str()};
    // if (data_list[i].toi <= .00000382)
    //   printf("not one toi %s, %s, %e\n", triple[0].c_str(),
    //   triple[1].c_str(),
    //          data_list[i].toi);
    r.j_object["toi_per_query"].push_back(triple);
  }
  spdlog::debug("tpq_cnt: {:d}", tpq_cnt);
  free(data_list);
  cudaDeviceSynchronize();
  // json jtmp(symbolic_tois.begin(), symbolic_tois.end());
  // std::cout << jtmp.dump(4) << std::endl;
  // r.j_object.insert(jtmp.begin(), jtmp.end());
  // r.j_object.push_back(r.j_object.end(), jtmp.begin(), jtmp.end());
  // r.j_object.push_back(symbolic_tois);
  //  symbolic_tois.end());

  // json j_vec(falseNegativePairs);
  // r.j_object.insert(r.j_object.end(), symbolic_tois.begin(),
  //                   symbolic_tois.end());

  // std::ofstream o(outputFilePath);
  // o << std::setw(4) << j_vec << std::endl;
  // auto outputFilename = std::filesystem::path(std::to_string(iter) +
  // ".json"); outputFilename = outputFolder / outputFilename; std::ofstream
  // o(outputFilename); o << std::setw(4) << j << std::endl;
#endif
  gpuErrchk(cudaFree(d_data_list));
  return;
}

} // namespace ccd
