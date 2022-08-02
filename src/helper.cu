#include <ccdgpu/CType.cuh>
#include <ccdgpu/helper.cuh>

#include <fstream>
#include <iostream>

// #include <gputi/book.h>
// #include <gputi/io.h>
#include <ccdgpu/root_finder.cuh>
#include <ccdgpu/timer.hpp>
#include <stq/gpu/io.cuh>

#include <ccdgpu/record.hpp>
#include <stq/gpu/memory.cuh>
#include <stq/gpu/simulation.cuh>

#include <spdlog/spdlog.h>

using namespace std;
using namespace stq::gpu;

namespace ccd::gpu {

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    spdlog::error("GPUassert: {} {} {:d}", cudaGetErrorString(code), file,
                  line);
    if (abort)
      exit(code);
  }
}

// Allocates and copies data to GPU
template <typename T> T *copy_to_gpu(const T *cpu_data, const int size) {
  T *gpu_data;
  gpuErrchk(cudaMalloc((void **)&gpu_data, sizeof(T) * size));
  gpuErrchk(
    cudaMemcpy(gpu_data, cpu_data, sizeof(T) * size, cudaMemcpyHostToDevice));
  return gpu_data;
}

__global__ void split_overlaps(const int2 *const overlaps,
                               const stq::gpu::Aabb *const boxes, int N,
                               int2 *vf_overlaps, int2 *ee_overlaps,
                               int *vf_count, int *ee_count) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N)
    return;

  int minner = min(overlaps[tid].x, overlaps[tid].y);
  int maxxer = max(overlaps[tid].x, overlaps[tid].y);
  int3 avids = boxes[minner].vertexIds;
  int3 bvids = boxes[maxxer].vertexIds;

  if (is_vertex(avids) && is_face(bvids)) {
    int i = atomicAdd(vf_count, 1);
    vf_overlaps[i].x = minner;
    vf_overlaps[i].y = maxxer;
  } else if (is_edge(avids) && is_edge(bvids)) {
    int j = atomicAdd(ee_count, 1);
    ee_overlaps[j].x = minner;
    ee_overlaps[j].y = maxxer;
  } else
    assert(false);
}

__global__ void addData(const int2 *const overlaps,
                        const stq::gpu::Aabb *const boxes,
                        const Scalar *const V0, const Scalar *const V1,
                        int Vrows, int N, Scalar ms, CCDData *data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N)
    return;

  data[tid].ms = ms;

  // spdlog::trace("vf_count {:d}, ee_count {:d}", *vf_count, *ee_count);

  int minner = min(overlaps[tid].x, overlaps[tid].y);
  int maxxer = max(overlaps[tid].x, overlaps[tid].y);
  int3 avids = boxes[minner].vertexIds;
  int3 bvids = boxes[maxxer].vertexIds;

#ifdef CCD_TOI_PER_QUERY
  data[tid].toi = std::numeric_limits<Scalar>::infinity();
  // data[tid].id = shift + tid;
  data[tid].aid = minner;
  data[tid].bid = maxxer;
#endif

  if (is_vertex(avids) && is_face(bvids)) {
    for (size_t i = 0; i < 3; i++) {
      data[tid].v0s[i] = V0[avids.x + i * Vrows];
      data[tid].v1s[i] = V0[bvids.x + i * Vrows];
      data[tid].v2s[i] = V0[bvids.y + i * Vrows];
      data[tid].v3s[i] = V0[bvids.z + i * Vrows];
      data[tid].v0e[i] = V1[avids.x + i * Vrows];
      data[tid].v1e[i] = V1[bvids.x + i * Vrows];
      data[tid].v2e[i] = V1[bvids.y + i * Vrows];
      data[tid].v3e[i] = V1[bvids.z + i * Vrows];
    }
  } else if (is_edge(avids) && is_edge(bvids)) {
    for (size_t i = 0; i < 3; i++) {
      data[tid].v0s[i] = V0[avids.x + i * Vrows];
      data[tid].v1s[i] = V0[avids.y + i * Vrows];
      data[tid].v2s[i] = V0[bvids.x + i * Vrows];
      data[tid].v3s[i] = V0[bvids.y + i * Vrows];
      data[tid].v0e[i] = V1[avids.x + i * Vrows];
      data[tid].v1e[i] = V1[avids.y + i * Vrows];
      data[tid].v2e[i] = V1[bvids.x + i * Vrows];
      data[tid].v3e[i] = V1[bvids.y + i * Vrows];
    }
  } else
    assert(false);
}

void run_narrowphase(int2 *d_overlaps, Aabb *d_boxes,
                     stq::gpu::MemHandler *memhandle, int count,
                     Scalar *d_vertices_t0, Scalar *d_vertices_t1, int Vrows,
                     int threads, int max_iter, Scalar tol, Scalar ms,
                     bool allow_zero_toi, vector<int> &result_list, Scalar &toi,
                     Record &r) {
  bool use_ms = ms > 0;

  int *d_vf_count;
  int *d_ee_count;

  int2 *d_vf_overlaps;
  int2 *d_ee_overlaps;

  int start_id = 0;
  int size = count;
  memhandle->MAX_QUERIES = size;

  // double tavg = 0;
  // double tmp_tall = 0;

  size_t remain;
  spdlog::trace("remain {:d}, size {:d}", remain, size);
  while ((remain = size - start_id) > 0
#ifndef CCD_TOI_PER_QUERY
         && toi > 0
#endif
  ) {
    spdlog::trace("remain {:d}, start_id {:d}", remain, start_id);

    int overflow = 1; // run at least once
    int itr = 0;
    int tmp_nbr;
    while (overflow) {
      tmp_nbr = std::min(remain, memhandle->MAX_QUERIES);

      spdlog::debug("itr {:d}", itr);
      if (itr == 0) {
        memhandle->handleNarrowPhase(tmp_nbr);
      } else {
        memhandle->handleOverflow(tmp_nbr);
      }

      itr++;

      r.Start("splitOverlaps", /*gpu=*/true);
      gpuErrchk(cudaMalloc((void **)&d_vf_count, sizeof(int)));
      gpuErrchk(cudaMalloc((void **)&d_ee_count, sizeof(int)));

      gpuErrchk(cudaMemset(d_vf_count, 0, sizeof(int)));
      gpuErrchk(cudaMemset(d_ee_count, 0, sizeof(int)));

      gpuErrchk(cudaMalloc((void **)&d_vf_overlaps, sizeof(int2) * tmp_nbr));
      gpuErrchk(cudaMalloc((void **)&d_ee_overlaps, sizeof(int2) * tmp_nbr));

      split_overlaps<<<tmp_nbr / threads + 1, threads>>>(
        d_overlaps + start_id, d_boxes, tmp_nbr, d_vf_overlaps, d_ee_overlaps,
        d_vf_count, d_ee_count);
      gpuErrchk(cudaDeviceSynchronize());
      r.Stop();

      r.Start("createDataList", /*gpu=*/true);
      int vf_size;
      int ee_size;
      gpuErrchk(
        cudaMemcpy(&vf_size, d_vf_count, sizeof(int), cudaMemcpyDeviceToHost));
      gpuErrchk(
        cudaMemcpy(&ee_size, d_ee_count, sizeof(int), cudaMemcpyDeviceToHost));
      spdlog::trace("vf_size {} ee_size {}", vf_size, ee_size);

      CCDData *d_ee_data_list;
      CCDData *d_vf_data_list;

      size_t ee_data_size = sizeof(CCDData) * ee_size;
      size_t vf_data_size = sizeof(CCDData) * vf_size;

      gpuErrchk(cudaMalloc((void **)&d_ee_data_list, ee_data_size));
      gpuErrchk(cudaMalloc((void **)&d_vf_data_list, vf_data_size));
      spdlog::trace("ee_data_size {:d}", ee_data_size);
      spdlog::trace("vf_data_size {:d}", vf_data_size);

      addData<<<vf_size / threads + 1, threads>>>(
        d_vf_overlaps, d_boxes, d_vertices_t0, d_vertices_t1, Vrows, vf_size,
        ms, d_vf_data_list);
      gpuErrchk(cudaDeviceSynchronize());
      addData<<<ee_size / threads + 1, threads>>>(
        d_ee_overlaps, d_boxes, d_vertices_t0, d_vertices_t1, Vrows, ee_size,
        ms, d_ee_data_list);
      gpuErrchk(cudaDeviceSynchronize());

      r.Stop();

      gpuErrchk(cudaFree(d_vf_overlaps));
      gpuErrchk(cudaFree(d_ee_overlaps));
      gpuErrchk(cudaFree(d_vf_count));
      gpuErrchk(cudaFree(d_ee_count));

      spdlog::trace("vf_size {:d}, ee_size {:d}", vf_size, ee_size);

      // int size = count;
      // spdlog::trace("data loaded, size {}", queries.size());
      spdlog::trace("data loaded, size {}", size);

      // result_list.resize(size);

      int parallel = 64;
      spdlog::trace("run_memory_pool_ccd using {:d} threads", parallel);
      r.Start("run_memory_pool_ccd (narrowphase)", /*gpu=*/true);
      // toi = 1;
      run_memory_pool_ccd(d_vf_data_list, memhandle, vf_size,
                          /*is_edge_edge=*/false, result_list, parallel,
                          max_iter, tol, use_ms, allow_zero_toi, toi, overflow,
                          r);

      gpuErrchk(cudaDeviceSynchronize());

      r.Stop();
      if (overflow) // rerun
      {
        spdlog::warn("overflow after vf");
        gpuErrchk(cudaFree(d_ee_data_list));
        continue;
      }
      spdlog::trace("toi after vf {:e}", toi);
      // spdlog::trace("time after vf {:.6f}",  tmp_tall);
      r.Start("run_memory_pool_ccd (narrowphase)", /*gpu=*/true);
      run_memory_pool_ccd(d_ee_data_list, memhandle, ee_size,
                          /*is_edge_edge=*/true, result_list, parallel,
                          max_iter, tol, use_ms, allow_zero_toi, toi, overflow,
                          r);
      gpuErrchk(cudaDeviceSynchronize());
      r.Stop();
      spdlog::trace("toi after ee {:e}", toi);
      if (overflow)
        spdlog::warn("overflow after ee");
    }

    start_id += tmp_nbr;
  }
}

void run_ccd(const vector<Aabb> boxes, stq::gpu::MemHandler *memhandle,
             const Eigen::MatrixXd &vertices_t0,
             const Eigen::MatrixXd &vertices_t1, Record &r, int N, int &nbox,
             int &parallel, int &devcount, int &limitGB,
             vector<pair<int, int>> &overlaps, vector<int> &result_list,
             bool &allow_zero_toi, Scalar &ms, Scalar &toi) {
  toi = 1;
  bool use_ms = ms > 0;

  int tidstart = 0;

  int bpthreads = 32; // HARDCODING THREADS FOR NOW
  int npthreads = 1024;

  int2 *d_overlaps;
  int *d_count;
  size_t tot_count = 0;
  while (N > tidstart && toi > 0) {

    spdlog::trace("Next loop: N {:d}, tidstart {:d}", N, tidstart);

    r.Start("run_sweep_sharedqueue (broadphase)", /*gpu=*/true);
    run_sweep_sharedqueue(boxes.data(), memhandle, N, nbox, overlaps,
                          d_overlaps, d_count, bpthreads, tidstart, devcount,
                          limitGB);
    r.Stop();

    spdlog::trace("First run end {:d}", tidstart);
    // memhandle->increaseOverlapCutoff(2);
    spdlog::trace("Next cutoff {:d}", memhandle->MAX_OVERLAP_CUTOFF);

    spdlog::trace("Threads now {:d}", npthreads);

    r.Start("copyBoxesToGpu", /*gpu=*/true);

    int count;
    gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    tot_count += count;
    spdlog::trace("Count {:d}", count);

    Aabb *d_boxes = copy_to_gpu(boxes.data(), boxes.size());
    r.Stop();

    r.Start("copyVerticesToGpu", /*gpu=*/true);
    spdlog::trace("Copying vertices");
    double *d_vertices_t0 = copy_to_gpu(vertices_t0.data(), vertices_t0.size());
    double *d_vertices_t1 = copy_to_gpu(vertices_t1.data(), vertices_t1.size());
    r.Stop();
    int Vrows = vertices_t0.rows();
    assert(Vrows == vertices_t1.rows());

    int max_iter = -1;
    Scalar tolerance = 1e-6;

    run_narrowphase(d_overlaps, d_boxes, memhandle, count, d_vertices_t0,
                    d_vertices_t1, Vrows, npthreads, max_iter, tolerance, ms,
                    allow_zero_toi, result_list, toi, r);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_count));
    gpuErrchk(cudaFree(d_overlaps));
    gpuErrchk(cudaFree(d_boxes));
    gpuErrchk(cudaFree(d_vertices_t0));
    gpuErrchk(cudaFree(d_vertices_t1));

    gpuErrchk(cudaGetLastError());

    cudaDeviceSynchronize();
  }
  spdlog::info("Total count {:d}", tot_count);
  spdlog::info("LimitGB {:d}", memhandle->limitGB);
}

void construct_static_collision_candidates(const Eigen::MatrixXd &V,
                                           const Eigen::MatrixXi &E,
                                           const Eigen::MatrixXi &F,
                                           vector<pair<int, int>> &overlaps,
                                           vector<stq::gpu::Aabb> &boxes,
                                           double inflation_radius) {
  construct_continuous_collision_candidates(V, V, E, F, overlaps, boxes,
                                            inflation_radius);
}

void construct_continuous_collision_candidates(const Eigen::MatrixXd &V0,
                                               const Eigen::MatrixXd &V1,
                                               const Eigen::MatrixXi &E,
                                               const Eigen::MatrixXi &F,
                                               vector<pair<int, int>> &overlaps,
                                               vector<stq::gpu::Aabb> &boxes,
                                               double inflation_radius) {
  constructBoxes(V0, V1, E, F, boxes, -1, inflation_radius);
  int N = boxes.size();
  int nbox = 0;
  int devcount = 1;

  int2 *d_overlaps;
  int *d_count;
  int bpthreads = 32;
  int start_id = 0;
  int limitGB = 0;
  stq::gpu::MemHandler *memhandle = new stq::gpu::MemHandler();
  while (N > start_id) {
    run_sweep_sharedqueue(boxes.data(), memhandle, N, nbox, overlaps,
                          d_overlaps, d_count, bpthreads, start_id, devcount,
                          limitGB);
    gpuErrchk(cudaDeviceSynchronize());
  }

  spdlog::trace("Overlaps size {:d}", overlaps.size());
  cudaFree(d_overlaps);
  cudaFree(d_count);
}

Scalar compute_toi_strategy(const Eigen::MatrixXd &V0,
                            const Eigen::MatrixXd &V1, const Eigen::MatrixXi &E,
                            const Eigen::MatrixXi &F, int max_iter,
                            Scalar min_distance, Scalar tolerance) {
  vector<stq::gpu::Aabb> boxes;
  constructBoxes(V0, V1, E, F, boxes);
  spdlog::trace("Finished constructing");
  int N = boxes.size();
  int nbox = 0;
  int devcount = 1;

  stq::gpu::MemHandler *memhandle = new stq::gpu::MemHandler();

  vector<pair<int, int>> overlaps;
  vector<int> result_list;

  // BROADPHASE
  int2 *d_overlaps;
  int *d_count;
  int bpthreads = 32;
  int npthreads = 1024;
  int start_id = 0;
  int limitGB = 0;

  json j;
  Record r(j);

  Scalar earliest_toi;

  while (N > start_id) {

    run_sweep_sharedqueue(boxes.data(), memhandle, N, nbox, overlaps,
                          d_overlaps, d_count, bpthreads, start_id, devcount,
                          limitGB);

    // copy overlap count
    int count;
    gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    spdlog::trace("Count {:d}", count);

    // Allocate boxes to GPU
    Aabb *d_boxes = copy_to_gpu(boxes.data(), boxes.size());

    spdlog::trace("Copying vertices");
    double *d_vertices_t0 = copy_to_gpu(V0.data(), V0.size());
    double *d_vertices_t1 = copy_to_gpu(V1.data(), V1.size());

    int Vrows = V0.rows();
    assert(Vrows == V1.rows());

    run_narrowphase(d_overlaps, d_boxes, memhandle, count, d_vertices_t0,
                    d_vertices_t1, Vrows, npthreads, /*max_iter=*/max_iter,
                    /*tol=*/tolerance,
                    /*ms=*/min_distance, /*allow_zero_toi=*/true, result_list,
                    earliest_toi, r);

    if (earliest_toi < 1e-6) {
      run_narrowphase(
        d_overlaps, d_boxes, memhandle, count, d_vertices_t0, d_vertices_t1,
        Vrows, npthreads, /*max_iter=*/-1, /*tol=*/tolerance,
        /*ms=*/0.0, /*allow_zero_toi=*/false, result_list, earliest_toi, r);
      earliest_toi *= 0.8;
    }

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(d_count));
    gpuErrchk(cudaFree(d_overlaps));
    gpuErrchk(cudaFree(d_boxes));
    gpuErrchk(cudaFree(d_vertices_t0));
    gpuErrchk(cudaFree(d_vertices_t1));

    gpuErrchk(cudaDeviceSynchronize());
  }

  return earliest_toi;
}

} // namespace ccd::gpu