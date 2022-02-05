#pragma once
#include <ccdgpu/CType.cuh>
#include <ccdgpu/Type.hpp>
#include <ccdgpu/record.hpp>
#include <gpubf/aabb.cuh>

#include <spdlog/spdlog.h>

#include <vector>

#define Vec3Conv(v)                                                            \
  { v[0], v[1], v[2] }

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

__global__ void addData(const int2 *const overlaps,
                        const ccdgpu::Aabb *const boxes,
                        const ccd::Scalar *const V0,
                        const ccd::Scalar *const V1, int Vrows, int N,
                        // ccd::Scalar3 *queries);
                        ccd::CCDdata *data, int shift = 0);

bool is_file_exist(const char *fileName);

void run_ccd(const std::vector<ccdgpu::Aabb> boxes,
             const Eigen::MatrixXd &vertices_t0,
             const Eigen::MatrixXd &vertices_t1, ccdgpu::Record &r, int N,
             int &nbox, int &parallel, int &devcount,
             std::vector<std::pair<int, int>> &overlaps,
             std::vector<int> &result_list, bool &use_ms, bool &allow_zero_toi,
             ccd::Scalar &min_distance, ccd::Scalar &toi);

void run_narrowphase(int2 *d_overlaps, ccdgpu::Aabb *d_boxes, int count,
                     ccd::Scalar *d_vertices_t0, ccd::Scalar *d_vertices_t1,
                     int Vrows, int threads, int max_iter, ccd::Scalar tol,
                     ccd::Scalar ms, bool use_ms, bool allow_zero_toi,
                     std::vector<int> &result_list, ccd::Scalar &toi,
                     ccdgpu::Record &r);

void construct_static_collision_candidates(
  const Eigen::MatrixXd &V, const Eigen::MatrixXi &E, const Eigen::MatrixXi &F,
  std::vector<std::pair<int, int>> &overlaps, std::vector<ccdgpu::Aabb> &boxes,
  double inflation_radius = 0);

void construct_continuous_collision_candidates(
  const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1,
  const Eigen::MatrixXi &E, const Eigen::MatrixXi &F,
  std::vector<std::pair<int, int>> &overlaps, std::vector<ccdgpu::Aabb> &boxes,
  double inflation_radius = 0);

ccd::Scalar compute_toi_strategy(const Eigen::MatrixXd &V0,
                                 const Eigen::MatrixXd &V1,
                                 const Eigen::MatrixXi &E,
                                 const Eigen::MatrixXi &F, int max_iter,
                                 ccd::Scalar min_distance,
                                 ccd::Scalar tolerance);