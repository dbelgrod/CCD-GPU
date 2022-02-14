#pragma once
#include <ccdgpu/CType.cuh>
#include <ccdgpu/Type.hpp>
#include <ccdgpu/record.hpp>
#include <gpubf/aabb.cuh>

#include <spdlog/spdlog.h>

#include <vector>

namespace ccd::gpu {

__global__ void addData(const int2 *const overlaps,
                        const ccdgpu::Aabb *const boxes, const Scalar *const V0,
                        const Scalar *const V1, int Vrows, int N,
                        // Scalar3 *queries);
                        CCDData *data, int shift = 0);

void run_ccd(const std::vector<ccdgpu::Aabb> boxes,
             const Eigen::MatrixXd &vertices_t0,
             const Eigen::MatrixXd &vertices_t1, Record &r, int N, int &nbox,
             int &parallel, int &devcount,
             std::vector<std::pair<int, int>> &overlaps,
             std::vector<int> &result_list, bool &allow_zero_toi,
             Scalar &min_distance, Scalar &toi);

void run_narrowphase(int2 *d_overlaps, ccdgpu::Aabb *d_boxes, int count,
                     Scalar *d_vertices_t0, Scalar *d_vertices_t1, int Vrows,
                     int threads, int max_iter, Scalar tol, Scalar ms,
                     bool allow_zero_toi, std::vector<int> &result_list,
                     Scalar &toi, Record &r);

void construct_static_collision_candidates(
  const Eigen::MatrixXd &V, const Eigen::MatrixXi &E, const Eigen::MatrixXi &F,
  std::vector<std::pair<int, int>> &overlaps, std::vector<ccdgpu::Aabb> &boxes,
  double inflation_radius = 0);

void construct_continuous_collision_candidates(
  const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1,
  const Eigen::MatrixXi &E, const Eigen::MatrixXi &F,
  std::vector<std::pair<int, int>> &overlaps, std::vector<ccdgpu::Aabb> &boxes,
  double inflation_radius = 0);

Scalar compute_toi_strategy(const Eigen::MatrixXd &V0,
                            const Eigen::MatrixXd &V1, const Eigen::MatrixXi &E,
                            const Eigen::MatrixXi &F, int max_iter,
                            Scalar min_distance, Scalar tolerance);

} // namespace ccd::gpu