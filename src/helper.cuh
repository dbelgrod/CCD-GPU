#pragma once
#include <ccdgpu/CType.cuh>
#include <ccdgpu/Type.hpp>
#include <ccdgpu/record.hpp>
#include <gpubf/aabb.cuh>

// using namespace ccdgpu;

#define Vec3Conv(v)                                                            \
  { v[0], v[1], v[2] }

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
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
                        ccd::CCDdata *data);

bool is_file_exist(const char *fileName);

void run_ccd(const vector<Aabb> boxes, const Eigen::MatrixXd &vertices_t0,
             const Eigen::MatrixXd &vertices_t1, ccdgpu::Record &r, int N,
             int &nbox, int &parallel, int &devcount,
             vector<pair<int, int>> &overlaps, vector<int> &result_list,
             bool &use_ms, bool &allow_zero_toi, ccd::Scalar &min_distance,
             ccd::Scalar &toi);

void compute_toi_strategy(const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1,
                          const Eigen::MatrixXi &E, const Eigen::MatrixXi &F,
                          int max_iter, ccd::Scalar min_distance,
                          ccd::Scalar tolerance, ccd::Scalar &earliest_toi);