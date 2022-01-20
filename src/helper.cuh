#pragma once
#include <ccdgpu/record.hpp>
#include <gpubf/aabb.cuh>
#include <gputi/CType.cuh>
#include <gputi/Type.hpp>

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

void addData(const ccdgpu::Aabb &a, const ccdgpu::Aabb &b,
             const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1,
             vector<array<array<ccd::Scalar, 3>, 8>> &queries);

bool is_file_exist(const char *fileName);

void run_ccd(vector<ccdgpu::Aabb> boxes, const Eigen::MatrixXd &vertices_t0,
             const Eigen::MatrixXd &vertices_t1, ccdgpu::Record &r, int N,
             int &nbox, int &parallel, int &devcount,
             vector<pair<int, int>> &overlaps, vector<int> &result_list,
             ccd::Scalar &toi);