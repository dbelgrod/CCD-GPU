#pragma once
#include <gpubf/aabb.cuh>

using namespace ccdgpu;


#define Vec3Conv(v) {v[0], v[1], v[2]}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


typedef float Scalar;

__global__ void addData
(
    int2 * overlaps, 
    Aabb * boxes,
    double* V0,
    double* V1, 
    int Vrows,
    int N, 
    float3* queries
);

void addData(
    const Aabb &a, 
    const Aabb &b, 
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    vector<array<array<float, 3>, 8>>& queries);

bool is_file_exist(const char *fileName);

// __global__ void run_parallel_vf_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
// );

// __global__ void run_parallel_ee_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
// );

// __global__ void run_parallel_ms_vf_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
// );

// __global__ void run_parallel_ms_ee_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
// );

// __global__ void array_to_ccd(float3 * a, int tmp_nbr, CCDdata * data );

void all_ccd_run(float3 * V, int tmp_nbr, bool is_edge,
    std::vector<bool> &result_list, double &run_time, std::vector<Scalar> &time_impact, int parallel_nbr);

void all_ccd_run(const std::vector<std::array<std::array<Scalar, 3>, 8>> &V, bool is_edge,
    std::vector<bool> &result_list, double &run_time, std::vector<Scalar> &time_impact, int parallel_nbr);

    