#pragma once
#include <array>
#include <ccdgpu/Type.hpp>
#include <ccdgpu/record.hpp>
#include <ccdgpu/timer.hpp>
#include <iostream>
#include <vector>


// #include <gputi/book.h>
namespace ccd {
// can be removed once device-only run_memory_pool_ccd copied over
__global__ void initialize_memory_pool(MP_unit *units, int query_size);
__global__ void compute_vf_tolerance_memory_pool(CCDdata *data,
                                                 CCDConfig *config,
                                                 const int query_size);
__global__ void compute_ee_tolerance_memory_pool(CCDdata *data,
                                                 CCDConfig *config,
                                                 const int query_size);
__global__ void shift_queue_pointers(CCDConfig *config);
// __global__ void vf_ccd_memory_pool(MP_unit *units, int query_size, CCDdata
// *data, CCDConfig *config, int *results);
__global__ void vf_ccd_memory_pool(MP_unit *units, int query_size,
                                   CCDdata *data, CCDConfig *config);
__global__ void ee_ccd_memory_pool(MP_unit *units, int query_size,
                                   CCDdata *data, CCDConfig *config);
__global__ void compute_ee_tolerance_memory_pool(CCDdata *data,
                                                 CCDConfig *config,
                                                 const int query_size);

void run_memory_pool_ccd(CCDdata *d_data_list, int tmp_nbr, bool is_edge,
                         std::vector<int> &result_list, int parallel_nbr,
                         int max_iter, ccd::Scalar tol, bool use_ms,
                         bool allow_zero_toi, ccd::Scalar &toi,
                         ccdgpu::Record &r);

// get the filter of ccd. the inputs are the vertices of the bounding box of
// the simulation scene this function is directly copied from
// https://github.com/Continuous-Collision-Detection/Tight-Inclusion/
std::array<Scalar, 3>
get_numerical_error(const std::vector<std::array<Scalar, 3>> &vertices,
                    const bool &check_vf, const bool using_minimum_separation);
} // namespace ccd