#include <ccdgpu/helper.cuh>

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <iostream>

// #include <gputi/timer.cuh>
#include <gputi/book.h>
#include <gputi/io.h>
// #include <gputi/read_rational_csv.cuh>
#include <gputi/root_finder.cuh>
#include <gputi/timer.hpp>

#include <ccdgpu/record.hpp>
#include <gpubf/simulation.cuh>

using namespace std;
using namespace ccd;
using namespace ccdgpu;

__global__ void split_overlaps(const int2 *const overlaps,
                               const ccdgpu::Aabb *const boxes, int N,
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
  }
}

__global__ void addData(const int2 *const overlaps,
                        const ccdgpu::Aabb *const boxes,
                        const ccd::Scalar *const V0,
                        const ccd::Scalar *const V1, int Vrows, int N,
                        CCDdata *data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N)
    return;

#ifndef NO_CHECK_MS
  data[tid].ms = MINIMUM_SEPARATION_BENCHMARK;
#endif

  // printf("vf_count %i, ee_count %i", *vf_count, *ee_count);

  int minner = min(overlaps[tid].x, overlaps[tid].y);
  int maxxer = max(overlaps[tid].x, overlaps[tid].y);
  int3 avids = boxes[minner].vertexIds;
  int3 bvids = boxes[maxxer].vertexIds;

  // data[tid].v0s[0] = a[8 * tid + 0].x;
  // data[tid].v1s[0] = a[8 * tid + 1].x;
  // data[tid].v2s[0] = a[8 * tid + 2].x;
  // data[tid].v3s[0] = a[8 * tid + 3].x;
  // data[tid].v0e[0] = a[8 * tid + 4].x;
  // data[tid].v1e[0] = a[8 * tid + 5].x;
  // data[tid].v2e[0] = a[8 * tid + 6].x;
  // data[tid].v3e[0] = a[8 * tid + 7].x;

  // data[tid].v0s[1] = a[8 * tid + 0].y;
  // data[tid].v1s[1] = a[8 * tid + 1].y;
  // data[tid].v2s[1] = a[8 * tid + 2].y;
  // data[tid].v3s[1] = a[8 * tid + 3].y;
  // data[tid].v0e[1] = a[8 * tid + 4].y;
  // data[tid].v1e[1] = a[8 * tid + 5].y;
  // data[tid].v2e[1] = a[8 * tid + 6].y;
  // data[tid].v3e[1] = a[8 * tid + 7].y;

  // data[tid].v0s[2] = a[8 * tid + 0].z;
  // data[tid].v1s[2] = a[8 * tid + 1].z;
  // data[tid].v2s[2] = a[8 * tid + 2].z;
  // data[tid].v3s[2] = a[8 * tid + 3].z;
  // data[tid].v0e[2] = a[8 * tid + 4].z;
  // data[tid].v1e[2] = a[8 * tid + 5].z;
  // data[tid].v2e[2] = a[8 * tid + 6].z;
  // data[tid].v3e[2] = a[8 * tid + 7].z;

  if (is_vertex(avids) && is_face(bvids)) {
    // int i = atomicAdd(vf_count, 1);
    // queries[8 * tid + 0] = ccd::make_Scalar3(
    //     V0[avids.x + 0], V0[avids.x + Vrows], V0[avids.x + 2 * Vrows]);
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

    // queries[8 * tid + 1] = ccd::make_Scalar3(
    //     V0[bvids.x + 0], V0[bvids.x + Vrows], V0[bvids.x + 2 * Vrows]);
    // queries[8 * tid + 2] = ccd::make_Scalar3(
    //     V0[bvids.y + 0], V0[bvids.y + Vrows], V0[bvids.y + 2 * Vrows]);

    // queries[8 * tid + 3] = ccd::make_Scalar3(
    //     V0[bvids.z + 0], V0[bvids.z + Vrows], V0[bvids.z + 2 * Vrows]);
    // queries[8 * tid + 4] = ccd::make_Scalar3(
    //     V1[avids.x + 0], V1[avids.x + Vrows], V1[avids.x + 2 * Vrows]);
    // ;
    // queries[8 * tid + 5] = ccd::make_Scalar3(
    //     V1[bvids.x + 0], V1[bvids.x + Vrows], V1[bvids.x + 2 * Vrows]);
    // ;
    // queries[8 * tid + 6] = ccd::make_Scalar3(
    //     V1[bvids.y + 0], V1[bvids.y + Vrows], V1[bvids.y + 2 * Vrows]);
    // ;
    // queries[8 * tid + 7] = ccd::make_Scalar3(
    //     V1[bvids.z + 0], V1[bvids.z + Vrows], V1[bvids.z + 2 * Vrows]);
    // ;
  } else if (is_edge(avids) && is_edge(bvids)) {
    // int j = atomicAdd(ee_count, 1);

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

    // queries[8 * tid + 0] = ccd::make_Scalar3(
    //     V0[avids.x + 0], V0[avids.x + Vrows], V0[avids.x + 2 * Vrows]);
    // ;

    // queries[8 * tid + 1] = ccd::make_Scalar3(
    //     V0[avids.y + 0], V0[avids.y + Vrows], V0[avids.y + 2 * Vrows]);
    // ;

    // queries[8 * tid + 2] = ccd::make_Scalar3(
    //     V0[bvids.x + 0], V0[bvids.x + Vrows], V0[bvids.x + 2 * Vrows]);
    // ;

    // queries[8 * tid + 3] = ccd::make_Scalar3(
    //     V0[bvids.y + 0], V0[bvids.y + Vrows], V0[bvids.y + 2 * Vrows]);
    // ;

    // queries[8 * tid + 4] = ccd::make_Scalar3(
    //     V1[avids.x + 0], V1[avids.x + Vrows], V1[avids.x + 2 * Vrows]);
    // ;

    // queries[8 * tid + 5] = ccd::make_Scalar3(
    //     V1[avids.y + 0], V1[avids.y + Vrows], V1[avids.y + 2 * Vrows]);
    // ;

    //   queries[8 * tid + 6] = ccd::make_Scalar3(
    //       V1[bvids.x + 0], V1[bvids.x + Vrows], V1[bvids.x + 2 * Vrows]);
    //   ;

    //   queries[8 * tid + 7] = ccd::make_Scalar3(
    //       V1[bvids.y + 0], V1[bvids.y + Vrows], V1[bvids.y + 2 * Vrows]);
    //   ;
  } else
    assert(0);
}

void addData(const ccdgpu::Aabb &a, const ccdgpu::Aabb &b,
             const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1,
             vector<array<array<ccd::Scalar, 3>, 8>> &queries) {
  // auto is_vertex = [&](Aabb x){return x.vertexIds.y < 0 ;};
  // auto is_edge = [&](Aabb x){return !is_vertex(x) && x.vertexIds.z < 0
  // ;}; auto is_face = [&](Aabb x){return !is_vertex(x) && !is_edge(x);};

  // auto is_face = [&](Aabb x){return x.vertexIds.z >= 0;};
  // auto is_edge = [&](Aabb x){return x.vertexIds.z < 0 && x.vertexIds.y >=
  // 0
  // ;}; auto is_vertex = [&](Aabb x){return x.vertexIds.z < 0  &&
  // x.vertexIds.y < 0;};

  if (is_vertex(a) && is_face(b)) {
    auto avids = a.vertexIds;
    auto bvids = b.vertexIds;
    // Point at t=0s
    auto vertex_start = V0.cast<ccd::Scalar>().row(avids.x);
    // // Triangle at t = 0
    auto face_vertex0_start = V0.cast<ccd::Scalar>().row(bvids.x);
    auto face_vertex1_start = V0.cast<ccd::Scalar>().row(bvids.y);
    auto face_vertex2_start = V0.cast<ccd::Scalar>().row(bvids.z);
    // // Point at t=1
    auto vertex_end = V1.cast<ccd::Scalar>().row(avids.x);
    // // Triangle at t = 1
    auto face_vertex0_end = V1.cast<ccd::Scalar>().row(bvids.x);
    auto face_vertex1_end = V1.cast<ccd::Scalar>().row(bvids.y);
    auto face_vertex2_end = V1.cast<ccd::Scalar>().row(bvids.z);

    array<array<ccd::Scalar, 3>, 8> tmp;
    tmp[0] = Vec3Conv(vertex_start);
    tmp[1] = Vec3Conv(face_vertex0_start);
    tmp[2] = Vec3Conv(face_vertex1_start);
    tmp[3] = Vec3Conv(face_vertex2_start);
    tmp[4] = Vec3Conv(vertex_end);
    tmp[5] = Vec3Conv(face_vertex0_end);
    tmp[6] = Vec3Conv(face_vertex1_end);
    tmp[7] = Vec3Conv(face_vertex2_end);
    queries.emplace_back(tmp);
  } else if (is_face(a) && is_vertex(b))
    return addData(b, a, V0, V1, queries);
  else if (is_edge(a) && is_edge(b)) {
    auto avids = a.vertexIds;
    auto bvids = b.vertexIds;
    //     // Edge 1 at t=0
    auto edge0_vertex0_start = V0.cast<ccd::Scalar>().row(avids.x);
    auto edge0_vertex1_start = V0.cast<ccd::Scalar>().row(avids.y);
    // // Edge 2 at t=0
    auto edge1_vertex0_start = V0.cast<ccd::Scalar>().row(bvids.x);
    auto edge1_vertex1_start = V0.cast<ccd::Scalar>().row(bvids.y);
    // // Edge 1 at t=1
    auto edge0_vertex0_end = V1.cast<ccd::Scalar>().row(avids.x);
    auto edge0_vertex1_end = V1.cast<ccd::Scalar>().row(avids.y);
    // // Edge 2 at t=1
    auto edge1_vertex0_end = V1.cast<ccd::Scalar>().row(bvids.x);
    auto edge1_vertex1_end = V1.cast<ccd::Scalar>().row(bvids.y);

    // queries.emplace_back(Vec3Conv(edge0_vertex0_start));
    array<array<ccd::Scalar, 3>, 8> tmp;
    tmp[0] = Vec3Conv(edge0_vertex0_start);
    tmp[1] = Vec3Conv(edge0_vertex1_start);
    tmp[2] = Vec3Conv(edge1_vertex0_start);
    tmp[3] = Vec3Conv(edge1_vertex1_start);
    tmp[4] = Vec3Conv(edge0_vertex0_end);
    tmp[5] = Vec3Conv(edge0_vertex1_end);
    tmp[6] = Vec3Conv(edge1_vertex0_end);
    tmp[7] = Vec3Conv(edge1_vertex1_end);
    queries.emplace_back(tmp);
  } else
    abort();
}

bool is_file_exist(const char *fileName) {
  ifstream infile(fileName);
  return infile.good();
}

// __global__ void array_to_ccd(const ccd::Scalar3 *const a, int tmp_nbr,
//                              CCDdata *data) {
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tid >= tmp_nbr)
//     return;

// #ifndef NO_CHECK_MS
//   data[tid].ms = MINIMUM_SEPARATION_BENCHMARK;
// #endif

//   data[tid].v0s[0] = a[8 * tid + 0].x;
//   data[tid].v1s[0] = a[8 * tid + 1].x;
//   data[tid].v2s[0] = a[8 * tid + 2].x;
//   data[tid].v3s[0] = a[8 * tid + 3].x;
//   data[tid].v0e[0] = a[8 * tid + 4].x;
//   data[tid].v1e[0] = a[8 * tid + 5].x;
//   data[tid].v2e[0] = a[8 * tid + 6].x;
//   data[tid].v3e[0] = a[8 * tid + 7].x;

//   data[tid].v0s[1] = a[8 * tid + 0].y;
//   data[tid].v1s[1] = a[8 * tid + 1].y;
//   data[tid].v2s[1] = a[8 * tid + 2].y;
//   data[tid].v3s[1] = a[8 * tid + 3].y;
//   data[tid].v0e[1] = a[8 * tid + 4].y;
//   data[tid].v1e[1] = a[8 * tid + 5].y;
//   data[tid].v2e[1] = a[8 * tid + 6].y;
//   data[tid].v3e[1] = a[8 * tid + 7].y;

//   data[tid].v0s[2] = a[8 * tid + 0].z;
//   data[tid].v1s[2] = a[8 * tid + 1].z;
//   data[tid].v2s[2] = a[8 * tid + 2].z;
//   data[tid].v3s[2] = a[8 * tid + 3].z;
//   data[tid].v0e[2] = a[8 * tid + 4].z;
//   data[tid].v1e[2] = a[8 * tid + 5].z;
//   data[tid].v2e[2] = a[8 * tid + 6].z;
//   data[tid].v3e[2] = a[8 * tid + 7].z;
// }

void run_memory_pool_ccd(CCDdata *d_data_list, int tmp_nbr, bool is_edge,
                         std::vector<int> &result_list, int parallel_nbr,
                         double &run_time, ccd::Scalar &toi) {
  int nbr = tmp_nbr;
  cout << "tmp_nbr " << tmp_nbr << endl;
  // int *res = new int[nbr];
  CCDConfig *config = new CCDConfig[1];
  // config[0].err_in[0] =
  //     -1; // the input error bound calculate from the AABB of the whole
  //     mesh
  config[0].co_domain_tolerance = 1e-6; // tolerance of the co-domain
  // config[0].max_t = 1;                  // the upper bound of the time

  // interval
  config[0].toi = toi;
  config[0].mp_end = nbr;
  config[0].mp_start = 0;
  config[0].mp_remaining = nbr;
  config[0].overflow_flag = 0;
  config[0].unit_size = nbr * 4; // 2.0 * nbr;
  printf("unit_size : %llu\n", config[0].unit_size);

  // device
  // CCDdata *d_data_list;
  // int *d_res;
  MP_unit *d_units;
  CCDConfig *d_config;

  size_t unit_size = sizeof(MP_unit) * config[0].unit_size; // arbitrary #

  // size_t result_size = sizeof(int) * nbr;

  // int dbg_size=sizeof(ccd::Scalar)*8;

  // cudaMalloc(&d_data_list, data_size);
  // cudaMalloc(&d_res, result_size);
  cudaMalloc(&d_units, unit_size);
  cudaMalloc(&d_config, sizeof(CCDConfig));

  // cudaMemcpy(d_data_list, data_list, data_size,
  // cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_config, config, sizeof(CCDConfig), cudaMemcpyHostToDevice);
  gpuErrchk(cudaGetLastError());

  ccd::Timer timer;
  // cudaProfilerStart();
  timer.start();
  printf("nbr: %i, parallel_nbr %i\n", nbr, parallel_nbr);
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

  printf("EACH_LAUNCH_SIZE: %llu\n", EACH_LAUNCH_SIZE);
  printf("sizeof(Scalar) %i\n", sizeof(ccd::Scalar));
  // cudaMemcpy(&toi, &d_config[0].toi, sizeof(ccd::Scalar),
  //            cudaMemcpyDeviceToHost);
  // printf("toi init %.6f\n", toi);

  int nbr_per_loop = nbr;
  int start;
  int end;
  //   int inc = 0;
  // int unit_size = config[0].unit_size;
  printf("Queue size t0: %i\n", nbr_per_loop);
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
    // std::cout << "toi " << toi << std::endl;
    // printf("toi %.4f\n", toi);
    // printf("Start %i, End %i, Queue size: %i\n", start, end,
    // nbr_per_loop);
    gpuErrchk(cudaGetLastError());
    printf("Queue size: %i\n", nbr_per_loop);
  }
  cudaDeviceSynchronize();
  double tt = timer.getElapsedTimeInMicroSec();
  run_time += tt / 1000.0f;
  // cudaProfilerStop();
  gpuErrchk(cudaGetLastError());

  // cudaMemcpy(res, d_res, result_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&toi, &d_config[0].toi, sizeof(ccd::Scalar),
             cudaMemcpyDeviceToHost);
  int overflow;
  cudaMemcpy(&overflow, &d_config[0].overflow_flag, sizeof(int),
             cudaMemcpyDeviceToHost);
  if (overflow)
    printf("OVERFLOW!!!!\n");
  // cudaMemcpy(dbg, d_dbg, dbg_size, cudaMemcpyDeviceToHost);

  gpuErrchk(cudaFree(d_data_list));
  // if (is_edge) {
  // cudaFree(d_res);
  gpuErrchk(cudaFree(d_units));
  gpuErrchk(cudaFree(d_config));
  // }

  // for (size_t i = 0; i < nbr; i++) {
  //   result_list[i] = res[i];
  // }

  // delete[] res;
  // delete[] data_list;
  // delete[] units;
  delete[] config;
  // delete[] dbg;
  cudaError_t ct = cudaGetLastError();
  printf("******************\n%s\n************\n", cudaGetErrorString(ct));

  return;
}

void run_ccd(const vector<Aabb> boxes, const Eigen::MatrixXd &vertices_t0,
             const Eigen::MatrixXd &vertices_t1, Record &r, int N, int &nbox,
             int &parallel, int &devcount, vector<pair<int, int>> &overlaps,
             vector<int> &result_list, ccd::Scalar &toi) {
  int2 *d_overlaps;
  int *d_count;
  int threads = 32; // HARDCODING THREADS FOR NOW
  r.Start("run_sweep_sharedqueue (broadphase)", /*gpu=*/true);
  run_sweep_sharedqueue(boxes.data(), N, nbox, overlaps, d_overlaps, d_count,
                        threads, devcount);
  threads = 1024;
  // gpuErrchk(cudaDeviceSynchronize());
  r.Stop();
  gpuErrchk(cudaGetLastError());
  printf("Threads now %i\n", threads);

  r.Start("broadphase -> narrowphase", /*gpu=*/true);
  // copy overlap count
  int count;
  gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  printf("Count %i\n", count);
  gpuErrchk(cudaGetLastError());

  // Allocate boxes to GPU
  Aabb *d_boxes;
  cudaMalloc((void **)&d_boxes, sizeof(Aabb) * N);
  cudaMemcpy(d_boxes, boxes.data(), sizeof(Aabb) * N, cudaMemcpyHostToDevice);
  gpuErrchk(cudaGetLastError());

  // ccd::Scalar3 *d_queries;
  // size_t queries_size = sizeof(ccd::Scalar3) * 8 * count;
  // cout << "queries size: " << queries_size << endl;
  // cudaMalloc((void **)&d_queries, queries_size);
  // gpuErrchk(cudaGetLastError());

  printf("Copying vertices\n");
  ccd::Scalar *d_vertices_t0;
  ccd::Scalar *d_vertices_t1;
  cudaMalloc((void **)&d_vertices_t0, sizeof(ccd::Scalar) * vertices_t0.size());
  cudaMalloc((void **)&d_vertices_t1, sizeof(ccd::Scalar) * vertices_t1.size());
  cudaMemcpy(d_vertices_t0, vertices_t0.data(),
             sizeof(ccd::Scalar) * vertices_t0.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertices_t1, vertices_t1.data(),
             sizeof(ccd::Scalar) * vertices_t1.size(), cudaMemcpyHostToDevice);

  int Vrows = vertices_t0.rows();
  assert(Vrows == vertices_t1.rows());

  int *d_vf_count;
  int *d_ee_count;
  cudaMalloc((void **)&d_vf_count, sizeof(int));
  cudaMalloc((void **)&d_ee_count, sizeof(int));
  cudaMemset(d_vf_count, 0, sizeof(int));
  cudaMemset(d_ee_count, 0, sizeof(int));

  gpuErrchk(cudaGetLastError());

  int2 *d_vf_overlaps;
  int2 *d_ee_overlaps;
  cudaMalloc((void **)&d_vf_overlaps, sizeof(int2) * count);
  cudaMalloc((void **)&d_ee_overlaps, sizeof(int2) * count);

  split_overlaps<<<count / threads + 1, threads>>>(d_overlaps, d_boxes, count,
                                                   d_vf_overlaps, d_ee_overlaps,
                                                   d_vf_count, d_ee_count);
  cudaDeviceSynchronize();

  int vf_size;
  int ee_size;
  cudaMemcpy(&vf_size, d_vf_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ee_size, d_ee_count, sizeof(int), cudaMemcpyDeviceToHost);
  cout << "vf_size " << vf_size << " ee_size " << ee_size << endl;

  // ccd::Scalar3 *d_ee_queries;
  // ccd::Scalar3 *d_vf_queries;
  // // size_t queries_size = sizeof(ccd::Scalar3) * 8 * count;
  // // cout << "queries size: " << queries_size << endl;
  // cudaMalloc((void **)&d_ee_queries, sizeof(ccd::Scalar3) * 8 * ee_size);
  // cudaMalloc((void **)&d_vf_queries, sizeof(ccd::Scalar3) * 8 * vf_size);
  // gpuErrchk(cudaGetLastError());

  CCDdata *d_ee_data_list;
  CCDdata *d_vf_data_list;
  size_t ee_data_size = sizeof(CCDdata) * ee_size;
  size_t vf_data_size = sizeof(CCDdata) * vf_size;
  // printf("data_size %llu\n", data_size);
  cudaMalloc((void **)&d_ee_data_list, ee_data_size);
  cudaMalloc((void **)&d_vf_data_list, vf_data_size);
  printf("ee_data_size %llu\n", ee_data_size);
  printf("vf_data_size %llu\n", vf_data_size);
  gpuErrchk(cudaGetLastError());

  addData<<<ee_size / threads + 1, threads>>>(d_ee_overlaps, d_boxes,
                                              d_vertices_t0, d_vertices_t1,
                                              Vrows, ee_size, d_ee_data_list);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  addData<<<vf_size / threads + 1, threads>>>(d_vf_overlaps, d_boxes,
                                              d_vertices_t0, d_vertices_t1,
                                              Vrows, vf_size, d_vf_data_list);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  r.Stop();

  gpuErrchk(cudaFree(d_overlaps));
  gpuErrchk(cudaFree(d_boxes));
  gpuErrchk(cudaFree(d_vertices_t0));
  gpuErrchk(cudaFree(d_vertices_t1));
  gpuErrchk(cudaFree(d_vf_overlaps));
  gpuErrchk(cudaFree(d_ee_overlaps));
  gpuErrchk(cudaFree(d_vf_count));
  gpuErrchk(cudaFree(d_ee_count));
  gpuErrchk(cudaGetLastError());

  cudaDeviceSynchronize();

  printf("vf_size %i, ee_size %i\n", vf_size, ee_size);

  int size = count;
  // cout << "data loaded, size " << queries.size() << endl;
  cout << "data loaded, size " << size << endl;
  double tavg = 0;
  size_t max_query_cp_size = EACH_LAUNCH_SIZE;
  int start_id = 0;

  // result_list.resize(size);
  double tmp_tall = 0;
  // bool is_edge_edge = true;

  printf("run_memory_pool_ccd using %i threads\n", parallel);
  r.Start("run_memory_pool_ccd (narrowphase)", /*gpu=*/true);
  toi = 1;
  run_memory_pool_ccd(d_vf_data_list, vf_size, /*is_edge_edge=*/false,
                      result_list, parallel, tmp_tall, toi);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  printf("toi after vf %e\n", toi);
  printf("time after vf %.6f\n", tmp_tall);

  run_memory_pool_ccd(d_ee_data_list, ee_size, /*is_edge_edge=*/true,
                      result_list, parallel, tmp_tall, toi);
  gpuErrchk(cudaGetLastError());
  printf("toi after ee %e\n", toi);
  printf("time after ee %.6f\n", tmp_tall);
  r.Stop();

  tavg += tmp_tall;
  cout << "tot time " << tavg << endl;
  tavg /= size;
  cout << "avg time " << tavg << endl;

  cout << "toi " << toi << endl;
  // cudaDeviceReset();
}