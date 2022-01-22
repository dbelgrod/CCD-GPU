#include <ccdgpu/helper.cuh>

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

// #include <gputi/book.h>
// #include <gputi/io.h>
#include <ccdgpu/root_finder.cuh>
#include <ccdgpu/timer.hpp>
#include <gpubf/io.cuh>

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
                        ccd::Scalar ms, ccd::CCDdata *data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N)
    return;

  data[tid].ms = ms;

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

bool is_file_exist(const char *fileName) {
  ifstream infile(fileName);
  return infile.good();
}

void run_narrowphase(int2 *d_overlaps, Aabb *d_boxes, int count,
                     ccd::Scalar *d_vertices_t0, ccd::Scalar *d_vertices_t1,
                     int Vrows, int threads, int max_iter, ccd::Scalar tol,
                     ccd::Scalar ms, bool use_ms, bool allow_zero_toi,
                     vector<int> &result_list, ccd::Scalar &toi, Record &r) {

  toi = 1.0;

  int *d_vf_count;
  int *d_ee_count;
  cudaMalloc((void **)&d_vf_count, sizeof(int));
  cudaMalloc((void **)&d_ee_count, sizeof(int));

  int2 *d_vf_overlaps;
  int2 *d_ee_overlaps;

  int start_id = 0;
  int size = count;

  // double tavg = 0;
  // double tmp_tall = 0;

  while (1) {

    int remain = size - start_id;
    if (remain <= 0 || toi == 0)
      break;
    printf("remain %i, start_id %i\n", remain, start_id);

    int tmp_nbr = std::min(remain, MAX_OVERLAP_SIZE);

    r.Start("splitOverlaps", /*gpu=*/true);
    cudaMemset(d_vf_count, 0, sizeof(int));
    cudaMemset(d_ee_count, 0, sizeof(int));
    gpuErrchk(cudaGetLastError());

    cudaMalloc((void **)&d_vf_overlaps, sizeof(int2) * tmp_nbr);
    cudaMalloc((void **)&d_ee_overlaps, sizeof(int2) * tmp_nbr);
    gpuErrchk(cudaGetLastError());

    split_overlaps<<<tmp_nbr / threads + 1, threads>>>(
        d_overlaps + start_id, d_boxes, tmp_nbr, d_vf_overlaps, d_ee_overlaps,
        d_vf_count, d_ee_count);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    r.Stop();

    r.Start("createDataList", /*gpu=*/true);
    int vf_size;
    int ee_size;
    cudaMemcpy(&vf_size, d_vf_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ee_size, d_ee_count, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "vf_size " << vf_size << " ee_size " << ee_size << endl;
    gpuErrchk(cudaGetLastError());

    CCDdata *d_ee_data_list;
    CCDdata *d_vf_data_list;

    size_t ee_data_size = sizeof(CCDdata) * ee_size;
    size_t vf_data_size = sizeof(CCDdata) * vf_size;

    cudaMalloc((void **)&d_ee_data_list, ee_data_size);
    cudaMalloc((void **)&d_vf_data_list, vf_data_size);
    printf("ee_data_size %llu\n", ee_data_size);
    printf("vf_data_size %llu\n", vf_data_size);
    gpuErrchk(cudaGetLastError());

    addData<<<ee_size / threads + 1, threads>>>(
        d_ee_overlaps, d_boxes, d_vertices_t0, d_vertices_t1, Vrows, ee_size,
        ms, d_ee_data_list);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    addData<<<vf_size / threads + 1, threads>>>(
        d_vf_overlaps, d_boxes, d_vertices_t0, d_vertices_t1, Vrows, vf_size,
        ms, d_vf_data_list);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    r.Stop();

    printf("vf_size %i, ee_size %i\n", vf_size, ee_size);

    // int size = count;
    // cout << "data loaded, size " << queries.size() <<
    // endl;
    cout << "data loaded, size " << size << endl;

    // result_list.resize(size);

    int parallel = 64;
    printf("run_memory_pool_ccd using %i threads\n", parallel);
    r.Start("run_memory_pool_ccd (narrowphase)",
            /*gpu=*/true);
    // toi = 1;
    run_memory_pool_ccd(d_vf_data_list, vf_size, /*is_edge_edge=*/false,
                        result_list, parallel, max_iter, tol, use_ms,
                        allow_zero_toi, toi);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    printf("toi after vf %e\n", toi);
    // printf("time after vf %.6f\n", tmp_tall);

    run_memory_pool_ccd(d_ee_data_list, ee_size, /*is_edge_edge=*/true,
                        result_list, parallel, max_iter, tol, use_ms,
                        allow_zero_toi, toi);
    gpuErrchk(cudaGetLastError());
    printf("toi after ee %e\n", toi);
    // printf("time after ee %.6f\n", tmp_tall);
    r.Stop();

    gpuErrchk(cudaFree(d_vf_overlaps));
    gpuErrchk(cudaFree(d_ee_overlaps));

    start_id += tmp_nbr;
  }
  gpuErrchk(cudaFree(d_vf_count));
  gpuErrchk(cudaFree(d_ee_count));
}

void run_ccd(const vector<Aabb> boxes, const Eigen::MatrixXd &vertices_t0,
             const Eigen::MatrixXd &vertices_t1, ccdgpu::Record &r, int N,
             int &nbox, int &parallel, int &devcount,
             vector<pair<int, int>> &overlaps, vector<int> &result_list,
             bool &use_ms, bool &allow_zero_toi, ccd::Scalar &ms,
             ccd::Scalar &toi) {
  int2 *d_overlaps;
  int *d_count;
  int threads = 32; // HARDCODING THREADS FOR NOW
  r.Start("run_sweep_sharedqueue (broadphase)", /*gpu=*/true);
  run_sweep_sharedqueue(boxes.data(), N, nbox, overlaps, d_overlaps, d_count,
                        threads, devcount);
  r.Stop();
  threads = 1024;
  gpuErrchk(cudaGetLastError());
  printf("Threads now %i\n", threads);

  r.Start("copyBoxesToGpu", /*gpu=*/true);
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
  r.Stop();

  r.Start("copyVerticesToGpu", /*gpu=*/true);
  printf("Copying vertices\n");
  ccd::Scalar *d_vertices_t0;
  ccd::Scalar *d_vertices_t1;
  cudaMalloc((void **)&d_vertices_t0, sizeof(ccd::Scalar) * vertices_t0.size());
  cudaMalloc((void **)&d_vertices_t1, sizeof(ccd::Scalar) * vertices_t1.size());
  cudaMemcpy(d_vertices_t0, vertices_t0.data(),
             sizeof(ccd::Scalar) * vertices_t0.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertices_t1, vertices_t1.data(),
             sizeof(ccd::Scalar) * vertices_t1.size(), cudaMemcpyHostToDevice);
  r.Stop();
  int Vrows = vertices_t0.rows();
  assert(Vrows == vertices_t1.rows());

  int max_iter = 1e6;
  ccd::Scalar tolerance = 1e-6;

  run_narrowphase(d_overlaps, d_boxes, count, d_vertices_t0, d_vertices_t1,
                  Vrows, threads, max_iter, tolerance, ms, use_ms,
                  allow_zero_toi, result_list, toi, r);

  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaFree(d_overlaps));
  gpuErrchk(cudaFree(d_boxes));
  gpuErrchk(cudaFree(d_vertices_t0));
  gpuErrchk(cudaFree(d_vertices_t1));

  gpuErrchk(cudaGetLastError());

  cudaDeviceSynchronize();
}

void compute_toi_strategy(const Eigen::MatrixXd &V0, const Eigen::MatrixXd &V1,
                          const Eigen::MatrixXi &E, const Eigen::MatrixXi &F,
                          ccd::Scalar min_distance, int max_iter, int tolerance,
                          ccd::Scalar &earliest_toi) {

  vector<ccdgpu::Aabb> boxes;
  constructBoxes(V0, V1, E, F, boxes);
  int N = boxes.size();
  int nbox = 0;
  int devcount = 1;

  vector<pair<int, int>> overlaps;
  vector<int> result_list;

  // BROADPHASE
  int2 *d_overlaps;
  int *d_count;
  int threads = 32; // HARDCODING THREADS FOR NOW
  run_sweep_sharedqueue(boxes.data(), N, nbox, overlaps, d_overlaps, d_count,
                        threads, devcount);
  threads = 1024;
  gpuErrchk(cudaGetLastError());
  printf("Threads now %i\n", threads);

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

  printf("Copying vertices\n");
  ccd::Scalar *d_vertices_t0;
  ccd::Scalar *d_vertices_t1;
  cudaMalloc((void **)&d_vertices_t0, sizeof(ccd::Scalar) * V0.size());
  cudaMalloc((void **)&d_vertices_t1, sizeof(ccd::Scalar) * V1.size());
  cudaMemcpy(d_vertices_t0, V0.data(), sizeof(ccd::Scalar) * V0.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertices_t1, V1.data(), sizeof(ccd::Scalar) * V1.size(),
             cudaMemcpyHostToDevice);

  int Vrows = V0.rows();
  assert(Vrows == V1.rows());

  Record r;

  run_narrowphase(d_overlaps, d_boxes, count, d_vertices_t0, d_vertices_t1,
                  Vrows, threads, max_iter, /*tol=*/tolerance,
                  /*ms=*/min_distance,
                  /*use_ms=*/false,
                  /*allow_zero_toi=*/true, result_list, earliest_toi, r);

  if (earliest_toi < 1e-6) {
    run_narrowphase(d_overlaps, d_boxes, count, d_vertices_t0, d_vertices_t1,
                    Vrows, threads, max_iter, /*tol=*/tolerance,
                    /*ms=*/min_distance,
                    /*use_ms=*/false,
                    /*allow_zero_toi=*/true, result_list, earliest_toi, r);
  }
}