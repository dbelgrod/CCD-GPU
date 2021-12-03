#include <ccdgpu/helper.cuh>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <iostream>

// #include <gputi/timer.cuh>
#include <gputi/timer.hpp>
#include <gputi/root_finder.h>
#include <gputi/book.h>
#include <gputi/io.h>

using namespace std;
using namespace ccd;

__global__ void addData
(
    int2 * overlaps, 
    Aabb * boxes,
    double* V0,
    double* V1, 
    int Vrows,
    int N, 
    float3* queries
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    int minner  = min(overlaps[tid].x, overlaps[tid].y);
    int maxxer = max(overlaps[tid].x, overlaps[tid].y);
    int3 avids = boxes[minner].vertexIds;
    int3 bvids = boxes[maxxer].vertexIds;

    // if (is_face(avids) && is_vertex(bvids))
    // {
    //     int3 tmp = avids;
    //     avids = bvids;
    //     bvids = tmp;
    // }

    if (is_vertex(avids) && is_face(bvids))
    {
        // auto vertex_start = V0.cast<float>().row(avids.x);
        float3 vertex_start = make_float3(V0[avids.x + 0], V0[avids.x + Vrows], V0[avids.x + 2*Vrows]);
        // // Triangle at t = 0
        auto face_vertex0_start = make_float3(V0[bvids.x + 0], V0[bvids.x + Vrows], V0[bvids.x + 2*Vrows]) ;
        auto face_vertex1_start = make_float3(V0[bvids.y + 0], V0[bvids.y + Vrows], V0[bvids.y + 2*Vrows]) ;
        auto face_vertex2_start = make_float3(V0[bvids.z + 0], V0[bvids.z + Vrows], V0[bvids.z + 2*Vrows]) ;
        // // Point at t=1
        auto vertex_end = make_float3(V1[avids.x + 0], V1[avids.x + Vrows], V1[avids.x + 2*Vrows]) ;
        // // Triangle at t = 1
        auto face_vertex0_end = make_float3(V1[bvids.x + 0], V1[bvids.x + Vrows], V1[bvids.x + 2*Vrows]) ;
        auto face_vertex1_end = make_float3(V1[bvids.y + 0], V1[bvids.y + Vrows], V1[bvids.y + 2*Vrows]) ;
        auto face_vertex2_end = make_float3(V1[bvids.z + 0], V1[bvids.z + Vrows], V1[bvids.z + 2*Vrows]) ;

        // array<array<float, 3>, 8> tmp;
        // float3 tmp[8];
        queries[8*tid + 0] = vertex_start;
        queries[8*tid + 1] = face_vertex0_start;
        queries[8*tid + 2] = face_vertex1_start;
        queries[8*tid + 3] = face_vertex2_start;
        queries[8*tid + 4] = vertex_end;
        queries[8*tid + 5] = face_vertex0_end;
        queries[8*tid + 6] = face_vertex1_end;
        queries[8*tid + 7] = face_vertex2_end;
        
    }
    else if (is_edge(avids) && is_edge(bvids))
    {
        //     // Edge 1 at t=0
        auto edge0_vertex0_start = make_float3(V0[avids.x + 0], V0[avids.x + Vrows], V0[avids.x + 2*Vrows]) ;
        auto edge0_vertex1_start = make_float3(V0[avids.y + 0], V0[avids.y + Vrows], V0[avids.y + 2*Vrows]) ;
        // // Edge 2 at t=0
        auto edge1_vertex0_start = make_float3(V0[bvids.x + 0], V0[bvids.x + Vrows], V0[bvids.x + 2*Vrows]) ;
        auto edge1_vertex1_start = make_float3(V0[bvids.y + 0], V0[bvids.y + Vrows], V0[bvids.y + 2*Vrows]) ;
        // // Edge 1 at t=1
        auto edge0_vertex0_end = make_float3(V1[avids.x + 0], V1[avids.x + Vrows], V1[avids.x + 2*Vrows]) ;
        auto edge0_vertex1_end = make_float3(V1[avids.y + 0], V1[avids.y + Vrows], V1[avids.y + 2*Vrows]) ;
        // // Edge 2 at t=1
        auto edge1_vertex0_end = make_float3(V1[bvids.x + 0], V1[bvids.x + Vrows], V1[bvids.x + 2*Vrows]) ;
        auto edge1_vertex1_end = make_float3(V1[bvids.y + 0], V1[bvids.y + Vrows], V1[bvids.y + 2*Vrows]) ;
        
        // queries.emplace_back(Vec3Conv(edge0_vertex0_start));
        // float3 tmp[8];
        queries[8*tid + 0] = edge0_vertex0_start;
        queries[8*tid + 1] = edge0_vertex1_start;
        queries[8*tid + 2] = edge1_vertex0_start;
        queries[8*tid + 3] = edge1_vertex1_start;
        queries[8*tid + 4] = edge0_vertex0_end;
        queries[8*tid + 5] = edge0_vertex1_end;
        queries[8*tid + 6] = edge1_vertex0_end;
        queries[8*tid + 7] = edge1_vertex1_end;
        // queries[tid] = tmp;
        
    }
    else assert(0);

    if (tid == 12135) 
    {
        printf("V[%i].x: %.6f\n", tid, V0[tid]);
        printf("queries[%i].x: %.3f\n", tid, queries[tid].x);
    }
}

void addData(
    const Aabb &a, 
    const Aabb &b, 
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    vector<array<array<float, 3>, 8>>& queries)
{
    // auto is_vertex = [&](Aabb x){return x.vertexIds.y < 0 ;};
    // auto is_edge = [&](Aabb x){return !is_vertex(x) && x.vertexIds.z < 0 ;};
    // auto is_face = [&](Aabb x){return !is_vertex(x) && !is_edge(x);};

    // auto is_face = [&](Aabb x){return x.vertexIds.z >= 0;};
    // auto is_edge = [&](Aabb x){return x.vertexIds.z < 0 && x.vertexIds.y >= 0 ;};
    // auto is_vertex = [&](Aabb x){return x.vertexIds.z < 0  && x.vertexIds.y < 0;};

    if (is_vertex(a) && is_face(b))
    {
        auto avids = a.vertexIds;
        auto bvids = b.vertexIds;
            // Point at t=0s
        auto vertex_start = V0.cast<float>().row(avids.x);
        // // Triangle at t = 0
        auto face_vertex0_start = V0.cast<float>().row(bvids.x);
        auto face_vertex1_start = V0.cast<float>().row(bvids.y);
        auto face_vertex2_start = V0.cast<float>().row(bvids.z);
        // // Point at t=1
        auto vertex_end = V1.cast<float>().row(avids.x);
        // // Triangle at t = 1
        auto face_vertex0_end = V1.cast<float>().row(bvids.x);
        auto face_vertex1_end = V1.cast<float>().row(bvids.y);
        auto face_vertex2_end = V1.cast<float>().row(bvids.z);

        array<array<float, 3>, 8> tmp;
        tmp[0] = Vec3Conv(vertex_start);
        tmp[1] = Vec3Conv(face_vertex0_start);
        tmp[2] = Vec3Conv(face_vertex1_start);
        tmp[3] = Vec3Conv(face_vertex2_start);
        tmp[4] = Vec3Conv(vertex_end);
        tmp[5] = Vec3Conv(face_vertex0_end);
        tmp[6] = Vec3Conv(face_vertex1_end);
        tmp[7] = Vec3Conv(face_vertex2_end);
        queries.emplace_back(tmp);
    }
    else if (is_face(a) && is_vertex(b))
        return addData(b, a, V0, V1, queries);
    else if (is_edge(a) && is_edge(b))
    {
        auto avids = a.vertexIds;
        auto bvids = b.vertexIds;
        //     // Edge 1 at t=0
        auto edge0_vertex0_start = V0.cast<float>().row(avids.x);
        auto edge0_vertex1_start = V0.cast<float>().row(avids.y);
        // // Edge 2 at t=0
        auto edge1_vertex0_start = V0.cast<float>().row(bvids.x);
        auto edge1_vertex1_start = V0.cast<float>().row(bvids.y);
        // // Edge 1 at t=1
        auto edge0_vertex0_end = V1.cast<float>().row(avids.x);
        auto edge0_vertex1_end = V1.cast<float>().row(avids.y);
        // // Edge 2 at t=1
        auto edge1_vertex0_end = V1.cast<float>().row(bvids.x);
        auto edge1_vertex1_end = V1.cast<float>().row(bvids.y);
        
        // queries.emplace_back(Vec3Conv(edge0_vertex0_start));
        array<array<float, 3>, 8> tmp;
        tmp[0] = Vec3Conv(edge0_vertex0_start);
        tmp[1] = Vec3Conv(edge0_vertex1_start);
        tmp[2] = Vec3Conv(edge1_vertex0_start);
        tmp[3] = Vec3Conv(edge1_vertex1_start);
        tmp[4] = Vec3Conv(edge0_vertex0_end);
        tmp[5] = Vec3Conv(edge0_vertex1_end);
        tmp[6] = Vec3Conv(edge1_vertex0_end);
        tmp[7] = Vec3Conv(edge1_vertex1_end);
        queries.emplace_back(tmp);
    }
    else abort();
}

bool is_file_exist(const char *fileName)
{
    ifstream infile(fileName);
    return infile.good();
}



__global__ void run_parallel_vf_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= size) return;
    // copy the input queries to __device__
    CCDdata data_in;
    for (int i = 0; i < 3; i++)
    {
        data_in.v0s[i] = data[tx].v0s[i];
        data_in.v1s[i] = data[tx].v1s[i];
        data_in.v2s[i] = data[tx].v2s[i];
        data_in.v3s[i] = data[tx].v3s[i];
        data_in.v0e[i] = data[tx].v0e[i];
        data_in.v1e[i] = data[tx].v1e[i];
        data_in.v2e[i] = data[tx].v2e[i];
        data_in.v3e[i] = data[tx].v3e[i];
    }
    // copy the configurations to the shared memory
    __shared__ CCDConfig config;
    config.err_in[0]=config_in->err_in[0];
    config.err_in[1]=config_in->err_in[1];
    config.err_in[2]=config_in->err_in[2];
    config.co_domain_tolerance=config_in->co_domain_tolerance; // tolerance of the co-domain
    config.max_t=config_in->max_t; // the upper bound of the time interval
    config.max_itr=config_in->max_itr;// the maximal nbr of iterations
    CCDOut out;
    vertexFaceCCD(data_in,config, out);
    res[tx] = out.result;
    tois[tx] = 0;
}
__global__ void run_parallel_ee_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= size) return;
    // copy the input queries to __device__
    CCDdata data_in;
    for (int i = 0; i < 3; i++)
    {
        data_in.v0s[i] = data[tx].v0s[i];
        data_in.v1s[i] = data[tx].v1s[i];
        data_in.v2s[i] = data[tx].v2s[i];
        data_in.v3s[i] = data[tx].v3s[i];
        data_in.v0e[i] = data[tx].v0e[i];
        data_in.v1e[i] = data[tx].v1e[i];
        data_in.v2e[i] = data[tx].v2e[i];
        data_in.v3e[i] = data[tx].v3e[i];
    }
    // copy the configurations to the shared memory
    __shared__ CCDConfig config;
    config.err_in[0]=config_in->err_in[0];
    config.err_in[1]=config_in->err_in[1];
    config.err_in[2]=config_in->err_in[2];
    config.co_domain_tolerance=config_in->co_domain_tolerance; // tolerance of the co-domain
    config.max_t=config_in->max_t; // the upper bound of the time interval
    config.max_itr=config_in->max_itr;// the maximal nbr of iterations
    CCDOut out;
    edgeEdgeCCD(data_in,config, out);
    res[tx] = out.result;
    tois[tx] = 0;
}

__global__ void run_parallel_ms_vf_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= size) return;
    // copy the input queries to __device__
    CCDdata data_in;
    for (int i = 0; i < 3; i++)
    {
        data_in.v0s[i] = data[tx].v0s[i];
        data_in.v1s[i] = data[tx].v1s[i];
        data_in.v2s[i] = data[tx].v2s[i];
        data_in.v3s[i] = data[tx].v3s[i];
        data_in.v0e[i] = data[tx].v0e[i];
        data_in.v1e[i] = data[tx].v1e[i];
        data_in.v2e[i] = data[tx].v2e[i];
        data_in.v3e[i] = data[tx].v3e[i];
    }
    data_in.ms=data[tx].ms;
    // copy the configurations to the shared memory
    __shared__ CCDConfig config;
    config.err_in[0]=config_in->err_in[0];
    config.err_in[1]=config_in->err_in[1];
    config.err_in[2]=config_in->err_in[2];
    config.co_domain_tolerance=config_in->co_domain_tolerance; // tolerance of the co-domain
    config.max_t=config_in->max_t; // the upper bound of the time interval
    config.max_itr=config_in->max_itr;// the maximal nbr of iterations
    CCDOut out;
# ifdef NO_CHECK_MS
    vertexFaceCCD(data_in,config, out);
# else
    vertexFaceMinimumSeparationCCD(data_in,config, out);
#endif
    res[tx] = out.result;
    tois[tx] = 0;
}
__global__ void run_parallel_ms_ee_ccd_all(CCDdata *data,CCDConfig *config_in, bool *res, int size, Scalar *tois
)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= size) return;
    // copy the input queries to __device__
    CCDdata data_in;
    for (int i = 0; i < 3; i++)
    {
        data_in.v0s[i] = data[tx].v0s[i];
        data_in.v1s[i] = data[tx].v1s[i];
        data_in.v2s[i] = data[tx].v2s[i];
        data_in.v3s[i] = data[tx].v3s[i];
        data_in.v0e[i] = data[tx].v0e[i];
        data_in.v1e[i] = data[tx].v1e[i];
        data_in.v2e[i] = data[tx].v2e[i];
        data_in.v3e[i] = data[tx].v3e[i];
    }
    data_in.ms=data[tx].ms;
    // copy the configurations to the shared memory
    __shared__ CCDConfig config;
    config.err_in[0]=config_in->err_in[0];
    config.err_in[1]=config_in->err_in[1];
    config.err_in[2]=config_in->err_in[2];
    config.co_domain_tolerance=config_in->co_domain_tolerance; // tolerance of the co-domain
    config.max_t=config_in->max_t; // the upper bound of the time interval
    config.max_itr=config_in->max_itr;// the maximal nbr of iterations
    CCDOut out;
# ifdef NO_CHECK_MS
    edgeEdgeCCD(data_in,config, out);
# else
   edgeEdgeMinimumSeparationCCD(data_in,config, out);
#endif
    res[tx] = out.result;
    tois[tx] = 0;
}

__global__ void array_to_ccd(float3 * a, int tmp_nbr, CCDdata * data )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= tmp_nbr) return;

    #ifndef NO_CHECK_MS
    data[tid].ms=MINIMUM_SEPARATION_BENCHMARK;
    #endif
    
    if (tid == 0)
        printf("for tid[0] %.3f\n", a[0].x );
    
  
    data[tid].v0s[0] = a[8*tid + 0].x;
    data[tid].v1s[0] = a[8*tid + 1].x;
    data[tid].v2s[0] = a[8*tid + 2].x;
    data[tid].v3s[0] = a[8*tid + 3].x;
    data[tid].v0e[0] = a[8*tid + 4].x;
    data[tid].v1e[0] = a[8*tid + 5].x;
    data[tid].v2e[0] = a[8*tid + 6].x;
    data[tid].v3e[0] = a[8*tid + 7].x;

    data[tid].v0s[1] = a[8*tid + 0].y;
    data[tid].v1s[1] = a[8*tid + 1].y;
    data[tid].v2s[1] = a[8*tid + 2].y;
    data[tid].v3s[1] = a[8*tid + 3].y;
    data[tid].v0e[1] = a[8*tid + 4].y;
    data[tid].v1e[1] = a[8*tid + 5].y;
    data[tid].v2e[1] = a[8*tid + 6].y;
    data[tid].v3e[1] = a[8*tid + 7].y;

    data[tid].v0s[2] = a[8*tid + 0].z;
    data[tid].v1s[2] = a[8*tid + 1].z;
    data[tid].v2s[2] = a[8*tid + 2].z;
    data[tid].v3s[2] = a[8*tid + 3].z;
    data[tid].v0e[2] = a[8*tid + 4].z;
    data[tid].v1e[2] = a[8*tid + 5].z;
    data[tid].v2e[2] = a[8*tid + 6].z;
    data[tid].v3e[2] = a[8*tid + 7].z;
}

void all_ccd_run(float3 * V, int tmp_nbr, bool is_edge,
    std::vector<bool> &result_list, double &run_time, std::vector<Scalar> &time_impact, int parallel_nbr)
{
    int nbr = tmp_nbr;
    result_list.resize(nbr);
    // host
    // CCDdata *data_list = new CCDdata[nbr];
    CCDdata *data_list;
    cudaMalloc((void**)&data_list, sizeof(CCDdata)*nbr);
    array_to_ccd<<<nbr / parallel_nbr + 1, parallel_nbr>>>( V, nbr, data_list);
    cudaDeviceSynchronize();
    gpuErrchk( cudaGetLastError() ); 
    printf("Finished array_to_ccd\n");

    bool *res = new bool[nbr];
    Scalar *tois = new Scalar[nbr];
    CCDConfig *config=new CCDConfig[1];
    config[0].err_in[0]=-1;// the input error bound calculate from the AABB of the whole mesh
    config[0].co_domain_tolerance=1e-6; // tolerance of the co-domain
    config[0].max_t=1; // the upper bound of the time interval
    config[0].max_itr=1e6;// the maximal nbr of iterations

    // device
    CCDdata *d_data_list;
    bool *d_res;
    Scalar *d_tois;
    CCDConfig *d_config;

    int data_size = sizeof(CCDdata) * nbr;
    int result_size = sizeof(bool) * nbr;
    int time_size = sizeof(Scalar) * nbr;
    // int dbg_size=sizeof(Scalar)*8;

    cudaMalloc(&d_data_list, data_size);
    cudaMalloc(&d_res, result_size);
    cudaMalloc(&d_tois, time_size);
    cudaMalloc(&d_config, sizeof(CCDConfig));

    cudaMemcpy(d_data_list, data_list, data_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_config, config, sizeof(CCDConfig), cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() ); 

    ccd::Timer timer;
    cudaProfilerStart();
    timer.start();
    #ifdef NO_CHECK_MS
    if(is_edge){
    run_parallel_ee_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
    d_data_list,d_config, d_res, nbr, d_tois);
    }
    else{
    run_parallel_vf_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
    d_data_list,d_config, d_res, nbr, d_tois);
    }
    #else
    if(is_edge){
    run_parallel_ms_ee_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
    d_data_list,d_config, d_res, nbr, d_tois);
    }
    else{
    run_parallel_ms_vf_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
    d_data_list,d_config, d_res, nbr, d_tois);
    }
    #endif

    cudaDeviceSynchronize();
    double tt = timer.getElapsedTimeInMicroSec();
    run_time = tt;
    cudaProfilerStop();

    cudaMemcpy(res, d_res, result_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(tois, d_tois, time_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(dbg, d_dbg, dbg_size, cudaMemcpyDeviceToHost);

    cudaFree(data_list);
    cudaFree(d_data_list);
    cudaFree(d_res);
    cudaFree(d_tois);
    cudaFree(d_config);
    //cudaFree(d_dbg);

    for (int i = 0; i < nbr; i++)
    {
    result_list[i] = res[i];
    }

    time_impact.resize(nbr);

    for (int i = 0; i < nbr; i++)
    {
    time_impact[i] = tois[i];
    }
    // std::cout << "dbg info\n"
    //           << dbg[0] << "," << dbg[1] << "," << dbg[2] << "," << dbg[3] << "," << dbg[4] << "," << dbg[5] << "," << dbg[6] << "," << dbg[7] << std::endl;
    delete[] res;
    // delete[] data_list;
    delete[] tois;
    delete[] config;
    //delete[] dbg;
    cudaError_t ct = cudaGetLastError();
    printf("******************\n%s\n************\n", cudaGetErrorString(ct));

    return;
}

void all_ccd_run(const std::vector<std::array<std::array<Scalar, 3>, 8>> &V, bool is_edge,
    std::vector<bool> &result_list, double &run_time, std::vector<Scalar> &time_impact, int parallel_nbr)
{
gpuErrchk( cudaGetLastError() ); 

int nbr = V.size();
result_list.resize(nbr);
// host
CCDdata *data_list = new CCDdata[nbr];
for (int i = 0; i < nbr; i++)
{
data_list[i] = array_to_ccd( V[i]);
#ifndef NO_CHECK_MS
data_list[i].ms=MINIMUM_SEPARATION_BENCHMARK;
#endif
}

bool *res = new bool[nbr];
Scalar *tois = new Scalar[nbr];
CCDConfig *config=new CCDConfig[1];
config[0].err_in[0]=-1;// the input error bound calculate from the AABB of the whole mesh
config[0].co_domain_tolerance=1e-6; // tolerance of the co-domain
config[0].max_t=1; // the upper bound of the time interval
config[0].max_itr=1e6;// the maximal nbr of iterations

// device
CCDdata *d_data_list;
bool *d_res;
Scalar *d_tois;
CCDConfig *d_config;

int data_size = sizeof(CCDdata) * nbr;
int result_size = sizeof(bool) * nbr;
int time_size = sizeof(Scalar) * nbr;
// int dbg_size=sizeof(Scalar)*8;

cudaMalloc(&d_data_list, data_size);
cudaMalloc(&d_res, result_size);
cudaMalloc(&d_tois, time_size);
cudaMalloc(&d_config, sizeof(CCDConfig));

cudaMemcpy(d_data_list, data_list, data_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_config, config, sizeof(CCDConfig), cudaMemcpyHostToDevice);
gpuErrchk( cudaGetLastError() ); 

ccd::Timer timer;
cudaProfilerStart();
timer.start();
#ifdef NO_CHECK_MS
if(is_edge){
run_parallel_ee_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
d_data_list,d_config, d_res, nbr, d_tois);
}
else{
run_parallel_vf_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
d_data_list,d_config, d_res, nbr, d_tois);
}
#else
if(is_edge){
run_parallel_ms_ee_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
d_data_list,d_config, d_res, nbr, d_tois);
}
else{
run_parallel_ms_vf_ccd_all<<<nbr / parallel_nbr + 1, parallel_nbr>>>( 
d_data_list,d_config, d_res, nbr, d_tois);
}
#endif

cudaDeviceSynchronize();
double tt = timer.getElapsedTimeInMicroSec();
run_time = tt;
cudaProfilerStop();

cudaMemcpy(res, d_res, result_size, cudaMemcpyDeviceToHost);
cudaMemcpy(tois, d_tois, time_size, cudaMemcpyDeviceToHost);
//cudaMemcpy(dbg, d_dbg, dbg_size, cudaMemcpyDeviceToHost);

cudaFree(d_data_list);
cudaFree(d_res);
cudaFree(d_tois);
cudaFree(d_config);
//cudaFree(d_dbg);

for (int i = 0; i < nbr; i++)
{
result_list[i] = res[i];
}

time_impact.resize(nbr);

for (int i = 0; i < nbr; i++)
{
time_impact[i] = tois[i];
}
// std::cout << "dbg info\n"
//           << dbg[0] << "," << dbg[1] << "," << dbg[2] << "," << dbg[3] << "," << dbg[4] << "," << dbg[5] << "," << dbg[6] << "," << dbg[7] << std::endl;
delete[] res;
delete[] data_list;
delete[] tois;
delete[] config;
//delete[] dbg;
cudaError_t ct = cudaGetLastError();
printf("******************\n%s\n************\n", cudaGetErrorString(ct));

return;
}
