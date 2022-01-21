#pragma once
#include<vector>
namespace ccd{
class CudaTimer {
    cudaEvent_t	_start, _end;
    std::vector<std::pair<std::string, float>>	_records;
    int lastSectionLen{ 0 };
public:
    CudaTimer() {
        cudaEventCreate(&_start);
        cudaEventCreate(&_end);
    }
    ~CudaTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_end);
    }
    void	tick(cudaStream_t streamid = 0) { cudaEventRecord(_start, streamid); }
	void	tock(cudaStream_t streamid = 0) { cudaEventRecord(_end, streamid); }
    float	elapsed() const {
        float ms;
        cudaEventSynchronize(_end);
        cudaEventElapsedTime(&ms, _start, _end);
        return ms;
    }
    void	clear() { _records.clear(); }
    void	record(std::string tag) {
        tock();
        _records.push_back(make_pair(tag, elapsed()));
    }
}; //CudaTimer

template <typename... Arguments>
void recordLaunch(char* tag, int gs, int bs, size_t mem, void(*f)(Arguments...), Arguments... args) {
    CudaTimer timer;
    if (!mem) {
        timer.tick();
        f << <gs, bs >> >(args...);
        timer.tock();
    } else {
        timer.tick();
        f << <gs, bs, mem >> >(args...);
        timer.tock();
    }
		
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) 
    {
        printf("Kernel launch failure %s\nTrying device-kernel launch\n", cudaGetErrorString(error));
        timer.tick();
        f(args...);
        timer.tock();
        cudaError_t err = cudaGetLastError();
	    if (err != cudaSuccess) printf("Device-kernel launch failure %s\n", cudaGetErrorString(err));
    }
    double elapsed = timer.elapsed();
    printf("%s : %.6f ms\n", tag, elapsed );
    return;
};

template <typename... Arguments>
void recordLaunch(char* tag, int gs, int bs, void(*f)(Arguments...), Arguments... args) {
    size_t mem = 0;
    recordLaunch(tag, gs, bs, mem, f, args...);
};


// __device__ void recordLaunch(char* tag, std::function<void()> f) {
//     clock_t start = clock();

//     f();
//     clock_t stop = clock();
//     clock_t t = stop - start;
//     if (threadIdx.x+blockIdx.x == 0)
//         printf ("%s: %d clicks (%f ms).\n", tag, t,((float)t)/CLOCKS_PER_SEC*1000);       
// };


template <typename... Arguments>
__device__ void recordLaunch(char* tag, void (*f)(Arguments...), Arguments... args) {
    clock_t start, stop;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        start = clock();

    f(args...);
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        stop = clock();
        clock_t t = stop - start;
        printf ("%s: %d clicks (%f ms).\n", tag, t,(float)t/(float)CLOCKS_PER_SEC*1000.0f);
    }     
};


// template <typename... Arguments>
// __device__ void recordLaunch(char* tag, void (*f)(Arguments...), Arguments&&... args) {
//     clock_t start = clock();

//     f(std::forward<Arguments>(args)...);

//     clock_t stop = clock();
//     clock_t t = stop - start;
//     if (threadIdx.x+blockIdx.x == 0)
//         printf ("%s: %d clicks (%f ms).\n", tag, t,((float)t)/CLOCKS_PER_SEC*1000);       
// };

template <typename Fun, typename... Arguments>
__device__ Fun recordLaunch(char* tag, Fun (*f)(Arguments...), Arguments... args) {
    
    // int gs = 0;
    // int bs = 0;
    // recordLaunch(tag, gs, bs, f, args...);
    clock_t start, stop;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        start = clock();

    Fun res = f(args...);
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        stop = clock();
        clock_t t = stop - start;
        printf ("%s: %d clicks (%f ms).\n", tag, t,(float)t/(float)CLOCKS_PER_SEC*1000.0f);
    }
    return res;
};

}

