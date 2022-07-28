#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ccdgpu/timer.hpp>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

namespace ccd::gpu {

// template <typename... Arguments>
// void recordLaunch(char* tag, void(*f)(Arguments...), Arguments... args) {
//       Timer timer;
//       timer.start();
//       f(args...);
//       timer.stop();
//       double elapsed = 0;
//       elapsed += timer.getElapsedTimeInMicroSec();
//       spdlog::trace("{} : {:.6f} ms", tag, elapsed);
// };

struct Record {
  ccd::Timer timer;
  cudaEvent_t start, stop;
  char *tag;
  json j_object;
  bool gpu_timer_on = false;

  Record(){
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // gpu_timer_on = false;
  };

  Record(json &jtmp) {
    j_object = jtmp;
    Record();
  };

  void Start(char *s, bool gpu = false) {
    tag = s;
    if (!gpu)
      timer.start();
    else {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      // spdlog::trace("Starting gpu timer for {}",  s);
      cudaEventRecord(start);
      gpu_timer_on = true;
    }
  }

  void Start(char *s, json &jtmp, bool gpu = false) {
    j_object = jtmp;
    Start(s, gpu);
  }

  void Stop() {
    float elapsed; // was double
    if (!gpu_timer_on) {
      timer.stop();
      elapsed += timer.getElapsedTimeInMilliSec();
      spdlog::trace("Cpu timer stopped for {}: {:.6f} ms", tag, elapsed);
    } else {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed, start, stop);
      spdlog::trace("Gpu timer stopped for {}: {:.6f} ms", tag, elapsed);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      gpu_timer_on = false;
    }
    // j_object[tag]=elapsed;
    if (j_object.contains(tag))
      j_object[tag] = (double)j_object[tag] + elapsed;
    else
      j_object[tag] = elapsed;
    spdlog::trace("{} : {:.3f} ms", tag, elapsed);
  }

  void Print() { spdlog::info("{}", j_object.dump()); }

  json Dump() { return j_object; }
};

} // namespace ccd::gpu