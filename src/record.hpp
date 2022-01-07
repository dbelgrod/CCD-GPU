// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
// High Resolution Timer.
//
// Resolution on Mac (clock tick)
// Resolution on Linux (1 us not tested)
// Resolution on Windows (clock tick not tested)

// clang-format off

#pragma once
#include <stdio.h>
#include <iostream>
#include <gputi/timer.hpp>

#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace ccd;

// template <typename... Arguments>
// void recordLaunch(char* tag, void(*f)(Arguments...), Arguments... args) {
//       Timer timer;
//       timer.start();
//       f(args...);
//       timer.stop();
//       double elapsed = 0;
//       elapsed += timer.getElapsedTimeInMicroSec();
//       printf("%s : %.6f ms\n", tag, elapsed );
// };

struct Record
{
      ccd::Timer timer;
      cudaEvent_t start, stop;
      char * tag;
      json j_object;
      bool gpu_timer_on;

      Record(){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            gpu_timer_on = false;
      };

      Record(json & jtmp)
      {
          j_object = jtmp;
          Record();
      };

      void Start(char * s, bool gpu = false)
      {
            tag = s;
            if (!gpu)
                  timer.start();
            else
            {
                  printf("Starting gpu timer for %s\n", s);
                  cudaEventRecord(start);
                  gpu_timer_on = true;
            }

      }

      void Start(char * s, json & jtmp, bool gpu = false)
      {
           j_object = jtmp;
           Start(s, gpu);
      } 

      void Stop()
      {
            float elapsed; //was double
            if (!gpu_timer_on)
            {
                  timer.stop();
                  elapsed += timer.getElapsedTimeInMicroSec();
                  elapsed /= 1000.f;
            }
            else
            {
                  cudaEventRecord(stop);
                  cudaEventSynchronize(stop);
                  cudaEventElapsedTime(&elapsed, start, stop);
                  printf("Gpu timer stopped for %s: %.6f ms\n", tag, elapsed);
                  gpu_timer_on = false;
            }
            // j_object[tag]=elapsed;
            if (j_object.contains(tag))
                  j_object[tag] = (double)j_object[tag] + elapsed;
            else
                  j_object[tag] = elapsed;
            printf("%s : %.3f ms\n", tag, elapsed);
      }

      void Print()
      {
            std::cout << j_object.dump() << std::endl;  
      }
      
      json Dump()
      {
            return j_object;
      }
};