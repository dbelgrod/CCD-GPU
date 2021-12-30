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
      char * tag;
      json j_object;

      Record(){};

      Record(json & jtmp)
      {
          j_object = jtmp;
      };

      void Start(char * s)
      {
            tag = s;
            timer.start();
      }

      void Start(char * s, json & jtmp)
      {
           j_object = jtmp;
           Start(s);
      } 

      void Stop()
      {
            timer.stop();
            double elapsed = 0;
            elapsed += timer.getElapsedTimeInMicroSec();
            elapsed /= 1000.f;
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