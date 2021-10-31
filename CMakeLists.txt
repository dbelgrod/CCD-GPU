cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(GPUBF LANGUAGES CXX CUDA)

option(LIBIGL_WITH_COMISO            "Use CoMiso"                   OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"                   OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"                   OFF)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     OFF)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    OFF)
option(LIBIGL_WITH_PNG               "Use PNG"                      OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"                   OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 OFF)
option(LIBIGL_WITH_PREDICATES        "Use exact predicates"         OFF)
option(LIBIGL_WITH_XML               "Use XML"                      OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"                   OFF)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(GPUBF_TOPLEVEL_PROJECT ON)
else()
  set(GPUBF_TOPLEVEL_PROJECT OFF)
endif()

### Configuration
set(GPUBF_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Warnings)
include(UseColors)
include(GPUBF_Utils)

include(${PROJECT_NAME}_Dependencies)

### the library
add_subdirectory(src)
prepend_current_path(GPUBF_SOURCES)
gpubf_copy_headers(${GPUBF_SOURCES})
GPUBF_set_source_group(${GPUBF_SOURCES})
add_library(GPUBF STATIC ${GPUBF_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR}/include)
# target_link_libraries(GPUBF PRIVATE warnings::all)
target_compile_definitions(${PROJECT_NAME} PUBLIC NOMINMAX)

# Request that GPUBF be built with -std=c++11
# As this is a public compile feature anything that links to GPUBF
# will also build with -std=c++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)


################################################################################
# CUDA
##########
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-v -use_fast_math -maxrregcount 24")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -Xptxas -O3,-v")

# We need to explicitly state that we need all CUDA files in the GPUBF
# library to be built with -dc as the member functions could be called by
# other libraries and executables

set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
                      
# Nvidia RTX8000 -> compute_75
# Nvidia V100 -> compute_70
# Nvidia 1080/1080Ti -> compute_61       
# Nvidia 3080Ti -> compute_86               
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 70 75 86)

# executable
add_executable(GPUBF_bin app/main.cu)
set_target_properties(GPUBF_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET GPUBF_bin PROPERTY CUDA_ARCHITECTURES 61 70 75 86)
target_link_libraries(GPUBF_bin PRIVATE ${PROJECT_NAME})
                  
                  
if(APPLE)
# We need to add the default path to the driver (libcuda.dylib) as an rpath,
# so that the static cuda runtime can find it at runtime.
set_property(TARGET GPUBF_bin PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


################################################################################
# Required Libraries
################################################################################

# libigl
target_link_libraries(${PROJECT_NAME} PUBLIC igl::core)

# JSON
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json::nlohmann_json)

###############################################################################

target_compile_options(GPUBF_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       --ptxas-options=-v
                      # --maxrregcount=7
                       >)  

################################################################################
# C++ code
################################################################################
add_executable(cpusweep_bin app/main.cpp src/sweep.cpp src/aabb.cpp)
add_library(CPUSWEEP STATIC src/sweep.cpp src/aabb.cpp src/aabb.h src/timer.hpp)
# target_include_directories(CPUSWEEP PUBLIC ${PROJECT_BINARY_DIR}/include)
target_link_libraries(CPUSWEEP PRIVATE warnings::all)
target_compile_definitions(CPUSWEEP PUBLIC NOMINMAX)

# Request that GPUBF be built with -std=c++11
# As this is a public compile feature anything that links to GPUBF
# will also build with -std=c++11
target_compile_features(CPUSWEEP PUBLIC cxx_std_17)

# TBB
# include(tbb)
target_link_libraries(CPUSWEEP PUBLIC tbb::tbb)


target_link_libraries(CPUSWEEP PUBLIC igl::core)

# JSON
target_link_libraries(CPUSWEEP PUBLIC nlohmann_json::nlohmann_json)


target_link_libraries(cpusweep_bin PRIVATE CPUSWEEP)