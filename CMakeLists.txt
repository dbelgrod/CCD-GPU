cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(CCDGPU LANGUAGES CXX CUDA)

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
  set(CCDGPU_TOPLEVEL_PROJECT ON)
else()
  set(CCDGPU_TOPLEVEL_PROJECT OFF)
endif()

### Configuration
set(CCDGPU_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Warnings)
include(UseColors)
include(CCDGPU_Utils)

include(${PROJECT_NAME}_Dependencies)

### the library
add_subdirectory(src)
prepend_current_path(CCDGPU_SOURCES)
CCDGPU_copy_headers(${CCDGPU_SOURCES})
CCDGPU_set_source_group(${CCDGPU_SOURCES})
add_library(${PROJECT_NAME} STATIC ${CCDGPU_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR}/include)
# target_link_libraries(CCDGPU PRIVATE warnings::all)
target_compile_definitions(${PROJECT_NAME} PUBLIC NOMINMAX)

# Request that CCDGPU be built with -std=c++11
# As this is a public compile feature anything that links to CCDGPU
# will also build with -std=c++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)


################################################################################
# CUDA
##########
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-v -use_fast_math -maxrregcount 24")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -Xptxas -O3,-v")

# We need to explicitly state that we need all CUDA files in the CCDGPU
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

# # executable
# add_executable(CCDGPU_bin app/main.cu)
# set_target_properties(CCDGPU_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_property(TARGET CCDGPU_bin PROPERTY CUDA_ARCHITECTURES 61 70 75 86)
# target_link_libraries(CCDGPU_bin PRIVATE ${PROJECT_NAME})
                  
                  
if(APPLE)
# We need to add the default path to the driver (libcuda.dylib) as an rpath,
# so that the static cuda runtime can find it at runtime.
# set_property(TARGET CCDGPU_bin PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


################################################################################
# Required Libraries
################################################################################

###############################################################################

# target_compile_options(CCDGPU_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                        --generate-line-info
#                        --use_fast_math
#                        --relocatable-device-code=true
#                        --ptxas-options=-v
#                       # --maxrregcount=7
#                        >)  
