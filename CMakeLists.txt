cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(CCDGPU LANGUAGES CXX CUDA)


SET(CUDA_SEPARABLE_COMPILATION ON)

set(LIBIGL_WITH_COMISO OFF CACHE BOOL "Use CoMiso")
set(LIBIGL_WITH_EMBREE OFF  CACHE BOOL  "Use Embree"  )
set(LIBIGL_WITH_OPENGL            OFF CACHE BOOL "Use OpenGL"          )
set(LIBIGL_WITH_OPENGL_GLFW       OFF CACHE BOOL "Use GLFW"            )
set(LIBIGL_WITH_OPENGL_GLFW_IMGUI OFF CACHE BOOL "Use ImGui"           )
set(LIBIGL_WITH_PNG               OFF CACHE BOOL "Use PNG"             )
set(LIBIGL_WITH_TETGEN            OFF CACHE BOOL "Use Tetgen"          )
set(LIBIGL_WITH_TRIANGLE          OFF CACHE BOOL "Use Triangle"        )
set(LIBIGL_WITH_PREDICATES        ON CACHE BOOL "Use exact predicates")
set(LIBIGL_WITH_XML               OFF CACHE BOOL "Use XML"             )
set(LIBIGL_WITH_PYTHON            OFF CACHE BOOL "Use Python"          )
# set(CCDGPU_BROADPHASE_PATH ${PROJECT_SOURCE_DIR}/repo/broadphase)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CCDGPU_TOPLEVEL_PROJECT ON)
else()
  set(CCDGPU_TOPLEVEL_PROJECT OFF)
endif()


OPTION(GPUTI_WITH_DOUBLE_PRECISION   "use double precision calculation" ON)
OPTION(CCD_TOI_PER_QUERY   "record symbolic toi per each query in json" OFF)


### Configuration
# set(CCDGPU_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/external")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")
include(Warnings)
include(UseColors)
include(CCDGPU_Utils)

### the library
add_subdirectory(src)
prepend_current_path(CCDGPU_SOURCES)
CCDGPU_copy_headers(${CCDGPU_SOURCES})
CCDGPU_set_source_group(${CCDGPU_SOURCES})
add_library(${PROJECT_NAME} STATIC ${CCDGPU_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR}/include)
# target_link_libraries(CCDGPU PRIVATE warnings::all)
target_compile_definitions(${PROJECT_NAME} PUBLIC NOMINMAX)

if(GPUTI_WITH_DOUBLE_PRECISION)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DGPUTI_USE_DOUBLE_PRECISION)
endif()
if(CCD_TOI_PER_QUERY)
  target_compile_definitions(${PROJECT_NAME} PUBLIC CCD_TOI_PER_QUERY)
endif()

# Request that CCDGPU be built with -std=c++11
# As this is a public compile feature anything that links to CCDGPU
# will also build with -std=c++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)


################################################################################
# CUDA
################################################################################
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-v -use_fast_math -maxrregcount 24")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -Xptxas -O3,-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr  -gencode arch=compute_86,code=sm_86")

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
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70 75 86)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       # --ptxas-options=-v
                       # --maxrregcount=7
                       >
                       -fPIC) #https://stackoverflow.com/questions/5311515/gcc-fpic-option

################################################################################
# Required Libraries
################################################################################

include(broadphase)
target_link_libraries(${PROJECT_NAME} PUBLIC STQ::CUDA)

# target_link_libraries(${PROJECT_NAME} PUBLIC gputi)

# libigl
target_link_libraries(${PROJECT_NAME} PUBLIC igl::core)

# JSON
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann::json)

# gmp
if(CCD_TOI_PER_QUERY)
find_package(GMP)
IF(NOT ${GMP_FOUND})
        MESSAGE(FATAL_ERROR "Cannot find GMP")
ENDIF()
target_include_directories(${PROJECT_NAME} PUBLIC ${GMP_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${GMP_LIBRARIES})
endif()

# target_link_libraries(${PROJECT_NAME} PUBLIC ipc_toolkit)

# Logger
include(spdlog)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog)

################################################################################
# Binary
################################################################################

if(CCDGPU_TOPLEVEL_PROJECT)
    # executable
    add_executable(ccdgpu_bin app/main.cu)
    set_target_properties(ccdgpu_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ccdgpu_bin PROPERTY CUDA_ARCHITECTURES 70 75 86)
    target_link_libraries(ccdgpu_bin PRIVATE ${PROJECT_NAME})


    if(APPLE)
        # We need to add the default path to the driver (libcuda.dylib) as an rpath,
        # so that the static cuda runtime can find it at runtime.
        set_property(TARGET ccdgpu_bin PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
    target_compile_options(
        ccdgpu_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        --ptxas-options=-v
        # --maxrregcount=7
        >
        -fPIC) #https://stackoverflow.com/questions/5311515/gcc-fpic-option
endif()
