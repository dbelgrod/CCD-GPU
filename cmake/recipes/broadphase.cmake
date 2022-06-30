if(TARGET STQ::CUDA)
    return()
endif()

message(STATUS "Third-party: creating target 'STQ::CUDA'")

option(STQ_WITH_CPU "Enable CPU Implementation" OFF)
option(STQ_WITH_CUDA "Enable CUDA Implementation" ON)

if(EXISTS "${CCDGPU_BROADPHASE_PATH}")
    message(STATUS "Using Broadphase found at: ${CCDGPU_BROADPHASE_PATH}")
    add_subdirectory("${CCDGPU_BROADPHASE_PATH}" "${PROJECT_BINARY_DIR}/broadphase")
else()
    include(FetchContent)
    FetchContent_Declare(
        broadphase
        GIT_REPOSITORY https://github.com/dbelgrod/broadphase-gpu.git
        GIT_TAG d657455b4a8282438b541569d2c43de9b4bda9f9
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(broadphase)
endif()
