if(TARGET broadphase::broadphase)
    return()
endif()

message(STATUS "Third-party: creating target 'broadphase::broadphase'")

if(EXISTS "${CCDGPU_BROADPHASE_PATH}")
    message(STATUS "Using Broadphase found at: ${CCDGPU_BROADPHASE_PATH}")
    add_subdirectory("${CCDGPU_BROADPHASE_PATH}" "${PROJECT_BINARY_DIR}/broadphase")
else()
    include(FetchContent)
    FetchContent_Declare(
        broadphase
        GIT_REPOSITORY git@github.com:dbelgrod/broadphase-gpu.git
        GIT_TAG 4304b983011943e632c73b81418baf0f454e58d9
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(broadphase)
endif()

add_library(broadphase::broadphase ALIAS GPUBF)
