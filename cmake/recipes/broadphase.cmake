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
        GIT_REPOSITORY https://github.com/dbelgrod/broadphase-gpu.git
        GIT_TAG be43935557d229f13c912d4983026d11d8ab7025
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(broadphase)
endif()

add_library(broadphase::broadphase ALIAS GPUBF)
