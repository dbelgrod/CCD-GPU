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
    message(WARNING "${CCDGPU_BROADPHASE_PATH} does not exist, fetching Broadphase")
    include(FetchContent)
    FetchContent_Declare(
        broadphase
        GIT_REPOSITORY https://github.com/dbelgrod/broadphase-gpu.git
        GIT_TAG 6c9664e7243b0dde374fcc547d47b1e0e7a4c2e1
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(broadphase)
endif()
