include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(CCDGPU_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(CCDGPU_EXTRA_OPTIONS "")
endif()

function(CCDGPU_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${CCDGPU_EXTERNAL}/${name}
        DOWNLOAD_DIR ${CCDGPU_EXTERNAL}/.cache/${name}
        QUIET
        ${CCDGPU_EXTRA_OPTIONS}
        ${ARGN}
    )
endfunction()

################################################################################

## broadphase
function(ccdgpu_download_broadphase)
    ccdgpu_download_project(broadphase
      GIT_REPOSITORY https://github.com/dbelgrod/broadphase-gpu.git ##https://github.com/dbelgrod/bruteforce-gpu.git
      GIT_TAG main
    )
endfunction()

# ## narrowphase
# function(ccdgpu_download_narrowphase)
#     ccdgpu_download_project(narrowphase
#       GIT_REPOSITORY https://github.com/wangbolun300/GPUTI.git ##https://github.com/wangbolun300/GPUTI.git
#       GIT_TAG master #b7c7c9f2b3a2f62a81fe0c064460071794f128a8
#     )
# endfunction()