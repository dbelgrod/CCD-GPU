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
