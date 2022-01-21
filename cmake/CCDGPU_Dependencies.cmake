# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.


# Download and update 3rd_party libraries
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
include(CCDGPU_DownloadExternal)

# broadphase
if(NOT TARGET broadphase)
    ccdgpu_download_broadphase()
    add_subdirectory(${CCDGPU_EXTERNAL}/broadphase EXCLUDE_FROM_ALL)
endif()

# # narrowphase
# if(NOT TARGET gputi)
#     ccdgpu_download_narrowphase()
#     add_subdirectory(${CCDGPU_EXTERNAL}/narrowphase EXCLUDE_FROM_ALL)
# endif()