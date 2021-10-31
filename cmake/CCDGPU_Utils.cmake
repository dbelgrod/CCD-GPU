function(CCDGPU_copy_headers)
	foreach(filepath IN ITEMS ${ARGN})
		get_filename_component(filename "${filepath}" NAME)
		if(${filename} MATCHES ".*\.(hpp|h|ipp)$")
			configure_file(${filepath} ${PROJECT_BINARY_DIR}/include/CCDGPU/${filename})
		endif()
	endforeach()
endfunction()

function(prepend_current_path SOURCE_FILES)
	# Use recursive substitution to expand SOURCE_FILES
	unset(MODIFIED)
	foreach(SOURCE_FILE IN ITEMS ${${SOURCE_FILES}})
		list(APPEND MODIFIED "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
	endforeach()
	set(${SOURCE_FILES} ${MODIFIED} PARENT_SCOPE)
endfunction()

function(CCDGPU_set_source_group)
	foreach(filepath IN ITEMS ${ARGN})
		get_filename_component(folderpath "${filepath}" DIRECTORY)
		get_filename_component(foldername "${folderpath}" NAME)
		source_group(foldername FILES "${filepath}")
	endforeach()
endfunction()