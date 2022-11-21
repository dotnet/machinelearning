project (OneDalNative)

# libraries
if(UNIX)
    if(APPLE)
        set(WRAPPER_LINK_OPTIONS -L${ONEDAL_REDIST_PATH}/lib -l:libonedal_core.1.1.dylib -l:libonedal_thread.1.1.dylib -L${ONETBB_LIB_PATH} -ltbb -ltbbmalloc)
    else()
        set(WRAPPER_LINK_OPTIONS -L${ONEDAL_REDIST_PATH}/lib/intel64 -l:libonedal_core.so.1.1 -l:libonedal_thread.so.1.1 -L${ONETBB_LIB_PATH} -ltbb -ltbbmalloc)
    endif()
else()
    set(PATH ${ONEDAL_REDIST_PATH}/redist/intel64;${ONETBB_LIB_PATH};${PATH})
    set(WRAPPER_LINK_OPTIONS onedal_core.1.dll onedal_thread.1.dll tbb12.dll tbbmalloc.dll)
endif()

# headers
include_directories(${ONEDAL_DEVEL_PATH}/include)

set(SOURCES OneDalAlgorithms.cpp)
add_library(OneDalNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(OneDalNative PUBLIC ${WRAPPER_LINK_OPTIONS})
install_library_and_symbols(OneDalNative)
