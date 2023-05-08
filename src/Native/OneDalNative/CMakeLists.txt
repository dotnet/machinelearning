project (OneDalNative)

# libraries
if(UNIX)
    if(APPLE)
        set(WRAPPER_LINK_OPTIONS -L${ONEDAL_REDIST_PATH}/lib libonedal_core.1.1.dylib libonedal_thread.1.1.dylib -L${ONETBB_REDIST_PATH} libtbb.12.dylib libtbbmalloc.2.dylib)
    else()
        set(WRAPPER_LINK_OPTIONS -L${ONEDAL_REDIST_PATH}/lib/intel64 -l:libonedal_core.so.1.1 -l:libonedal_thread.so.1.1 -L${ONETBB_REDIST_PATH} -l:libtbb.so.12 -l:libtbbmalloc.so.2 "-Wl,-rpath,'$ORIGIN'")
    endif()
else()
    link_directories("${ONEDAL_DEVEL_PATH}\\lib\\intel64")
    link_directories("${ONETBB_DEVEL_PATH}")
    set(WRAPPER_LINK_OPTIONS onedal_core_dll.1.lib tbb12.lib tbbmalloc.lib)
endif()

# headers
include_directories(${ONEDAL_DEVEL_PATH}/include)

if(WIN32)
    add_compile_options(/wd4996)
    add_compile_options(/wd4477)
    add_compile_options(/wd4267)
endif()

set(SOURCES OneDalAlgorithms.cpp)
add_library(OneDalNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(OneDalNative PUBLIC ${WRAPPER_LINK_OPTIONS})
install_library_and_symbols(OneDalNative)
