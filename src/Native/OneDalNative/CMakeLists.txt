project (OneDalNative)

if(NOT ${ARCHITECTURE} MATCHES "arm.*")
    add_definitions(-DUSESSE)
endif()

# libraries
if(UNIX)
    if(APPLE)
        set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${ONEDAL_REDIST_PATH}/lib ${ONETBB_LIB_PATH})
        link_libraries(libonedal_core.1.1.dylib libonedal_thread.1.1.dylib tbb tbbmalloc)
    else()
        set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${ONEDAL_REDIST_PATH}/lib/intel64 ${ONETBB_LIB_PATH})
        link_libraries(libonedal_core.so.1.1 libonedal_thread.so.1.1 tbb tbbmalloc)
    endif()
else()
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${ONEDAL_REDIST_PATH}/lib/intel64 ${ONETBB_LIB_PATH})
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${ONEDAL_REDIST_PATH}/redist/intel64)
    link_libraries(onedal_core.1.dll onedal_thread.1.dll tbb12.dll tbbmalloc.dll)
endif()

# headers
include_directories(${ONEDAL_DEVEL_PATH}/include)

set(SOURCES OneDalAlgorithms.cpp)

add_library(OneDalNative SHARED ${SOURCES} ${RESOURCES})
install_library_and_symbols(OneDalNative)
