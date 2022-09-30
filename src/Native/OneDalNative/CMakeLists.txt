project (OneDalNative)

if(NOT ${ARCHITECTURE} MATCHES "arm.*")
    add_definitions(-DUSESSE)
endif()

set(USE_DPCPP no)
set(TARGET_LINK dynamic)
set(USE_PARALLEL yes)

set(DALROOT "${ONEDAL_LIB_PATH}/daal")

list(APPEND CMAKE_PREFIX_PATH "${ONEDAL_LIB_PATH}/daal")
list(APPEND CMAKE_PREFIX_PATH "${ONEDAL_LIB_PATH}/tbb")

find_package(oneDAL CONFIG REQUIRED onedal_core onedal_thread)

include_directories(${oneDAL_INCLUDE_DIRS})
link_libraries(${oneDAL_IMPORTED_TARGETS})

set(SOURCES OneDalAlgorithms.cpp)

add_library(OneDalNative SHARED ${SOURCES} ${RESOURCES})
install_library_and_symbols(OneDalNative)
