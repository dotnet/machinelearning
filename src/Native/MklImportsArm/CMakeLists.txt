project(MklImportsArm)

# On ARM platforms, Intel MKL is not available. This target provides
# a compatible libMklImports.so backed by the system BLAS (typically
# OpenBLAS) with stubs for MKL-specific sparse BLAS and FFT functions.

find_package(BLAS REQUIRED)

set(SOURCES
    MklImportsArm.c
)

if(NOT WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    SET(CMAKE_INSTALL_RPATH "$ORIGIN/")
endif()

add_library(MklImports SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(MklImports PUBLIC ${BLAS_LIBRARIES})

install_library_and_symbols(MklImports)
