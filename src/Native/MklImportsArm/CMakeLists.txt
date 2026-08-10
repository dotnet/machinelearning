project(MklImportsArm)

# On ARM platforms, Intel MKL is not available. This target provides a small,
# self-contained libMklImports covering exactly the symbols SymSGD needs
# (dense/sparse level-1 CBLAS) plus DFTI stubs, with NO external BLAS
# dependency. This is required because the CI cross-compilation sysroots do
# not ship OpenBLAS or any system BLAS.

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

install_library_and_symbols(MklImports)
