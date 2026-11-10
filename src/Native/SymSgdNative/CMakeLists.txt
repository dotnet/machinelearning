project (SymSgdNative)


set(SOURCES
    SymSgdNative.cpp
)

if(APPLE)
    # CMake has support for OpenMP, however, Apple has a version of Clang
    # that does not support openMP out of the box. Therefore
    # these commands are added to sepcifically handle the Apple Clang scenario
    # If the LLVM version of clang is used for Apple builds, this can be removed
    # and the else condition can be used instead.
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    SET(OPENMP_LIBRARY "omp")
    # Apple silicon and Intel macs store brew in different locations, this finds it no matter where it is.
    execute_process(
        COMMAND brew --prefix libomp
        RESULT_VARIABLE BREW_LIBOMP
        OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT BREW_LIBOMP EQUAL 0)
        message(FATAL_ERROR "Homebrew libomp is required to build SymSgdNative on macOS.")
    endif()

    set(OPENMP_ARCHITECTURE ${ARCHITECTURE})
    if(OPENMP_ARCHITECTURE STREQUAL "x64")
        set(OPENMP_ARCHITECTURE "x86_64")
    endif()
    execute_process(
        COMMAND lipo -archs "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib"
        RESULT_VARIABLE LIBOMP_ARCH_RESULT
        OUTPUT_VARIABLE LIBOMP_ARCHITECTURES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(FIND "${LIBOMP_ARCHITECTURES}" "${OPENMP_ARCHITECTURE}" LIBOMP_ARCH_INDEX)
    if(NOT LIBOMP_ARCH_RESULT EQUAL 0 OR LIBOMP_ARCH_INDEX EQUAL -1)
        message(FATAL_ERROR
            "libomp at ${BREW_LIBOMP_PREFIX} does not contain ${OPENMP_ARCHITECTURE}; "
            "install a matching libomp or build SymSgdNative on a ${OPENMP_ARCHITECTURE} host.")
    endif()

    include_directories("${BREW_LIBOMP_PREFIX}/include")
    link_directories("${BREW_LIBOMP_PREFIX}/lib")

    list(APPEND SOURCES ${VERSION_FILE_PATH})
else()
    find_package(OpenMP)
    if (OPENMP_FOUND)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()

    if (NOT WIN32)
        list(APPEND SOURCES ${VERSION_FILE_PATH})
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        SET(CMAKE_INSTALL_RPATH "$ORIGIN/")
    endif()
endif()

if(${ARCHITECTURE} MATCHES "arm.*")
    # On ARM, Intel MKL is unavailable. Compile the minimal, self-contained CBLAS shim
    # (the four level-1 routines SymSGD needs, implemented as plain C loops with no external
    # BLAS dependency) directly into SymSgdNative. We deliberately do NOT build or ship a
    # separate libMklImports on ARM, so components that require the full MKL (LAPACK/DFTI)
    # continue to correctly report it as unavailable there.
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../MklImportsArm/MklImportsArm.c)
    set(MKL_LIBRARY "")
else()
    find_library(MKL_LIBRARY MklImports HINTS ${MKL_LIB_PATH})
endif()

add_definitions(-DUSE_OMP)
add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
if (MSVC AND NOT MSVC_VERSION LESS 1900)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /guard:cf")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Qspectre")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /guard:cf")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qspectre")
endif()
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY} PUBLIC ${OPENMP_LIBRARY})

if(APPLE)
    set_target_properties(SymSgdNative PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/${MKL_LIB_RPATH}}")
endif()

install_library_and_symbols (SymSgdNative)
