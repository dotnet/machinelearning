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
    include_directories("/usr/local/opt/libomp/include")
    link_directories("/usr/local/opt/libomp/lib")

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

find_library(MKL_LIBRARY MklImports HINTS ${MKL_LIB_PATH})

add_definitions(-DUSE_OMP)
add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY} PUBLIC ${OPENMP_LIBRARY})

if(APPLE)
    set_target_properties(SymSgdNative PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/${MKL_LIB_RPATH}}")
endif()

install_library_and_symbols (SymSgdNative)
