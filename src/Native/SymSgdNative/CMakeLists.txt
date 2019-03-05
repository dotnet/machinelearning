project (SymSgdNative)


set(SOURCES
    SymSgdNative.cpp
)

find_package(OpenMP)
if (OPENMP_FOUND)
    if(APPLE)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
        SET(OPENMP_LIBRARY "omp")
        include_directories("/usr/local/opt/libomp/include")
        link_directories("/usr/local/opt/libomp/lib")
    else()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

find_library(MKL_LIBRARY MklImports HINTS ${MKL_LIB_PATH})

if(NOT WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(NOT APPLE)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        SET(CMAKE_INSTALL_RPATH "$ORIGIN/")
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY} PUBLIC ${OPENMP_LIBRARY})

if(APPLE)
    set_target_properties(SymSgdNative PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/${MKL_LIB_RPATH}}")
endif()

install_library_and_symbols (SymSgdNative)
