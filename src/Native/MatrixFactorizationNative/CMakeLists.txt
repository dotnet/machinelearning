project (MatrixFactorizationNative)
add_definitions(-D_SCL_SECURE_NO_WARNINGS)

include_directories(libmf)

if(APPLE)
    # Disable OPENMP and SSE
else()
    add_definitions(-DUSEOMP)
    add_definitions(-DUSESSE)
    if(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -pthread -std=c++0x -march=native")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /nologo /O2 /EHsc /D \"_CRT_SECURE_NO_DEPRECATE\" /openmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

set(SOURCES UnmanagedMemory.cpp libmf/mf.cpp)

if(NOT WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_library(MatrixFactorizationNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (MatrixFactorizationNative)
