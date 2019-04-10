project (MatrixFactorizationNative)
add_definitions(-D_SCL_SECURE_NO_WARNINGS)
add_definitions(-DUSESSE)
add_definitions(-DUSEOMP)

find_package(OPENMP REQUIRED)


include_directories(libmf)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(SOURCES UnmanagedMemory.cpp libmf/mf.cpp)

if(NOT WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_library(MatrixFactorizationNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (MatrixFactorizationNative)
