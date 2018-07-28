project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

if(WIN32)
    link_directories(${CMAKE_SOURCE_DIR}/../../packages/MlNetMklDeps/0.0.0.1/runtimes/win-x64/native)
else()
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
    
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    
    if(APPLE)
        link_directories(${CMAKE_SOURCE_DIR}/../../packages/MlNetMklDeps/0.0.0.1/runtimes/osx-x64/native)
    else()
        link_directories(${CMAKE_SOURCE_DIR}/../../packages/MlNetMklDeps/0.0.0.1/runtimes/linux-x64/native)
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative Microsoft.ML.MklImports)

install_library_and_symbols (SymSgdNative)