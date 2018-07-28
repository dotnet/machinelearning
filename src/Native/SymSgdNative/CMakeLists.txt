project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

if(WIN32)
    link_directories(${CMAKE_SOURCE_DIR}/../../packages/MlNetMklDeps/0.0.0.1/runtimes/win-x64/native)
else()
    
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    
if(APPLE)
        link_directories(${CMAKE_SOURCE_DIR}/../../packages/MlNetMklDeps/0.0.0.1/runtimes/osx-x64/native)
    else()
        link_directories(${CMAKE_SOURCE_DIR}/../../packages/MlNetMklDeps/0.0.0.1/runtimes/linux-x64/native)
    endif()
endif()

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
llvm_map_components_to_libnames(llvm_libs support core irreader )
include_directories( ${LLVM_INCLUDE_DIRS})
set(CMAKE_C_COMPILER   /usr/local/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative Microsoft.ML.MklImports)
target_link_libraries(SymSgdNative ${llvm_libs} /usr/local/opt/llvm/lib/libomp.dylib )

install_library_and_symbols (SymSgdNative)