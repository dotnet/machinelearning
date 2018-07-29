project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

if(WIN32)
    link_directories(${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.1/runtimes/win-x64/native)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    
    if(APPLE)
        link_directories(${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.1/runtimes/osx-x64/native)
    else()
        link_directories(${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.1/runtimes/linux-x64/native)
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative Microsoft.ML.MklImports)

install_library_and_symbols (SymSgdNative)