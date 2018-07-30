project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

if(WIN32)
    find_library(MKL_LIBRARY Microsoft.ML.MklImports HINTS ${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.1/runtimes/win-x64/native)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        #set(CMAKE_MACOSX_RPATH 1)
        message("Linking SymSgdNative with MKL on macOS.")
        find_library(MKL_LIBRARY Microsoft.ML.MklImports.dylib HINTS "${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.1/runtimes/osx-x64/native")
    else()
        message("Linking SymSgdNative with MKL on linux.")
        find_library(MKL_LIBRARY Microsoft.ML.MklImports.so HINTS ${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.1/runtimes/linux-x64/native)
        message("${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.1/runtimes/linux-x64/native")
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY})

install_library_and_symbols (SymSgdNative)