project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

if(WIN32)
    find_library(MKL_LIBRARY MklImports HINTS ${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.5/runtimes/win-x64/native)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        message("Linking SymSgdNative with MKL on macOS.")
        find_library(MKL_LIBRARY libMklImports.dylib HINTS "${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.5/runtimes/osx-x64/native")
    else()
        message("Linking SymSgdNative with MKL on linux.")
        find_library(MKL_LIBRARY libMklImports.so HINTS ${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.5/runtimes/linux-x64/native)
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY})
        
install_library_and_symbols (SymSgdNative)