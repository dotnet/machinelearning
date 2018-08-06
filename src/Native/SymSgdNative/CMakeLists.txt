project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

SET(MKL_LIB_ROOT ${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/${MKL_VERSION}/runtimes)

if(WIN32)
    find_library(MKL_LIBRARY MklImports HINTS ${MKL_LIB_ROOT}/win-x64/native)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        message("Linking SymSgdNative with MKL on macOS.")
        find_library(MKL_LIBRARY libMklImports.dylib HINTS "${MKL_LIB_ROOT}/osx-x64/native")
    else()
        message("Linking SymSgdNative with MKL on linux.")
        find_library(MKL_LIBRARY libMklImports.so HINTS ${MKL_LIB_ROOT}/linux-x64/native)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        SET(CMAKE_INSTALL_RPATH "$ORIGIN/")
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY})

install_library_and_symbols (SymSgdNative)