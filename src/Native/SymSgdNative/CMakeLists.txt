project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

file(TO_NATIVE_PATH "${MKL_LIB_PATH}" MKL_LIB_PATH)
find_library(MKL_LIBRARY MklImports HINTS ${MKL_LIB_PATH})
if(CMAKE_SYSTEM_NAME STREQUAL Linux OR CMAKE_SYSTEM_NAME STREQUAL Darwin)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(CMAKE_SYSTEM_NAME STREQUAL Linux)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        SET(CMAKE_INSTALL_RPATH "$ORIGIN/")
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY})

install_library_and_symbols (SymSgdNative)