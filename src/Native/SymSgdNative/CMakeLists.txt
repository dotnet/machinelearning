project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

find_library(MKL_LIBRARY MklImports HINTS ${MKL_LIB_PATH})
if(!WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        message("executing commands for macOS.")
    else()
        message("executing commands for linux.")
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        SET(CMAKE_INSTALL_RPATH "$ORIGIN/")
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY})

install_library_and_symbols (SymSgdNative)