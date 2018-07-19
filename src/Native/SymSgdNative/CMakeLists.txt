project (SymSgdNative)

set(SOURCES
    SymSgdNative.cpp
)

if(WIN32)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (SymSgdNative)
