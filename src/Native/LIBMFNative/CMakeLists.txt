project (LIBMFNative)

include_directories(libmf)

set(SOURCES
    UnmanagedMemory.cpp
    libmf/mf.cpp
)

if(NOT WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

if(APPLE)
    include_directories(/usr/include/malloc) # for malloc.h
endif()

add_library(LIBMFNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (LIBMFNative)
