project (MatrixFactorizationNative)

include_directories(libmf)

set(SOURCES
    UnmanagedMemory.cpp
    libmf/mf.cpp
)

if(NOT WIN32)
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_library(MatrixFactorizationNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (MatrixFactorizationNative)
