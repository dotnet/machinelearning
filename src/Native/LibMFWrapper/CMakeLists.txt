project (FactorizationMachineNative)

set(SOURCES
    UnmanagedMemory.cpp
    mf.cpp
)

if(WIN32)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_definitions(-DUSESSE)
add_library(LibMFWrapper SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (LibMFWrapper)
