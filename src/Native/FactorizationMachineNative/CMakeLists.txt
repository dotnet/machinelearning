project (FactorizationMachineNative)

set(SOURCES
    FactorizationMachineCore.cpp
)

if(WIN32)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(CMAKE_MACOSX_RPATH 1)
endif()

add_library(FactorizationMachineNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (FactorizationMachineNative)