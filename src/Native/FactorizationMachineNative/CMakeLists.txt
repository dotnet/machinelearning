project (FactorizationMachineNative)

set(SOURCES
    FactorizationMachineCore.cpp
)

if(WIN32)
else()
    set_property(SOURCE FactorizationMachineCore.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -msse3")
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_library(FactorizationMachineNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (FactorizationMachineNative)