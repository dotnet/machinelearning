project (FactorizationMachineNative)

set(SOURCES
    FactorizationMachineCoreSSE.cpp
    FactorizationMachineCoreAVX.cpp
    FactorizationMachineCoreFMA.cpp
)

if(WIN32)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    set_property(SOURCE FactorizationMachineCoreAVX.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx")
    set_property(SOURCE FactorizationMachineCoreFMA.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -mfma")
endif()

add_library(FactorizationMachineNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (FactorizationMachineNative)