project (CpuMathNative)

set(SOURCES
    Sse.cpp
    MathLinux.S
    FactorizationMachineCore.cpp
)

if(NOT WIN32)
    set_property(SOURCE Sse.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -msse3")
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_library(CpuMathNative SHARED ${SOURCES} ${RESOURCES})

if (MSVC AND NOT MSVC_VERSION LESS 1900)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /guard:cf")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qspectre")
endif()

install_library_and_symbols (CpuMathNative)
