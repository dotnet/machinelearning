project (CpuMathNative)

set(SOURCES
    Sse.cpp
    MathLinux.S
)

if(WIN32)
    set_property(SOURCE Sse.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx")
else()
    set_property(SOURCE Sse.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -msse3")
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_library(CpuMathNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (CpuMathNative)
