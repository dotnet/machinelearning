project (CpuMathNative)

set(SOURCES
    Sse.cpp
    Avx.cpp
    MathLinux.S
)

if(WIN32)
    set_property(SOURCE Avx.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " /arch:AVX")
else()
    set_property(SOURCE Sse.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -msse3")
    set_property(SOURCE Avx.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx")
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(CMAKE_MACOSX_RPATH 1)
endif()

add_library(CpuMathNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (CpuMathNative)
