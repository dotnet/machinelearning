project (FastTreeNative)

set(SOURCES
    ExpandFloatType.cpp
    FastTreeNative.cpp
    getderivatives.cpp
	getsurplusderivatives.cpp
	segment.cpp
	stdafx.cpp
)

if(WIN32)
else()
    set_property(SOURCE segment.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -msse4.1")
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(CMAKE_MACOSX_RPATH 1)
endif()

add_library(FastTreeNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (FastTreeNative)
