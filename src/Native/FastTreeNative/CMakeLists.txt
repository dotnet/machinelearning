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

add_library(FastTreeNative SHARED ${SOURCES} ${RESOURCES})

if (MSVC AND NOT MSVC_VERSION LESS 1900)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /guard:cf")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qspectre")
endif()

install_library_and_symbols (FastTreeNative)
