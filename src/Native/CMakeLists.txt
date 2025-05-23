cmake_minimum_required(VERSION 3.20)

project(Native)

if (POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # MACOSX_RPATH enabled by default
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX $ENV{__CMakeBinDir})
set(RESOURCES)

# Include 'bin/obj' dir since it contains _version.h
include_directories("$ENV{__IntermediatesDir}")

if(WIN32)
    add_definitions(-DWIN32)
    add_definitions(-D_WIN32=1)
    add_definitions(-DUNICODE -D_UNICODE)
    if(IS_64BIT_BUILD)
        add_definitions(-D_WIN64=1)
    endif()
    add_compile_options($<$<CONFIG:Debug>:-DDEBUG>)
    add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)
    add_compile_options($<$<CONFIG:RelWithDebInfo>:-DNDEBUG>)
    add_compile_options($<$<CONFIG:Debug>:/Od>)
    add_compile_options($<$<CONFIG:Debug>:/MTd>) # /MT will static link the VC runtime library, so it doesn't need to be installed on the target machine
    add_compile_options($<$<CONFIG:Release>:/MT>)
    add_compile_options($<$<CONFIG:RelWithDebInfo>:/MT>)
    add_compile_options(/guard:cf)
    add_compile_options(/Zo) # make optimized builds debugging easier. /Zo is the newer documented flag.
    add_compile_options(/nologo) # Suppress Startup Banner
    add_compile_options(/W3) # set warning level to 3
    add_compile_options(/WX) # treat warnings as errors
    add_compile_options(/Oi) # enable intrinsics
    add_compile_options(/Oy-) # disable suppressing of the creation of frame pointers on the call stack for quicker function calls
    add_compile_options(/GF) # enable read-only string pooling
    add_compile_options(/FC) # use full pathnames in diagnostics
    add_compile_options(/DEBUG)
    add_compile_options(/GS)
    add_compile_options(/Zc:inline)
    add_compile_options(/fp:precise)
    add_compile_options(/EHsc)

    # From here below are warnings required to be explicitly enabled.
    add_compile_options(/w34242)
    add_compile_options(/w34244)
    add_compile_options(/w34510)
    add_compile_options(/w34610)
    add_compile_options(/w34611)
    add_compile_options(/w34700)
    add_compile_options(/w34701)
    add_compile_options(/w34703)

    if ($ENV{__BuildArch} STREQUAL "x86")
        add_compile_options(/Gz)
    endif ()

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LARGEADDRESSAWARE") # can handle addresses larger than 2 gigabytes
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /INCREMENTAL:NO")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /PDBCOMPRESS")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:1572864")

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /guard:cf")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /guard:cf")

    # Debug build specific flags
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "/NOVCFEATURE")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /NODEFAULTLIB:vcompd.lib /DEFAULTLIB:vcomp.lib")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /NODEFAULTLIB:vcompd.lib /DEFAULTLIB:vcomp.lib")

    # Release build specific flags
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /NODEFAULTLIB:libucrt.lib /DEFAULTLIB:ucrt.lib")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /NODEFAULTLIB:libucrt.lib /DEFAULTLIB:ucrt.lib")

    # RelWithDebInfo specific flags
    set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /NODEFAULTLIB:libucrt.lib /DEFAULTLIB:ucrt.lib")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /NODEFAULTLIB:libucrt.lib /DEFAULTLIB:ucrt.lib")
    list(APPEND RESOURCES $ENV{__IntermediatesDir}/NativeVersion.rc)
else()
    add_compile_options(-Wno-unused-local-typedef)
    add_compile_options(-fPIC)
    add_compile_options(-fvisibility=hidden)
    add_definitions(-Werror) # treat warnings as errors
    add_compile_options(-g) # generate debug information

    # On Unix CMAKE_BUILD_TYPE is not passed in as just Debug/Release, so manually adding the extra flags
    if(${CMAKE_BUILD_TYPE} MATCHES "Release*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    endif()

endif()

# Get the version of the compiler that is in the file name for tool location.
set (CLR_CMAKE_COMPILER_FILE_NAME_VERSION "")
if (CMAKE_C_COMPILER MATCHES "-?[0-9]+(\\.[0-9]+)?$")
  set(CLR_CMAKE_COMPILER_FILE_NAME_VERSION "${CMAKE_MATCH_0}")
endif()

# Set the architecture we are compiling for on APPLE. This lets you cross target from x86_64 -> arm64.
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES ${ARCHITECTURE})
endif()

# Older CMake doesn't support CMAKE_CXX_STANDARD and GCC/Clang need a switch to enable C++ 11
if(${CMAKE_CXX_COMPILER_ID} MATCHES "(Clang|GNU)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# This is required to map a symbol reference to a matching definition local to the module (.so)
# containing the reference instead of using definitions from other modules.
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Xlinker -Bsymbolic -Bsymbolic-functions")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=sha1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--build-id=sha1")
    add_compile_options(-fstack-protector-strong)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_compile_options(-fstack-protector)
endif()

add_definitions(-D_NO_ASYNCRTIMP)
add_definitions(-D_NO_PPLXIMP)
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    add_definitions(-D__LINUX__)
endif()


if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    set(CLR_CMAKE_PLATFORM_UNIX 1)
endif(CMAKE_SYSTEM_NAME STREQUAL Linux)

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(CLR_CMAKE_PLATFORM_UNIX 1)
endif(CMAKE_SYSTEM_NAME STREQUAL Darwin)

if(CMAKE_SYSTEM_NAME STREQUAL FreeBSD)
    set(CLR_CMAKE_PLATFORM_UNIX 1)
    add_definitions(-D_BSD_SOURCE) # required for getline
endif(CMAKE_SYSTEM_NAME STREQUAL FreeBSD)

if(CMAKE_SYSTEM_NAME STREQUAL OpenBSD)
    set(CLR_CMAKE_PLATFORM_UNIX 1)
endif(CMAKE_SYSTEM_NAME STREQUAL OpenBSD)

if(CMAKE_SYSTEM_NAME STREQUAL NetBSD)
    set(CLR_CMAKE_PLATFORM_UNIX 1)
endif(CMAKE_SYSTEM_NAME STREQUAL NetBSD)

if(CMAKE_SYSTEM_NAME STREQUAL SunOS)
    set(CLR_CMAKE_PLATFORM_UNIX 1)
endif(CMAKE_SYSTEM_NAME STREQUAL SunOS)

if(CLR_CMAKE_PLATFORM_UNIX)
    if(STRIP_SYMBOLS)
        find_program(STRIP strip)

        if(CMAKE_SYSTEM_NAME STREQUAL Linux AND DEFINED CMAKE_TOOLCHAIN_FILE)
            find_program(OBJCOPY
                NAMES llvm-objcopy
                llvm-objcopy${CLR_CMAKE_COMPILER_FILE_NAME_VERSION})
        else(CMAKE_SYSTEM_NAME STREQUAL Linux AND DEFINED CMAKE_TOOLCHAIN_FILE)
            if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
                find_program(DSYMUTIL dsymutil)
            else(CMAKE_SYSTEM_NAME STREQUAL Darwin)
                find_program(OBJCOPY objcopy)
            endif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        endif(CMAKE_SYSTEM_NAME STREQUAL Linux AND DEFINED CMAKE_TOOLCHAIN_FILE)
    endif(STRIP_SYMBOLS)
endif(CLR_CMAKE_PLATFORM_UNIX)

function(strip_symbols targetName outputFilename)
    if(CLR_CMAKE_PLATFORM_UNIX)
        if(STRIP_SYMBOLS)
            # On the older version of cmake (2.8.12) used on Ubuntu 14.04 the TARGET_FILE
            # generator expression doesn't work correctly returning the wrong path and on
            # the newer cmake versions the LOCATION property isn't supported anymore.
            if(CMAKE_VERSION VERSION_EQUAL 3.0 OR CMAKE_VERSION VERSION_GREATER 3.0)
                set(strip_source_file $<TARGET_FILE:${targetName}>)
            else()
                get_property(strip_source_file TARGET ${targetName} PROPERTY LOCATION)
            endif()

            if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
                set(strip_destination_file ${strip_source_file}.dwarf)

                execute_process(
                    COMMAND ${DSYMUTIL} --help
                    OUTPUT_VARIABLE DSYMUTIL_HELP_OUTPUT
                )

                set(DSYMUTIL_OPTS "--flat")
                if ("${DSYMUTIL_HELP_OUTPUT}" MATCHES "--minimize")
                    list(APPEND DSYMUTIL_OPTS "--minimize")
                endif ()

                add_custom_command(
                    TARGET ${targetName}
                    POST_BUILD
                    VERBATIM
                    COMMAND ${DSYMUTIL} ${DSYMUTIL_OPTS} ${strip_source_file}
                    COMMAND ${STRIP} -u -r ${strip_source_file}
                    COMMENT Stripping symbols from ${strip_source_file} into file ${strip_destination_file}
                )
            else(CMAKE_SYSTEM_NAME STREQUAL Darwin)
                set(strip_destination_file ${strip_source_file}.dbg)

                add_custom_command(
                    TARGET ${targetName}
                    POST_BUILD
                    VERBATIM
                    COMMAND ${OBJCOPY} --only-keep-debug ${strip_source_file} ${strip_destination_file}
                    COMMAND ${OBJCOPY} --strip-unneeded ${strip_source_file}
                    COMMAND ${OBJCOPY} --add-gnu-debuglink=${strip_destination_file} ${strip_source_file}
                    COMMENT Stripping symbols from ${strip_source_file} into file ${strip_destination_file}
                )
            endif(CMAKE_SYSTEM_NAME STREQUAL Darwin)

            set(${outputFilename} ${strip_destination_file} PARENT_SCOPE)
        endif(STRIP_SYMBOLS)
    endif(CLR_CMAKE_PLATFORM_UNIX)
endfunction()

function(install_library_and_symbols targetName)
    strip_symbols(${targetName} strip_destination_file)

    install (TARGETS ${targetName} DESTINATION .)

    if(WIN32)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${targetName}.pdb DESTINATION .)
    else()
        install(FILES ${strip_destination_file} DESTINATION .)
    endif()
endfunction()

if (${ARCHITECTURE} STREQUAL "arm")
    add_compile_options(-mthumb)
    add_compile_options(-fPIC)
    add_compile_options(-march=armv7-a)

    if (NOT DEFINED CLR_ARM_FPU_TYPE)
        set (CLR_ARM_FPU_TYPE vfpv3)
    endif (NOT DEFINED CLR_ARM_FPU_TYPE)

    add_compile_options (-mfpu=${CLR_ARM_FPU_TYPE})
    if (NOT DEFINED CLR_ARM_FPU_CAPABILITY)
        set (CLR_ARM_FPU_CAPABILITY 0x7)
    endif (NOT DEFINED CLR_ARM_FPU_CAPABILITY)
endif()

if(NOT ${ARCHITECTURE} MATCHES "arm.*")
    add_subdirectory(CpuMathNative)
    add_subdirectory(FastTreeNative)
    add_subdirectory(MklProxyNative)
    # TODO: once we fix the 4 intel MKL methods, SymSgdNative will need to go back in.
    add_subdirectory(SymSgdNative)
  endif()

if(${ARCHITECTURE} MATCHES "[xX].*64")
  add_subdirectory(OneDalNative)
else()
  if (DEFINED ONEDAL_DEVEL_PATH)
    message("Path to OneDal library defined [${ONEDAL_DEVEL_PATH}] but it is not being built on this arch.")
  endif()
  if (DEFINED MKL_LIB_PATH)
    message("Path to Intel MKL library defined [${MKL_LIB_PATH}] but it is not being built on this arch.")
  endif()
endif()

add_subdirectory(LdaNative)
add_subdirectory(MatrixFactorizationNative)
