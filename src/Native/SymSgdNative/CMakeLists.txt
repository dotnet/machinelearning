project (SymSgdNative)
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes")


set(SOURCES
    SymSgdNative.cpp
)

if(WIN32)
    find_library(MKL_LIBRARY libMklImports HINTS ${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes/win-x64/native)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        message("Linking SymSgdNative with MKL on macOS.")
        find_library(MKL_LIBRARY libMklImports.dylib HINTS "${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes/osx-x64/native")
    else()
        message("Linking SymSgdNative with MKL on linux.")
        find_library(MKL_LIBRARY libMklImports.so HINTS ${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes/linux-x64/native)
    endif()
endif()

add_library(SymSgdNative SHARED ${SOURCES} ${RESOURCES})
target_link_libraries(SymSgdNative PUBLIC ${MKL_LIBRARY})

#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
#if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
#    set_target_properties(SymSgdNative PROPERTIES INSTALL_RPATH "${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes/osx-x64/native")
#else()
#    set_target_properties(SymSgdNative PROPERTIES INSTALL_RPATH "${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes/linux-x64/native;${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes/win-x64/native")
#endif()        
#set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_SOURCE_DIR}/../../packages/mlnetmkldeps/0.0.0.4/runtimes")
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

install_library_and_symbols (SymSgdNative)