project (OneDALNative)

if(NOT ${ARCHITECTURE} MATCHES "arm.*")
    add_definitions(-DUSESSE)
endif()

# if(UNIX)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")

#     if(NOT ${ARCHITECTURE} MATCHES "arm.*")
#         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64")
#     endif()

#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
#     if (APPLE)
#         # Apple silicon and Intel macs store brew in different locations, this finds it no matter where it is.
#         execute_process(
#             COMMAND brew --prefix libomp
#             RESULT_VARIABLE BREW_LIBOMP
#             OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
#             OUTPUT_STRIP_TRAILING_WHITESPACE
#         )

#         include_directories("${BREW_LIBOMP_PREFIX}/include")
#         link_directories("${BREW_LIBOMP_PREFIX}/lib")
#     endif()
# endif()

# if(WIN32)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /nologo /O2 /EHsc /D \"_CRT_SECURE_NO_DEPRECATE\" /openmp")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

find_package(oneDAL CONFIG REQUIRED)

set(SOURCES DALSimpleRegression.cpp)

add_library(OneDALNative SHARED ${SOURCES} ${RESOURCES})
install_library_and_symbols (OneDALNative)
