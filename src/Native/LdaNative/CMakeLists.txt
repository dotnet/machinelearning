project (LdaNative)

set(SOURCES
    alias_multinomial_rng_int.cpp
    data_block.cpp
    hybrid_alias_map.cpp
    hybrid_map.cpp
    lda_document.cpp
    lda_engine.cpp
    lda_engine_export.cpp
    light_doc_sampler.cpp
    light_hash_map.cpp
    model_block.cpp
    utils.cpp
)

add_library(LdaNative SHARED ${SOURCES} ${RESOURCES})

if (MSVC AND NOT MSVC_VERSION LESS 1900)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /guard:cf")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qspectre")
endif()

if (${ARCHITECTURE} STREQUAL "arm")
    if (WIN32)
        target_link_libraries(LdaNative shell32.lib)
    else()
        target_link_libraries(LdaNative atomic.a)
    endif()
endif()

install_library_and_symbols (LdaNative)
