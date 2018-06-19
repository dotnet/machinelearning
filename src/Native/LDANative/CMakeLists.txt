project (LDANative)

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

if(WIN32)
else()
    set_property(SOURCE segment.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -msse4.1")
    list(APPEND SOURCES ${VERSION_FILE_PATH})
endif()

add_library(LDANative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (LDANative)
