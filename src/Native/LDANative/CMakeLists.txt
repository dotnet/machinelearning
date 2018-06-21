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

install_library_and_symbols (LdaNative)
