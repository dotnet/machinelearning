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

if (${ARCHITECTURE} STREQUAL "arm")
    if (WIN32)
        target_link_libraries(LdaNative shell32.lib)
    else()
        target_link_libraries(LdaNative atomic.a)
    endif()
    
    # Arcade is adding mthumbs and it sets floats to 16 bits. Need to remove it.
    get_target_property(MAIN_FLAGS LdaNative COMPILE_OPTIONS)
    list(REMOVE_ITEM MAIN_FLAGS "-mthumb")
        
    set_property(TARGET LdaNative PROPERTY COMPILE_OPTIONS ${MAIN_FLAGS})
endif()

install_library_and_symbols (LdaNative)
