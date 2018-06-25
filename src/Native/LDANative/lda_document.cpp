// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include "lda_document.h"

namespace lda
{
    LDADocument::LDADocument(int32_t* memory_begin, int32_t* memory_end) :
        memory_begin_(memory_begin), memory_end_(memory_end), cursor_(*memory_begin) {}

    // should be called when sweeped over all the tokens in a document
    void LDADocument::ResetCursor()
    {
        cursor_ = 0;
    }
    void LDADocument::GetDocTopicCounter(lda::light_hash_map& doc_topic_counter)
    {
        int32_t* p = memory_begin_ + 2;
        int32_t num = 0;
        while (p < memory_end_)
        {
            doc_topic_counter.inc(*p, 1);
            ++p; ++p;
            if (++num == 512)
                return;
        }
    }
}