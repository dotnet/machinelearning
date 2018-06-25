// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once

#include <string>
#include <algorithm>
#include <memory>
#include "light_hash_map.h"

namespace lda
{
    class LDADocument
    {
    public:
        const int32_t kMaxSizeLightHash = 512; // This is for the easy use of LightHashMap

        LDADocument(int32_t* memory_begin, int32_t* memory_end);
        
        inline int32_t size() const;
        inline int32_t& get_cursor();
        inline int32_t Word(int32_t index) const;
        inline int32_t Topic(int32_t index) const;
        inline void SetTopic(int32_t index, int32_t topic);

        // should be called when sweeped over all the tokens in a document
        void ResetCursor();
        void GetDocTopicCounter(lda::light_hash_map& doc_topic_counter);

    private:
        LDADocument(const LDADocument &other) = delete;
        LDADocument& operator=(const LDADocument &other) = delete;

        int32_t* memory_begin_;
        int32_t* memory_end_;
        int32_t& cursor_; // cursor_ is reference of *memory_begin_
    };

    inline int32_t LDADocument::size() const
    {
        return (std::min)(static_cast<int32_t>((memory_end_ - memory_begin_) / 2), kMaxSizeLightHash);
    }
    inline int32_t& LDADocument::get_cursor()
    {
        return cursor_;
    }
    inline int32_t LDADocument::Word(int32_t index) const
    {
        return *(memory_begin_ + 1 + index * 2);
    }
    inline int32_t LDADocument::Topic(int32_t index) const
    {
        return *(memory_begin_ + 2 + index * 2);
    }
    inline void LDADocument::SetTopic(int32_t index, int32_t topic)
    {
        *(memory_begin_ + 2 + index * 2) = topic;
    }
}