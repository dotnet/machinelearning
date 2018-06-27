// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


#include <fstream>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include <stdlib.h>
#include "model_block.h"
#include "lda_document.h"

namespace lda
{
    int64_t upper_bound(int64_t x)
    {
        if (x == 0)
        {
            return 0;
        }
        int64_t shift = 0;
        int64_t y = 1;
        x--;
        while (x)
        {
            x = x >> 1;
            y = y << 1;
            ++shift;
        }
        return y;
    }

    int32_t align64(int32_t size)
    {
        if (size % 64 == 0)
        {
            return size;
        }
        else
        {
            size = 64 * (size / 64) + 64;
            return size;
        }
    }


    LDAModelBlock::LDAModelBlock()
        : dict_(nullptr),
        num_vocabs_(0),
        mem_block_size_(0),
        mem_block_(nullptr),
        alias_mem_block_size_(0),
        alias_mem_block_(nullptr)
    {
    }
    LDAModelBlock::~LDAModelBlock()
    {
        Clear();
    }

    void LDAModelBlock::Clear()
    {
        if (dict_)
        {
            delete[]dict_;
            dict_ = nullptr;
        }
        if (mem_block_)
        {
            delete[]mem_block_;
            mem_block_ = nullptr;
        }
        if (alias_mem_block_)
        {
            delete[]alias_mem_block_;
            alias_mem_block_ = nullptr;
        }

        num_vocabs_ = -1;
        num_topics_ = -1;

        mem_block_size_ = 0;
        alias_mem_block_size_ = 0;
    }

    void LDAModelBlock::Init(int32_t num_vocabs, int32_t num_topics, int64_t nonzero_num)
    {
        num_vocabs_ = num_vocabs;
        num_topics_ = num_topics;

        dict_ = new WordEntry[num_vocabs_];
        for (int i = 0; i < num_vocabs_; ++i)
        {
            // This warning is a false positive. Supressing it similar to the existing one on Line 140 below.
#pragma warning(suppress: 6386)
            dict_[i].is_dense_ = 0;
            dict_[i].is_alias_dense_ = 0;
        }

        mem_block_size_ = 2 * upper_bound(load_factor_ * nonzero_num);
        alias_mem_block_size_ = nonzero_num * 3; 

        mem_block_ = new int32_t[mem_block_size_]();                // NOTE: force to initialize the values to be zero
        alias_mem_block_ = new int32_t[alias_mem_block_size_]();    // NOTE: force to initialize the values to be zero
    }

    void LDAModelBlock::Init(int32_t num_vocabs, int32_t num_topics, int64_t mem_block_size, int64_t alias_mem_block_size)
    {
        num_vocabs_ = num_vocabs;
        num_topics_ = num_topics;

        dict_ = new WordEntry[num_vocabs_];
        for (int i = 0; i < num_vocabs_; ++i)
        {
            // This warning is a false positive. Supressing it similar to the existing one on Line 140 below.
#pragma warning(suppress: 6386)
            dict_[i].is_dense_ = 0;
            dict_[i].is_alias_dense_ = 0;
        }

        mem_block_size_ = mem_block_size;
        mem_block_ = new int32_t[mem_block_size_]();   // NOTE : force to initialize the values to be zero

        alias_mem_block_size_ = alias_mem_block_size;
        alias_mem_block_ = new int32_t[alias_mem_block_size_]();    //NOTE: force to initialize the values to be zero

        std::cout << "mem_block_size = " << mem_block_size_ * 4 << std::endl;
        std::cout << "alias_mem_block_size = " << alias_mem_block_size_ * 4 << std::endl;

        offset_ = 0;
        alias_offset_ = 0;
    }

    void LDAModelBlock::Init(int32_t num_vocabs, int32_t num_topics)
    {
        num_vocabs_ = num_vocabs;
        num_topics_ = num_topics;

        dict_ = new WordEntry[num_vocabs_];
        for (int i = 0; i < num_vocabs_; ++i)
        {
            // This warning is a false positive caused by an old bug in PREfast. It is fixed in VS 2015.
#pragma warning(suppress: 6386) 
            dict_[i].tf = 0;
            dict_[i].is_dense_ = 0;
            dict_[i].is_alias_dense_ = 0;
        }
    }

    void LDAModelBlock::SetWordInfo(int word_id, int32_t nonzero_num, bool fullSparse)
    {
        dict_[word_id].word_id_ = word_id;
        dict_[word_id].tf = nonzero_num;

        int32_t hot_thresh;
        if (fullSparse)
        {
            // use a very large threshold to ensure every row of word-topic-table using a sparse representation
            hot_thresh = std::numeric_limits<int>::max();
        }
        else
        {
            hot_thresh = num_topics_ / (2 * load_factor_);  //hybrid
        }
        int32_t alias_hot_thresh;
        if (fullSparse)
        {
            // use a very large threshold to ensure every row of alias table using a sparse representation
            alias_hot_thresh = std::numeric_limits<int>::max();
        }
        else
        {
            alias_hot_thresh = (num_topics_ * 2) / 3;
        }

        int32_t capacity = 0;
        int32_t row_size = 0;
        int32_t alias_capacity = 0;
        int32_t alias_row_size = 0;

        if (dict_[word_id].tf >= hot_thresh)
        {
            dict_[word_id].is_dense_ = 1;
            capacity = num_topics_;
            row_size = capacity;
        }
        else if (dict_[word_id].tf > 0)
        {
            dict_[word_id].is_dense_ = 0;
            int capacity_lower_bound = load_factor_ * dict_[word_id].tf;
            capacity = (int32_t)upper_bound(capacity_lower_bound);
            row_size = capacity * 2;
        }
        else
        {
            dict_[word_id].is_dense_ = 1;
            row_size = 0;
            capacity = 0;
        }

        dict_[word_id].offset_ = offset_;
        dict_[word_id].end_offset_ = offset_ + row_size;
        dict_[word_id].capacity_ = capacity;

        offset_ += row_size;

        if (dict_[word_id].tf >= alias_hot_thresh)
        {
            alias_capacity = num_topics_;
            alias_row_size = 2 * num_topics_;
            dict_[word_id].is_alias_dense_ = 1;
        }
        else if (dict_[word_id].tf > 0)
        {
            alias_capacity = dict_[word_id].tf;
            alias_row_size = 3 * dict_[word_id].tf;
            dict_[word_id].is_alias_dense_ = 0;
        }
        else
        {
            alias_capacity = 0;
            alias_row_size = 0;
            dict_[word_id].is_alias_dense_ = 1;
        }
        dict_[word_id].alias_capacity_ = alias_capacity;
        dict_[word_id].alias_offset_ = alias_offset_;
        dict_[word_id].alias_end_offset_ = alias_offset_ + alias_row_size;

        alias_offset_ += alias_row_size;
    }

    // NOTE: sometimes, we use totally sparse representation (in testing phase), fullSparse == true
    // in other times, we use hybrid structure (in training phase), fullSparse == false
    void LDAModelBlock::InitModelBlockByTFS(bool fullSparse)
    {
        const int32_t max_tf_thresh = std::numeric_limits<int32_t>::max();
        int32_t hot_thresh;
        if (fullSparse)
        {
            // totally sparse
            // use a very large threshold to ensure every row of word-topic-table using a sparse representation
            hot_thresh = std::numeric_limits<int>::max();
        }
        else
        {
            // hybrid
            hot_thresh = num_topics_ / (2 * load_factor_);
        }
        int32_t alias_hot_thresh;
        if (fullSparse)
        {
            // use a very large threshold to ensure every row of alias table using a sparse representation
            alias_hot_thresh = std::numeric_limits<int>::max();
        }
        else
        {
            alias_hot_thresh = (num_topics_ * 2) / 3;
        }

        int32_t word_id;
        int32_t capacity = 0;
        int32_t row_size = 0;
        int32_t alias_capacity = 0;
        int32_t alias_row_size = 0;

        int64_t offset = 0;
        int64_t alias_offset = 0;

        for (word_id = 0; word_id < num_vocabs_; ++word_id)
        {
            int32_t tf = dict_[word_id].tf;

            dict_[word_id].word_id_ = word_id;
            dict_[word_id].tf = tf;

            if (tf >= hot_thresh)
            {
                dict_[word_id].is_dense_ = 1;
                capacity = num_topics_;
                row_size = capacity;
            }
            else if (tf > 0)
            {
                dict_[word_id].is_dense_ = 0;
                int capacity_lower_bound = load_factor_ * tf;
                capacity = (int32_t)upper_bound(capacity_lower_bound);
                row_size = capacity * 2;
            }
            else
            {
                dict_[word_id].is_dense_ = 1;
                capacity = 0;
                row_size = 0;
            }

            dict_[word_id].offset_ = offset;
            dict_[word_id].end_offset_ = offset + row_size;
            dict_[word_id].capacity_ = capacity;

            offset += row_size;

            if (tf >= alias_hot_thresh)
            {
                alias_capacity = num_topics_;
                alias_row_size = 2 * num_topics_;
                dict_[word_id].is_alias_dense_ = 1;
            }
            else if (tf > 0)
            {
                alias_capacity = tf;
                alias_row_size = 3 * tf;
                dict_[word_id].is_alias_dense_ = 0;
            }
            else
            {
                alias_capacity = 0;
                alias_row_size = 0;
                dict_[word_id].is_alias_dense_ = 1;
            }
            dict_[word_id].alias_capacity_ = alias_capacity;
            dict_[word_id].alias_offset_ = alias_offset;
            dict_[word_id].alias_end_offset_ = alias_offset + alias_row_size;
            alias_offset += alias_row_size;
        }

        mem_block_size_ = dict_[num_vocabs_ - 1].end_offset_;
        mem_block_ = new int32_t[mem_block_size_]();                // NOTE: force to initialize the values to be zero

        alias_mem_block_size_ = dict_[num_vocabs_ - 1].alias_end_offset_;
        alias_mem_block_ = new int32_t[alias_mem_block_size_]();    //NOTE: force to initialize the values to be zero

        std::cout << "mem_block_size = " << mem_block_size_ * 4 << std::endl;
        std::cout << "alias_mem_block_size = " << alias_mem_block_size_ * 4 << std::endl;
    }

    void LDAModelBlock::InitFromDataBlock(const LDADataBlock *data_block, int32_t num_vocabs, int32_t num_topics)
    {
        num_vocabs_ = num_vocabs;
        num_topics_ = num_topics;

        int32_t doc_num = data_block->num_documents();
        dict_ = new WordEntry[num_vocabs_];
        for (int i = 0; i < num_vocabs_; ++i)
        {
            dict_[i].tf = 0;
        }

        for (int i = 0; i < doc_num; ++i)
        {
            std::shared_ptr<LDADocument> doc = data_block->GetOneDoc(i);
            int32_t doc_size = doc->size();
            for (int j = 0; j < doc_size; ++j)
            {
                int32_t w = doc->Word(j);
                dict_[w].tf++;
            }
        }

        InitModelBlockByTFS(false);
    }
    // Count the number of nonzero values in each row
    void LDAModelBlock::CountNonZero(std::vector<int32_t> &tfs)
    {
        for (int i = 0; i < num_vocabs_; ++i)
        {
            hybrid_map row(mem_block_ + dict_[i].offset_,
                dict_[i].is_dense_,
                dict_[i].capacity_,
                0,
                nullptr);
            tfs[i] = row.nonzero_num();
        }
    }

    void LDAModelBlock::GetModelSizeByTFS(bool fullSparse, std::vector<int32_t> &tfs, int64_t &mem_block_size, int64_t &alias_mem_block_size)
    {
        const int32_t max_tf_thresh = std::numeric_limits<int32_t>::max();
        int32_t hot_thresh;
        if (fullSparse)
        {
            // totally sparse
            // use a very large threshold to ensure every row of word-topic-table using a sparse representation
            hot_thresh = std::numeric_limits<int>::max();
        }
        else
        {
            // hybrid
            hot_thresh = num_topics_ / (2 * load_factor_);
        }
        // hot_thresh = 0;  // totally dense
        int32_t alias_hot_thresh;
        if (fullSparse)
        {
            // use a very large threshold to ensure every row of alias table using a sparse representation
            alias_hot_thresh = std::numeric_limits<int>::max();
        }
        else
        {
            alias_hot_thresh = (num_topics_ * 2) / 3;
        }

        int32_t word_id;
        int32_t capacity = 0;
        int32_t alias_capacity = 0;
        int32_t row_size = 0;
        int32_t alias_row_size = 0;

        mem_block_size = 0;
        alias_mem_block_size = 0;

        for (word_id = 0; word_id < num_vocabs_; ++word_id)
        {
            int32_t tf = tfs[word_id];

            if (tf >= hot_thresh)
            {
                capacity = num_topics_;
                row_size = capacity;
            }
            else if (tf > 0)
            {
                int capacity_lower_bound = load_factor_ * tf;
                capacity = (int32_t)upper_bound(capacity_lower_bound);
                row_size = capacity * 2;
            }
            else
            {    
                capacity = 0;
                row_size = 0;
            }
            mem_block_size += row_size;

            if (tf >= alias_hot_thresh)
            {
                alias_capacity = num_topics_;
                alias_row_size = 2 * num_topics_;
            }
            else if (tf > 0)
            {
                alias_capacity = tf;
                alias_row_size = 3 * tf;
            }
            else
            {
                alias_capacity = 0;
                alias_row_size = 0;
            }
            alias_mem_block_size += alias_row_size;
        }
    }

    // NOTE: we can re-use the dict_ variable here, but we deliberately not use it.
    // This function should not change the internal state of model_block_
    void LDAModelBlock::GetModelStat(int64_t &mem_block_size, int64_t &alias_mem_block_size)
    {
        std::vector<int32_t> tfs(num_vocabs_, 0);
        CountNonZero(tfs);

        // calculate the mem_block_size, alias_mem_block_size
        GetModelSizeByTFS(true, tfs, mem_block_size, alias_mem_block_size);
    }
}
