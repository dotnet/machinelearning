// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include <stdint.h>
#include <fstream>
#include <unordered_map>
#include <cassert>

/*
A light-weight hash table, borrowing the idea from google::dense_hash_map
0, <key, value> pair must be <int32_t, int32_t>
1, It can or can not own memory,
2, It has a fixed capacity, needless to resize or shrink,
3, capacity_ should at lease be twice of the maximum number of inserted items, guaranteeing a low load factor,
4, capacity_ should be an integer power of 2
5, emptry_key_ is fixed to 0
6, deleted_key_ is fixed to -2
*/

namespace lda
{
// The probing method:
// Linear probing
// #define JUMP_(key, num_probes)    ( 1 )

// Quadratic probing
#define JUMP_(key, num_probes)    ( num_probes )

#define ILLEGAL_BUCKET -1

    class light_hash_map
    {
    public:

        // must call set_memory after construction before use
        light_hash_map();
        // NOTE: the size of mem_block_ = 2 * capacity_
        light_hash_map(int32_t *mem_block, int32_t capacity);
        light_hash_map(int32_t capacity);

        ~light_hash_map();

        void clear();
        void set_memory(int32_t *mem_block);
        void sort();

        inline int32_t capacity() const;
        inline int32_t size() const;
        inline int32_t* key() const;
        inline int32_t* value() const;
        // whether we can find the |key| in this hash table
        inline bool has(int32_t key) const;

        // if |key| is already in table, increase its coresponding |value| with |delta|
        // if not, insert |key| into the table and set |delta| as the |value| of |key|
        inline void inc(int32_t key, int32_t delta);

        // query the value of |key|
        // if |key| is in the table, return the |value| corresonding to |key|
        // if not, just return 0
        inline int32_t operator[](int32_t key);

    private:

        light_hash_map(const light_hash_map &other) = delete;
        light_hash_map& operator=(const light_hash_map &other) = delete;

        // Returns a pair of positions: 1st where the object is, 2nd where
        // it would go if you wanted to insert it.  1st is ILLEGAL_BUCKET
        // if object is not found; 2nd is ILLEGAL_BUCKET if it is.
        // NOTE: because of deletions where-to-insert is not trivial: it's the
        // first deleted bucket we see, as long as we don't find the key later
        inline std::pair<int32_t, int32_t> find_position(const int32_t key) const;

        bool own_memory_;
        int32_t capacity_;
        int32_t *mem_block_;
        int32_t *key_;
        int32_t *value_;

        int32_t empty_key_;
        int32_t deleted_key_;
    };
    
    inline int32_t light_hash_map::capacity() const
    {
        return capacity_;
    }
    inline int32_t light_hash_map::size() const
    {
        int32_t size = 0;
        for (int i = 0; i < capacity_; ++i)
        {
            if (key_[i] > 0)
            {
                ++size;
            }
        }
        return size;
    }

    inline int32_t* light_hash_map::key() const
    {
        return key_;
    }
    inline int32_t* light_hash_map::value() const
    {
        return value_;
    }

    inline bool light_hash_map::has(int32_t key) const
    {
        int32_t internal_key = key + 1;
        std::pair<int32_t, int32_t> pos = find_position(internal_key);
        return pos.first != ILLEGAL_BUCKET;
    }

    inline void light_hash_map::inc(int32_t key, int32_t delta)
    {
        int32_t internal_key = key + 1;
        std::pair<int32_t, int32_t> pos = find_position(internal_key);
        if (pos.first != ILLEGAL_BUCKET)
        {
            value_[pos.first] += delta;
            if (value_[pos.first] == 0)       // the value becomes zero, delete the key
            {
                key_[pos.first] = deleted_key_;
            }
        }
        else                                 // not found the key, insert it with delta as value
        {
            key_[pos.second] = internal_key;
            value_[pos.second] = delta;
        }
    }

    inline int32_t light_hash_map::operator[](int32_t key)
    {
        int32_t internal_key = key + 1;
        std::pair<int32_t, int32_t> pos = find_position(internal_key);
        if (pos.first != ILLEGAL_BUCKET)
        {
            return value_[pos.first];
        }
        else
        {
            return 0;
        }
    }

    inline std::pair<int32_t, int32_t> light_hash_map::find_position(const int32_t key) const
    {
        int num_probes = 0;
        int32_t capacity_minus_one = capacity_ - 1;
        //int32_t idx = hasher_(key) & capacity_minus_one;
        int32_t idx = key % capacity_;
        int32_t insert_pos = ILLEGAL_BUCKET;
        while (1)                                           // probe until something happens
        {
            if (key_[idx] == empty_key_)                    // bucket is empty
            {
                if (insert_pos == ILLEGAL_BUCKET)           // found no prior place to insert
                {
                    return std::pair<int32_t, int32_t>(ILLEGAL_BUCKET, idx);
                }
                else                                        // previously, there is a position to insert
                {
                    return std::pair<int32_t, int32_t>(ILLEGAL_BUCKET, insert_pos);
                }
            }
            else if (key_[idx] == deleted_key_)            // keep searching, but makr to insert
            {
                if (insert_pos == ILLEGAL_BUCKET)
                {
                    insert_pos = idx;
                }
            }
            else if (key_[idx] == key)
            {
                return std::pair<int32_t, int32_t>(idx, ILLEGAL_BUCKET);
            }
            ++num_probes;                                // we are doing another probe
            idx = (idx + JUMP_(key, num_probes) & capacity_minus_one);
            assert(num_probes < capacity_
                && "Hashtable is full: an error in key_equal<> or hash<>");
        }
    }
}