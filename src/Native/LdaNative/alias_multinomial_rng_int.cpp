// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include "alias_multinomial_rng_int.hpp"
#include "rand_int_rng.h"
#include <ctime>
#include <list>
#include <algorithm>
#include <iostream>

namespace wood
{
    AliasMultinomialRNGInt::AliasMultinomialRNGInt()
        : n_(-1), internal_memory_(nullptr)
    {

    }
    AliasMultinomialRNGInt::~AliasMultinomialRNGInt()
    {
        if (internal_memory_ != nullptr)
        {
            delete[]internal_memory_;
        }
    }
    
    int32_t AliasMultinomialRNGInt::Next(xorshift_rng& rng, std::vector<alias_k_v>& alias_kv)
    {
        // NOTE: stl uniform_real_distribution generates the highest quality random numbers
        // yet, the other two are much faster
        auto sample = rng.rand();
        
        // NOTE: use std::floor is too slow
        // here we guarantee sample * n_ is nonnegative, this makes cast work
        int idx = sample / a_int_;

        if (n_ <= idx)
        {
            idx = n_ - 1;
        }

        // the following code is equivalent to 
        // return sample < V_[idx] ? idx : K_[idx];
        // but faster, see
        // https://stackoverflow.com/questions/6754454/speed-difference-between-if-else-and-ternary-operator-in-c
        int m = -(sample < alias_kv[idx].v_);
        return (idx & m) | (alias_kv[idx].k_ & ~m);
    }
}
