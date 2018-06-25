// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#define  NOMINMAX

#include <condition_variable>
#include <deque>
#include <atomic>
#include <mutex>
#include <list>


namespace lda {

    double LogGamma(double xx);
    double get_time();

    struct LDAEngineAtomics
    {
        LDAEngineAtomics() :doc_ll_(0), word_ll_(0), num_tokens_clock_(0), thread_counter_(0){}
        ~LDAEngineAtomics() {}

        std::atomic<double> doc_ll_;
        std::atomic<double> word_ll_;

        // # of tokens processed in a Clock() call.
        std::atomic<int> num_tokens_clock_;
        std::atomic<int> thread_counter_;

        std::mutex global_mutex_;
    };

    class CBlockedIntQueue
    {
    public:
        void clear();
        int pop();
        void push(int value);

    private:
        std::mutex _mutex;
        std::condition_variable _condition;
        std::deque<int> _queue;
    };


}
