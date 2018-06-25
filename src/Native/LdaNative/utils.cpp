// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include "utils.hpp"

#include "math.h"
#include <chrono>

namespace {
    const double cof[6] = { 76.18009172947146, -86.50532032941677,
        24.01409824083091, -1.231739572450155,
        0.1208650973866179e-2, -0.5395239384953e-5
    };
}

namespace lda {

    double LogGamma(double xx)
    {
        int j;
        double x, y, tmp1, ser;
        y = xx;
        x = xx;
        tmp1 = x + 5.5;
        tmp1 -= (x + 0.5)*log(tmp1);
        ser = 1.000000000190015;
        for (j = 0; j < 6; j++) ser += cof[j] / ++y;
        return -tmp1 + log(2.5066282746310005*ser / x);
    }


    double get_time() {
        auto start = std::chrono::high_resolution_clock::now();
        auto since_epoch = start.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1>>>(since_epoch).count();
    }

    void CBlockedIntQueue::clear()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.clear();
    }

    int CBlockedIntQueue::pop()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [this] { return !_queue.empty(); });
        auto val = _queue.front();
        _queue.pop_front();
        return val;
    }

    void CBlockedIntQueue::push(int value)
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _queue.push_back(value);
        }
        _condition.notify_one();
    }
}
