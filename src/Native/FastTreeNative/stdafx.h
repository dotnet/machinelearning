//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <cstdint>
#include <stdlib.h>

#ifndef _WIN32
#define EXPORT_API(ret) extern "C" __attribute__((visibility("default"))) ret
#include "../UnixSal.h"

// Workaround for _BitScanReverse using the gcc-specific __builtin_clz, which returns the number of leading zeros.
const int LongNumBitsMinusOne = sizeof(long) * 8 - 1;
inline unsigned char _BitScanReverse(_Inout_ unsigned long * index, _In_ unsigned long mask)
{
    *index = LongNumBitsMinusOne - __builtin_clz(mask);
    return mask == 0 ? 0 : 1;
}

// The gcc counterpart of qsort_s is qsort_r. The difference between those two is that
// context is the first argument in the comparator of qsort_s while for qsort_r
// it will be the last. Therefore, we create a ContextWrapper which will capture the original function pointer and 
// create the comparator gccCompare which will shift the arguments and call the captured function pointer.
struct ContextWrapper
{
    void *ctx;
    int(*compare)(void *ctx, const void *o1, const void *o2);
};

inline int gccCompare(const void * o1, const void * o2, _In_ void * ctxw)
{
    struct ContextWrapper *ctxws = (struct ContextWrapper*)ctxw;
    return (ctxws->compare)(ctxws->ctx, o1, o2);
}

inline int gccCompareApple(_In_ void * ctxw, const void * o1, const void * o2)
{
    struct ContextWrapper *ctxws = (struct ContextWrapper*)ctxw;
    return (ctxws->compare)(ctxws->ctx, o1, o2);
}

inline void qsort_s(_Inout_updates_all_(num) void* base, _In_ size_t num, _In_ size_t width, _In_ int(*compare)(void *, const void *, const void *), _In_ void* ctx)
{
    struct ContextWrapper wrapper;
    wrapper.ctx = ctx;
    wrapper.compare = compare;
#ifdef __APPLE__
    qsort_r(base, num, width, ctx, gccCompareApple);
#else
	qsort_r(base, num, width, gccCompare, ctx);
#endif
}

// __emul(u) is VS-specific.
inline uint64_t __emulu(unsigned int i1, unsigned int i2) { return (uint64_t)i1 * i2; };
inline int64_t __emul(int i1, int i2) { return (int64_t)i1 * i2; };
#else
#include <intrin.h>
#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret __stdcall
#endif