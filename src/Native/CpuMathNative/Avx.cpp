// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// The exported function names need to be unique (can't be disambiguated based on signature), hence
// we introduce suffix letters to indicate the general patterns used.
// * A suffix means aligned and padded for SSE operations.
// * X suffix means aligned and padded for AVX operations.
// * U suffix means unaligned and unpadded.
// * S suffix means sparse (unaligned) vector.
// * P suffix means sparse (unaligned) partial vector - the vector is only part of a larger sparse vector.
// * R suffix means sparse matrix.
// * C suffix means convolution matrix.
// * D suffix means convolution matrix, with implicit source padding.
// * Tran means the matrix is transposed.
//
// Other notes:
// * AVX methods should end with _vleave() to avoid performance hit. See:
//   https://stackoverflow.com/questions/7839925/using-avx-cpu-instructions-poor-performance-without-archavx.
// * Keep Avx.cpp in sync with Sse.cpp. Note that Avx.cpp is compiled with /arch:AVX, but Sse.cpp is not.

// REVIEW: There is code below that mixes SSE and AVX instructions. Does compiling with /arch:AVX
// make that OK? Does the code need to be rewritten?

#include "../Stdafx.h"
#include <immintrin.h>

#ifndef _WIN32
#define _mm256_set_m128(va, vb) _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)
#endif

#define _vleave _mm256_zeroupper

#define _get_lo(x) _mm256_extractf128_ps(x, 0)
#define _get_hi(x) _mm256_extractf128_ps(x, 1)

#define _load1(ps, pi) \
    _mm_set_ss(ps[pi[0]])

#define _load4(ps, pi) \
    _mm_setr_ps(ps[pi[0]], ps[pi[1]], ps[pi[2]], ps[pi[3]])

#define _load8(ps, pi) \
    _mm256_setr_ps(ps[pi[0]], ps[pi[1]], ps[pi[2]], ps[pi[3]], ps[pi[4]], ps[pi[5]], ps[pi[6]], ps[pi[7]])

#define _rotate(x) _mm_shuffle_ps(x, x, 0x39)

#define _store1(x, pd, pi) \
    _mm_store_ss(pd + pi[0], x)

//Warning: this operation changes the value of x => do not reuse x
#define _store4(x, pd, pi) \
    _mm_store_ss(pd + pi[0], x); \
    x = _rotate(x); _mm_store_ss(pd + pi[1], x); \
    x = _rotate(x); _mm_store_ss(pd + pi[2], x); \
    x = _rotate(x); _mm_store_ss(pd + pi[3], x)

#define _store8(x, pd, pi) \
    __m128 tmp = _get_lo(x); _mm_store_ss(pd + pi[0], tmp); \
    tmp = _rotate(tmp);      _mm_store_ss(pd + pi[1], tmp); \
    tmp = _rotate(tmp);      _mm_store_ss(pd + pi[2], tmp); \
    tmp = _rotate(tmp);      _mm_store_ss(pd + pi[3], tmp); \
    tmp = _get_hi(x);        _mm_store_ss(pd + pi[4], tmp); \
    tmp = _rotate(tmp);      _mm_store_ss(pd + pi[5], tmp); \
    tmp = _rotate(tmp);      _mm_store_ss(pd + pi[6], tmp); \
    tmp = _rotate(tmp);      _mm_store_ss(pd + pi[7], tmp)

// Multiply matrix times vector into vector.
EXPORT_API(void) MatMulX(bool add, _In_ const float * pmat, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    const float * psLim = psrc + ccol;
    const float * pdLim = pdst + crow;
    const float * pm = pmat;
    for (float * pd = pdst; pd < pdLim; pd += 4, pm += 3 * ccol)
    {
        const float * ps = psrc;
        __m256 res0 = _mm256_setzero_ps();
        __m256 res1 = res0;
        __m256 res2 = res0;
        __m256 res3 = res0;
        for (; ps < psLim; ps += 8, pm += 8)
        {
            const float * pmTmp;
            __m256 x01 = _mm256_load_ps(pmTmp = pm);
            __m256 x11 = _mm256_load_ps(pmTmp += ccol);
            __m256 x21 = _mm256_load_ps(pmTmp += ccol);
            __m256 x31 = _mm256_load_ps(pmTmp += ccol);
            __m256 x02 = _mm256_load_ps(ps);
            x01 = _mm256_mul_ps(x01, x02);
            x11 = _mm256_mul_ps(x11, x02);
            x21 = _mm256_mul_ps(x21, x02);
            x31 = _mm256_mul_ps(x31, x02);
            res0 = _mm256_add_ps(res0, x01);
            res1 = _mm256_add_ps(res1, x11);
            res2 = _mm256_add_ps(res2, x21);
            res3 = _mm256_add_ps(res3, x31);
        }

        // Add up the entries of each, with the 4x2 results in res0
        res0 = _mm256_hadd_ps(res0, res1);
        res2 = _mm256_hadd_ps(res2, res3);
        res0 = _mm256_hadd_ps(res0, res2);

        __m128 sum = _mm_add_ps(_get_lo(res0), _get_hi(res0));
        if (add)
            sum = _mm_add_ps(sum, _mm_loadu_ps(pd));
        _mm_storeu_ps(pd, sum);
    }

    _vleave();
}

// Partial sparse source vector.
EXPORT_API(void) MatMulPX(bool add, _In_ const float * pmat, _In_ const int * pposSrc, _In_ const float * psrc,
    int posMin, int iposMin, int iposLim, _Inout_ float * pdst, int crow, int ccol)
{
    const int * pposMin = pposSrc + iposMin;
    const int * pposLim = pposSrc + iposLim;
    const float * pdLim = pdst + crow;
    const float * pm0 = pmat - posMin;
    const float * ps = psrc - posMin;
    for (float * pd = pdst; pd < pdLim; pd += 8, pm0 += 8 * ccol)
    {
        const float * pm1 = pm0 + ccol;
        const float * pm2 = pm1 + ccol;
        const float * pm3 = pm2 + ccol;
        __m256 res = _mm256_setzero_ps();
        for (const int * ppos = pposMin; ppos < pposLim; ppos++)
        {
            int col1 = *ppos;
            int col2 = col1 + 4 * ccol;
            __m256 x1 = _mm256_setr_ps(
                pm0[col1], pm1[col1], pm2[col1], pm3[col1],
                pm0[col2], pm1[col2], pm2[col2], pm3[col2]);
            __m256 x2 = _mm256_set1_ps(ps[col1]);
            x2 = _mm256_mul_ps(x2, x1);
            res = _mm256_add_ps(res, x2);
        }

        if (add)
            res = _mm256_add_ps(res, _mm256_load_ps(pd));
        _mm256_store_ps(pd, res);
    }

    _vleave();
}

EXPORT_API(void) MatMulTranX(bool add, _In_ const float * pmat, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    const float * psLim = psrc + ccol;
    const float * pdLim = pdst + crow;
    const float * pm = pmat;
    const float * ps = psrc;

    // We do 4-way unrolling
    if (!add)
    {
        __m128 h01 = _mm_loadu_ps(ps);
        // Replicate each slot of x01 into its own register.
        __m128 h11 = _mm_shuffle_ps(h01, h01, 0x55);
        __m128 h21 = _mm_shuffle_ps(h01, h01, 0xAA);
        __m128 h31 = _mm_shuffle_ps(h01, h01, 0xFF);
        h01 = _mm_shuffle_ps(h01, h01, 0x00);

        __m256 x01 = _mm256_set_m128(h01, h01);
        __m256 x11 = _mm256_set_m128(h11, h11);
        __m256 x21 = _mm256_set_m128(h21, h21);
        __m256 x31 = _mm256_set_m128(h31, h31);
        ps += 4;

        for (float * pd = pdst; pd < pdLim; pd += 8, pm += 8)
        {
            const float * pmTmp;
            __m256 x02 = _mm256_load_ps(pmTmp = pm);
            __m256 x12 = _mm256_load_ps(pmTmp += crow);
            __m256 x22 = _mm256_load_ps(pmTmp += crow);
            __m256 x32 = _mm256_load_ps(pmTmp += crow);
            x02 = _mm256_mul_ps(x01, x02);
            x12 = _mm256_mul_ps(x11, x12);
            x22 = _mm256_mul_ps(x21, x22);
            x32 = _mm256_mul_ps(x31, x32);
            x02 = _mm256_add_ps(x02, x12);
            x22 = _mm256_add_ps(x22, x32);
            x02 = _mm256_add_ps(x02, x22);
            _mm256_store_ps(pd, x02);
        }

        pm += 3 * crow;
    }

    for (; ps < psLim; ps += 4)
    {
        __m128 h01 = _mm_loadu_ps(ps);
        // Replicate each slot of x01 into its own register.
        __m128 h11 = _mm_shuffle_ps(h01, h01, 0x55);
        __m128 h21 = _mm_shuffle_ps(h01, h01, 0xAA);
        __m128 h31 = _mm_shuffle_ps(h01, h01, 0xFF);
        h01 = _mm_shuffle_ps(h01, h01, 0x00);

        __m256 x01 = _mm256_set_m128(h01, h01);
        __m256 x11 = _mm256_set_m128(h11, h11);
        __m256 x21 = _mm256_set_m128(h21, h21);
        __m256 x31 = _mm256_set_m128(h31, h31);

        for (float * pd = pdst; pd < pdLim; pd += 8, pm += 8)
        {
            const float * pmTmp;
            __m256 x02 = _mm256_load_ps(pmTmp = pm);
            __m256 x12 = _mm256_load_ps(pmTmp += crow);
            __m256 x22 = _mm256_load_ps(pmTmp += crow);
            __m256 x32 = _mm256_load_ps(pmTmp += crow);
            __m256 x3 = _mm256_load_ps(pd);
            x02 = _mm256_mul_ps(x01, x02);
            x12 = _mm256_mul_ps(x11, x12);
            x22 = _mm256_mul_ps(x21, x22);
            x32 = _mm256_mul_ps(x31, x32);
            x02 = _mm256_add_ps(x02, x12);
            x22 = _mm256_add_ps(x22, x32);
            x02 = _mm256_add_ps(x02, x22);
            x3 = _mm256_add_ps(x02, x3);
            _mm256_store_ps(pd, x3);
        }

        pm += 3 * crow;
    }

    _vleave();
}

// Partial sparse source vector.
EXPORT_API(void) MatMulTranPX(bool add, _In_ const float * pmat, _In_ const int * pposSrc, _In_ const float * psrc,
    int posMin, int iposMin, int iposLim, _Inout_ float * pdst, int crow)
{
    const int * ppos = pposSrc + iposMin;
    const int * pposLim = pposSrc + iposLim;
    const float * pdLim = pdst + crow;

    if (!add)
    {
        int col = *ppos++ - posMin;
        const float * pm = pmat + col * crow;
        __m256 x0 = _mm256_set1_ps(psrc[col]);
        for (float * pd = pdst; pd < pdLim; pd += 8, pm += 8)
        {
            __m256 x1 = _mm256_load_ps(pm);
            x1 = _mm256_mul_ps(x1, x0);
            _mm256_store_ps(pd, x1);
        }
    }

    // REVIEW: Should we explore unrolling the outer loop?
    for (; ppos < pposLim; ppos++)
    {
        int col = *ppos - posMin;
        __m256 x0 = _mm256_set1_ps(psrc[col]);
        const float * pm = pmat + col * crow;
        for (float * pd = pdst; pd < pdLim; pd += 8, pm += 8)
        {
            __m256 x1 = _mm256_load_ps(pm);
            __m256 x2 = _mm256_load_ps(pd);
            x1 = _mm256_mul_ps(x1, x0);
            x2 = _mm256_add_ps(x2, x1);
            _mm256_store_ps(pd, x2);
        }
    }

    _vleave();
}