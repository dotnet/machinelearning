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
//   http://stackoverflow.com/questions/7839925/using-avx-cpu-instructions-poor-performance-without-archavx.
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
            sum = _mm_add_ps(sum, _mm_load_ps(pd));
        _mm_store_ps(pd, sum);
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

// Sparse matrix.
EXPORT_API(void) MatMulRX(bool add, _In_ const int * pstarts, _In_ const int * pindices, _In_ const float * pcoefs,
    _In_ const float * ps, _Inout_ float * pdst, int crow)
{
    const int * pii = pstarts + 1;
    const int * pi = pindices;
    const float * pm = pcoefs;
    const float * pdLim = pdst + crow;
    for (float * pd = pdst; pd < pdLim; pd++)
    {
        const int * piLim = pindices + *pii++;

        __m256 res2 = _mm256_setzero_ps();
        for (; pi + 8 <= piLim; pi += 8, pm += 8)
        {
            __m256 x = _mm256_mul_ps(_load8(ps, pi), _mm256_loadu_ps(pm));
            res2 = _mm256_add_ps(res2, x);
        }
        __m128 res = _mm_add_ps(_get_lo(res2), _get_hi(res2));
        if (pi + 4 <= piLim)
        {
            __m128 x = _mm_mul_ps(_load4(ps, pi), _mm_loadu_ps(pm));
            res = _mm_add_ps(res, x);
            pi += 4; pm += 4;
        }
        for (; pi < piLim; pi++, pm++)
        {
            __m128 x = _mm_mul_ss(_load1(ps, pi), _mm_set_ss(*pm));
            res = _mm_add_ss(res, x);
        }
        res = _mm_hadd_ps(res, res);
        res = _mm_hadd_ps(res, res);

        if (add)
            res = _mm_add_ss(res, _mm_set_ss(*pd));
        _mm_store_ss(pd, res);
    }

    _vleave();
}

// Unpadded convolution.
EXPORT_API(void) MatMulCX(bool add, _In_ const int * pmprowiv, _In_ const int * pmprowcol,
    _In_ const int * pruns, _In_ const float * pcoefs, _In_ const float * psrc, _Inout_ float * pdst, int crow)
{
    int size = pruns[1];
    const int * psupport = pruns + 2;
    const int * piv = pmprowiv;
    const int * pcol = pmprowcol;
    const int * piLim = psupport + size;
    const float * pdLim = pdst + crow;

    for (float * pd = pdst; pd < pdLim; pd++)
    {
        const float * pm = pcoefs + *piv++;
        const float * ps = psrc + *pcol++;
        const int * pi = psupport;

        __m256 res2 = _mm256_setzero_ps();
        for (; pi + 8 <= piLim; pi += 8, pm += 8)
        {
            __m256 x = _mm256_mul_ps(_load8(ps, pi), _mm256_loadu_ps(pm));
            res2 = _mm256_add_ps(res2, x);
        }
        __m128 res = _mm_add_ps(_get_lo(res2), _get_hi(res2));
        if (pi + 4 <= piLim)
        {
            __m128 x = _mm_mul_ps(_load4(ps, pi), _mm_loadu_ps(pm));
            res = _mm_add_ps(res, x);
            pi += 4; pm += 4;
        }
        for (; pi < piLim; pi++, pm++)
        {
            __m128 x = _mm_mul_ss(_load1(ps, pi), _mm_set_ss(*pm));
            res = _mm_add_ss(res, x);
        }
        res = _mm_hadd_ps(res, res);
        res = _mm_hadd_ps(res, res);

        // Add the bias.
        res = _mm_add_ss(res, _mm_set_ss(*pm));

        if (add)
            res = _mm_add_ss(res, _mm_set_ss(*pd));
        _mm_store_ss(pd, res);
    }

    _vleave();
}

// Padded convolution.
EXPORT_API(void) MatMulDX(bool add, _In_ const int * pmprowiv, _In_ const int * pmprowcol, _In_ const int * pmprowrun,
    _In_ const int * pruns, _In_ const float * pcoefs, _In_ const float * psrc, _Inout_ float * pdst, int crow)
{
    const int * piv = pmprowiv;
    const int * pcol = pmprowcol;
    const float * pdLim = pdst + crow;
    int kernelSize = pruns[1];

    const int * pirun = pmprowrun;
    for (float * pd = pdst; pd < pdLim; pd++)
    {
        const float * pm = pcoefs + *piv++;
        const float * pmBias = pm + kernelSize;
        const float * ps = psrc + *pcol++;
        int irun = *pirun++;

        const int * pi = pruns + 2 + irun;
        const int * piLim = pi + pi[-1];
        __m256 res2 = _mm256_setzero_ps();
        __m128 res;
        if (irun == 0)
        {
            // No masking needed.
            for (; pi + 8 <= piLim; pi += 8, pm += 8)
            {
                __m256 x = _mm256_mul_ps(_load8(ps, pi), _mm256_loadu_ps(pm));
                res2 = _mm256_add_ps(res2, x);
            }
            res = _mm_add_ps(_get_lo(res2), _get_hi(res2));
            if (pi + 4 <= piLim)
            {
                __m128 x = _mm_mul_ps(_load4(ps, pi), _mm_loadu_ps(pm));
                res = _mm_add_ps(res, x);
                pi += 4; pm += 4;
            }
            for (; pi < piLim; pi++, pm++)
            {
                __m128 x = _mm_mul_ss(_load1(ps, pi), _mm_set_ss(*pm));
                res = _mm_add_ss(res, x);
            }
        }
        else
        {
            // Need masking.
            pm += pi[-2];
            const float * pmask = reinterpret_cast<const float *>(piLim);
            for (; pi + 8 <= piLim; pi += 8, pm += 8, pmask += 8)
            {
                __m256 x = _mm256_mul_ps(_load8(ps, pi), _mm256_and_ps(_mm256_loadu_ps(pmask), _mm256_loadu_ps(pm)));
                res2 = _mm256_add_ps(res2, x);
            }
            res = _mm_add_ps(_get_lo(res2), _get_hi(res2));
            if (pi + 4 <= piLim)
            {
                __m128 x = _mm_mul_ps(_load4(ps, pi), _mm_and_ps(_mm_loadu_ps(pmask), _mm_loadu_ps(pm)));
                res = _mm_add_ps(res, x);
                pi += 4; pm += 4; pmask += 4;
            }
            for (; pi < piLim; pi++, pm++, pmask++)
            {
                __m128 x = _mm_mul_ss(_load1(ps, pi), _mm_and_ps(_mm_set_ss(*pmask), _mm_set_ss(*pm)));
                res = _mm_add_ss(res, x);
            }
        }
        res = _mm_hadd_ps(res, res);
        res = _mm_hadd_ps(res, res);

        res = _mm_add_ss(res, _mm_set_ss(*pmBias));
        if (add)
            res = _mm_add_ss(res, _mm_set_ss(*pd));
        _mm_store_ss(pd, res);
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
        __m128 h01 = _mm_load_ps(ps);
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
        __m128 h01 = _mm_load_ps(ps);
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

// Sparse matrix.
EXPORT_API(void) MatMulTranRX(bool add, _In_ const int * pstarts, _In_ const int * pindices, _In_ const float * pcoefs,
    _In_ const float * psrc, _Inout_ float * pd, int crow, int ccol)
{
    if (!add)
        memset(pd, 0, crow * sizeof(float));

    const int * pii = pstarts + 1;
    const int * pi = pindices;
    const float * pm = pcoefs;
    const float * psLim = psrc + ccol;
    for (const float * ps = psrc; ps < psLim; ps++)
    {
        float x = *ps;
        const int * piLim = pindices + *pii++;

        __m128 x0 = _mm_set1_ps(x);
        __m256 x1 = _mm256_set_m128(x0, x0);
        for (; pi + 8 <= piLim; pi += 8, pm += 8)
        {
            __m256 x2 = _mm256_mul_ps(x1, _mm256_loadu_ps(pm));
            x2 = _mm256_add_ps(x2, _load8(pd, pi));
            _store8(x2, pd, pi);
        }
        if (pi + 4 <= piLim)
        {
            __m128 x2 = _mm_mul_ps(x0, _mm_loadu_ps(pm));
            x2 = _mm_add_ps(x2, _load4(pd, pi));
            _store4(x2, pd, pi);
            pi += 4; pm += 4;
        }
        for (; pi < piLim; pi++, pm++)
        {
            __m128 x2 = _mm_mul_ss(x0, _mm_set_ss(*pm));
            x2 = _mm_add_ss(x2, _load1(pd, pi));
            _store1(x2, pd, pi);
        }
    }

    _vleave();
}

// Unpadded convolution.
EXPORT_API(void) MatMulTranCX(bool add, _In_ const int * pmpcoliv, _In_ const int * pmpcolrow,
    _In_ const int * pruns, _In_ const float * pcoefs, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    if (!add)
        memset(pdst, 0, crow * sizeof(float));

    int size = pruns[1];
    const int * psupport = pruns + 2;
    const int * piv = pmpcoliv;
    const int * prow = pmpcolrow;
    const int * piLim = psupport + size;
    const float * psLim = psrc + ccol;
    for (const float * ps = psrc; ps < psLim; ps++)
    {
        const float * pm = pcoefs + *piv++;
        float * pd = pdst + *prow++;
        const int * pi = psupport;

        float x = *ps;
        __m128 x0 = _mm_set1_ps(x);
        __m256 x1 = _mm256_set_m128(x0, x0);
        for (; pi + 8 <= piLim; pi += 8, pm += 8)
        {
            __m256 x2 = _mm256_mul_ps(x1, _mm256_loadu_ps(pm));
            x2 = _mm256_add_ps(x2, _load8(pd, pi));
            _store8(x2, pd, pi);
        }
        if (pi + 4 <= piLim)
        {
            __m128 x2 = _mm_mul_ps(x0, _mm_loadu_ps(pm));
            x2 = _mm_add_ps(x2, _load4(pd, pi));
            _store4(x2, pd, pi);
            pi += 4; pm += 4;
        }
        for (; pi < piLim; pi++, pm++)
        {
            __m128 x2 = _mm_mul_ss(x0, _mm_set_ss(*pm));
            x2 = _mm_add_ss(x2, _load1(pd, pi));
            _store1(x2, pd, pi);
        }
    }

    _vleave();
}

// Padded convolution.
EXPORT_API(void) MatMulTranDX(bool add, _In_ const int * pmpcoliv, _In_ const int * pmpcolrow, _In_ const int * pmpcolrun,
    _In_ const int * pruns, _In_ const float * pcoefs, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    if (!add)
        memset(pdst, 0, crow * sizeof(float));

    const int * piv = pmpcoliv;
    const int * prow = pmpcolrow;
    const float * psLim = psrc + ccol;
    int kernelSize = pruns[1];

    const int * pirun = pmpcolrun;
    for (const float * ps = psrc; ps < psLim; ps++)
    {
        const float * pm = pcoefs + *piv++;
        float * pd = pdst + *prow++;
        int irun = *pirun++;
        const int * pi = pruns + 2 + irun;
        const int * piLim = pi + pi[-1];

        float x = *ps;
        __m128 x0 = _mm_set1_ps(x);
        __m256 x1 = _mm256_set_m128(x0, x0);
        if (irun == 0)
        {
            // No masking needed.
            for (; pi + 8 <= piLim; pi += 8, pm += 8)
            {
                __m256 x2 = _mm256_mul_ps(x1, _mm256_loadu_ps(pm));
                x2 = _mm256_add_ps(x2, _load8(pd, pi));
                _store8(x2, pd, pi);
            }
            if (pi + 4 <= piLim)
            {
                __m128 x2 = _mm_mul_ps(x0, _mm_loadu_ps(pm));
                x2 = _mm_add_ps(x2, _load4(pd, pi));
                _store4(x2, pd, pi);
                pi += 4; pm += 4;
            }
            for (; pi < piLim; pi++, pm++)
            {
                __m128 x2 = _mm_mul_ss(x0, _mm_set_ss(*pm));
                x2 = _mm_add_ss(x2, _load1(pd, pi));
                _store1(x2, pd, pi);
            }
        }
        else
        {
            // Need masking.
            pm += pi[-2];
            const float * pmask = reinterpret_cast<const float *>(piLim);
            for (; pi + 8 <= piLim; pi += 8, pm += 8, pmask += 8)
            {
                __m256 x2 = _mm256_mul_ps(_mm256_and_ps(_mm256_loadu_ps(pmask), x1), _mm256_loadu_ps(pm));
                x2 = _mm256_add_ps(x2, _load8(pd, pi));
                _store8(x2, pd, pi);
            }
            if (pi + 4 <= piLim)
            {
                __m128 x2 = _mm_mul_ps(_mm_and_ps(_mm_loadu_ps(pmask), x0), _mm_loadu_ps(pm));
                x2 = _mm_add_ps(x2, _load4(pd, pi));
                _store4(x2, pd, pi);
                pi += 4; pm += 4; pmask += 4;
            }
            for (; pi < piLim; pi++, pm++, pmask++)
            {
                __m128 x2 = _mm_mul_ss(_mm_and_ps(_mm_set_ss(*pmask), x0), _mm_set_ss(*pm));
                x2 = _mm_add_ss(x2, _load1(pd, pi));
                _store1(x2, pd, pi);
            }
        }
    }

    _vleave();
}

template <bool useDecay>
void AddXYTranXCore(float a, _In_ const float * px, _In_ const float * py, _Inout_ float * pmat, int crow, int ccol, float decay)
{
    const float * pyBase = py;
    const float * pxLim = px + crow;
    const float * pyLim = py + ccol;
    float * pm = pmat;
    __m256 wd;
    if (useDecay)
        wd = _mm256_set1_ps(1 - decay);
    for (; px < pxLim; px++)
    {
        float r = a * *px;
        py = pyBase;

        __m256 x1 = _mm256_set1_ps(r);
        for (; py + 32 <= pyLim; py += 32, pm += 32)
        {
            __m256 x02 = _mm256_load_ps(py);
            __m256 x12 = _mm256_load_ps(py + 8);
            __m256 x22 = _mm256_load_ps(py + 16);
            __m256 x32 = _mm256_load_ps(py + 24);
            __m256 x03 = _mm256_load_ps(pm);
            __m256 x13 = _mm256_load_ps(pm + 8);
            __m256 x23 = _mm256_load_ps(pm + 16);
            __m256 x33 = _mm256_load_ps(pm + 24);
            x02 = _mm256_mul_ps(x1, x02);
            x12 = _mm256_mul_ps(x1, x12);
            x22 = _mm256_mul_ps(x1, x22);
            x32 = _mm256_mul_ps(x1, x32);
            if (useDecay)
            {
                x03 = _mm256_mul_ps(wd, x03);
                x13 = _mm256_mul_ps(wd, x13);
                x23 = _mm256_mul_ps(wd, x23);
                x33 = _mm256_mul_ps(wd, x33);
            }
            x03 = _mm256_add_ps(x02, x03);
            x13 = _mm256_add_ps(x12, x13);
            x23 = _mm256_add_ps(x22, x23);
            x33 = _mm256_add_ps(x32, x33);
            _mm256_store_ps(pm, x03);
            _mm256_store_ps(pm + 8, x13);
            _mm256_store_ps(pm + 16, x23);
            _mm256_store_ps(pm + 24, x33);
        }
        for (; py < pyLim; py += 8, pm += 8)
        {
            __m256 x02 = _mm256_load_ps(py);
            __m256 x03 = _mm256_load_ps(pm);
            x02 = _mm256_mul_ps(x1, x02);
            if (useDecay)
                x03 = _mm256_mul_ps(wd, x03);
            x03 = _mm256_add_ps(x02, x03);
            _mm256_store_ps(pm, x03);
        }
    }

    _vleave();
}

EXPORT_API(void) AddXYTranX(float a, _In_ const float * px, _In_ const float * py, _Inout_ float * pmat, int crow, int ccol, float decay)
{
    if (decay == 0)
        AddXYTranXCore<false>(a, px, py, pmat, crow, ccol, decay);
    else
        AddXYTranXCore<true>(a, px, py, pmat, crow, ccol, decay);
}

// Partial sparse source vector.
EXPORT_API(void) AddXYTranPX(float a, _In_ const float * px, _In_ const int * pposY, _In_ const float * pvaluesY,
    int posMinY, int iposMinY, int iposLimY, _Inout_ float * pmat, int crow, int ccol)
{
    const int * pposMin = pposY + iposMinY;
    const int * pposLim = pposY + iposLimY;
    const float * pxLim = px + crow;
    float * pm0 = pmat - posMinY;
    const float * py = pvaluesY - posMinY;

    __m256 x0 = _mm256_set1_ps(a);
    for (; px < pxLim; px += 8, pm0 += 8 * ccol)
    {
        float * pm1 = pm0 + ccol;
        float * pm2 = pm1 + ccol;
        float * pm3 = pm2 + ccol;

        __m256 x1 = _mm256_load_ps(px);
        x1 = _mm256_mul_ps(x1, x0);

        for (const int * ppos = pposMin; ppos < pposLim; ppos++)
        {
            int col1 = *ppos;
            int col2 = col1 + 4 * ccol;
            __m256 x2 = _mm256_set1_ps(py[col1]);
            __m256 x3 = _mm256_setr_ps(
                pm0[col1], pm1[col1], pm2[col1], pm3[col1],
                pm0[col2], pm1[col2], pm2[col2], pm3[col2]);
            x2 = _mm256_mul_ps(x2, x1);
            x3 = _mm256_add_ps(x3, x2);

            __m128 t1 = _get_lo(x3);
            __m128 t2 = _get_hi(x3);
            _mm_store_ss(pm0 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pm1 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pm2 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pm3 + col1, t1);
            _mm_store_ss(pm0 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pm1 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pm2 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pm3 + col2, t2);
        }
    }

    _vleave();
}

template <bool useDecay>
void AddXYTranRXCore(float a, _In_ const float * px, _In_ const float * py,
    _In_ const int * pstarts, _In_ const int * pindices, _Inout_ float * pcoefs, int crow, float decay)
{
    const int * pii = pstarts + 1;
    const int * pi = pindices;
    float * pm = pcoefs;
    const float * pxLim = px + crow;
    __m128 wd0;
    __m256 wd1;
    if (useDecay)
    {
        wd0 = _mm_set1_ps(1 - decay);
        wd1 = _mm256_set_m128(wd0, wd0);
    }
    for (; px < pxLim; px++)
    {
        const int * piLim = pindices + *pii++;
        float r = a * *px;

        __m128 x0 = _mm_set1_ps(r);
        __m256 x1 = _mm256_set_m128(x0, x0);
        for (; pi + 8 <= piLim; pi += 8, pm += 8)
        {
            __m256 x2 = _mm256_mul_ps(x1, _load8(py, pi));
            __m256 x3 = _mm256_loadu_ps(pm);
            if (useDecay)
                x3 = _mm256_mul_ps(x3, wd1);
            x2 = _mm256_add_ps(x2, x3);
            _mm256_storeu_ps(pm, x2);
        }
        if (pi + 4 <= piLim)
        {
            __m128 x2 = _mm_mul_ps(x0, _load4(py, pi));
            __m128 x3 = _mm_loadu_ps(pm);
            if (useDecay)
                x3 = _mm_mul_ps(x3, wd0);
            x2 = _mm_add_ps(x2, x3);
            _mm_storeu_ps(pm, x2);
            pi += 4; pm += 4;
        }
        for (; pi < piLim; pi++, pm++)
            *pm = (useDecay ? (*pm * (1 - decay)) : *pm) + py[*pi] * r;
    }

    _vleave();
}

// Sparse matrix.
EXPORT_API(void) AddXYTranRX(float a, _In_ const float * px, _In_ const float * py,
    _In_ const int * pstarts, _In_ const int * pindices, _Inout_ float * pcoefs, int crow, float decay)
{
    if (decay == 0)
        AddXYTranRXCore<false>(a, px, py, pstarts, pindices, pcoefs, crow, decay);
    else
        AddXYTranRXCore<true>(a, px, py, pstarts, pindices, pcoefs, crow, decay);
}

// Unpadded convolution.
EXPORT_API(void) AddXYTranCX(float a, _In_ const float * px, _In_ const float * py, _In_ const int * pmprowiv, _In_ const int * pmprowcol,
    _In_ const int * pruns, _Inout_ float * pcoefs, int crow)
{
    int size = pruns[1];
    const int * psupport = pruns + 2;
    const int * piv = pmprowiv;
    const int * pcol = pmprowcol;
    const float * pxLim = px + crow;
    const int * piLim = psupport + size;

    for (; px < pxLim; px++)
    {
        float * pm = pcoefs + *piv++;
        const float * ps = py + *pcol++;
        const int * pi = psupport;
        float r = a * *px;

        __m128 x0 = _mm_set1_ps(r);
        __m256 x1 = _mm256_set_m128(x0, x0);
        for (; pi + 8 <= piLim; pi += 8, pm += 8)
        {
            __m256 x2 = _mm256_mul_ps(x1, _load8(ps, pi));
            x2 = _mm256_add_ps(x2, _mm256_loadu_ps(pm));
            _mm256_storeu_ps(pm, x2);
        }
        if (pi + 4 <= piLim)
        {
            __m128 x2 = _mm_mul_ps(x0, _load4(ps, pi));
            x2 = _mm_add_ps(x2, _mm_loadu_ps(pm));
            _mm_storeu_ps(pm, x2);
            pi += 4; pm += 4;
        }
        for (; pi < piLim; pi++, pm++)
            *pm += ps[*pi] * r;
        // Update the bias.
        *pm += r;
    }

    _vleave();
}

// Padded convolution.
EXPORT_API(void) AddXYTranDX(float a, _In_ const float * px, _In_ const float * py, _In_ const int * pmprowiv, _In_ const int * pmprowcol,
    _In_ const int * pmprowrun, _In_ const int * pruns, _Inout_ float * pcoefs, int crow)
{
    const int * piv = pmprowiv;
    const int * pcol = pmprowcol;
    const float * pxLim = px + crow;
    int kernelSize = pruns[1];

    const int * pirun = pmprowrun;
    for (; px < pxLim; px++)
    {
        float * pm = pcoefs + *piv++;
        const float * ps = py + *pcol++;
        int irun = *pirun++;
        const int * pi = pruns + 2 + irun;
        const int * piLim = pi + pi[-1];

        float r = a * *px;

        // Update the bias.
        pm[kernelSize] += r;

        __m128 x0 = _mm_set1_ps(r);
        __m256 x1 = _mm256_set_m128(x0, x0);
        if (irun == 0)
        {
            // No masking needed.
            for (; pi + 8 <= piLim; pi += 8, pm += 8)
            {
                __m256 x2 = _mm256_mul_ps(x1, _load8(ps, pi));
                x2 = _mm256_add_ps(x2, _mm256_loadu_ps(pm));
                _mm256_storeu_ps(pm, x2);
            }
            if (pi + 4 <= piLim)
            {
                __m128 x2 = _mm_mul_ps(x0, _load4(ps, pi));
                x2 = _mm_add_ps(x2, _mm_loadu_ps(pm));
                _mm_storeu_ps(pm, x2);
                pi += 4; pm += 4;
            }
            for (; pi < piLim; pi++, pm++)
                *pm += ps[*pi] * r;
        }
        else
        {
            // Need masking.
            pm += pi[-2];
            const float * pmask = reinterpret_cast<const float *>(piLim);
            for (; pi + 8 <= piLim; pi += 8, pm += 8, pmask += 8)
            {
                __m256 x2 = _mm256_mul_ps(_mm256_and_ps(_mm256_loadu_ps(pmask), x1), _load8(ps, pi));
                x2 = _mm256_add_ps(x2, _mm256_loadu_ps(pm));
                _mm256_storeu_ps(pm, x2);
            }
            if (pi + 4 <= piLim)
            {
                __m128 x2 = _mm_mul_ps(_mm_and_ps(_mm_loadu_ps(pmask), x0), _load4(ps, pi));
                x2 = _mm_add_ps(x2, _mm_loadu_ps(pm));
                _mm_storeu_ps(pm, x2);
                pi += 4; pm += 4; pmask += 4;
            }
            for (; pi < piLim; pi++, pm++, pmask++)
            {
                __m128 x2 = _mm_mul_ss(_mm_and_ps(_mm_set_ss(*pmask), x0), _load1(ps, pi));
                x2 = _mm_add_ss(x2, _mm_set_ss(*pm));
                _mm_store_ss(pm, x2);
            }
        }
    }

    _vleave();
}

// With momentum.
EXPORT_API(void) AddXYTranMomX(float a, _In_ const float * px, _In_ const float * py, _Inout_ float * pmat, float momentum, _Inout_ float * pdel, int crow, int ccol)
{
    const float * pyBase = py;
    const float * pxLim = px + crow;
    const float * pyLim = py + ccol;
    float * pm = pmat;
    float * pd = pdel;

    __m256 x0 = _mm256_set1_ps(momentum);
    for (; px < pxLim; px++)
    {
        float r = a * *px;

        __m256 x1 = _mm256_set1_ps(r);
        for (py = pyBase; py < pyLim; pm += 8, pd += 8, py += 8)
        {
            __m256 x2 = _mm256_load_ps(py);
            __m256 x3 = _mm256_load_ps(pd);
            __m256 x4 = _mm256_load_ps(pm);
            x2 = _mm256_mul_ps(x1, x2);
            x3 = _mm256_mul_ps(x0, x3);
            x3 = _mm256_add_ps(x2, x3);
            x4 = _mm256_add_ps(x3, x4);

            _mm256_store_ps(pd, x3);
            _mm256_store_ps(pm, x4);
        }
    }

    _vleave();
}

// coef: coefs to update, ag: accumulated grads, au: accumulated updates, g: cur grads.
// Note: parameters coef, ag, au and g will be updated, do not reuse parameter g in calling code.
__forceinline void UpdateAdadelta(__m256& coef, __m256& ag, __m256& au, __m256& g, const __m256& dec, const __m256& decc, const __m256& c)
{
    __m256 x4 = _mm256_mul_ps(g, g);   // x4 == g * g
    x4 = _mm256_mul_ps(decc, x4);      // x4 == (1 - decay) * g * g
    ag = _mm256_mul_ps(dec, ag);       // ag == decay * accG
    ag = _mm256_add_ps(ag, x4);        // ag == decay * accG + (1 - decay) * g * g
    __m256 x41 = _mm256_add_ps(ag, c); // x41 == ag + cond
    __m256 x51 = _mm256_add_ps(au, c); // x51 == accU + cond
#if 0
    // naive version:
    x51 = _mm256_div_ps(x51, x41);
    x41 = _mm256_sqrt_ps(x51);         // x41 == rate
#else
    // faster (approximate) version:
    x41 = _mm256_rsqrt_ps(x41);
    __m256 x52 = _mm256_rsqrt_ps(x51);
    x51 = _mm256_mul_ps(x51, x52);
    x41 = _mm256_mul_ps(x41, x51);     // x41 == rate
#endif
    g = _mm256_mul_ps(g, x41);         // g - current update
    coef = _mm256_add_ps(coef, g);

    g = _mm256_mul_ps(g, g);           // g  == newU * newU
    g = _mm256_mul_ps(decc, g);        // g  == (1 - decay) * newU * newU
    au = _mm256_mul_ps(dec, au);       // au == decay * accU
    au = _mm256_add_ps(au, g);         // au == decay * accU + (1 - decay) * newU * newU
}

// For Adadelta.
EXPORT_API(void) AddXYTranGradX(_In_ const float * px, _In_ const float * py, _Inout_ float * pmat, _Inout_ float * paccGrads, _Inout_ float * paccUpdates,
    float decay, float cond, int crow, int ccol)
{
    const float * pyBase = py;
    const float * pxLim = px + crow;
    const float * pyLim = py + ccol;
    float * pm = pmat;
    float * pag = paccGrads;
    float * pau = paccUpdates;

    __m256 dec = _mm256_set1_ps(decay);
    __m256 decc = _mm256_set1_ps(1 - decay);
    __m256 c = _mm256_set1_ps(cond);
    for (; px < pxLim; px++)
    {
        float r = *px;

        __m256 x1 = _mm256_set1_ps(r);
        for (py = pyBase; py < pyLim; pm += 8, pag += 8, pau += 8, py += 8)
        {
            __m256 x2 = _mm256_load_ps(py);
            __m256 ag = _mm256_load_ps(pag);
            __m256 au = _mm256_load_ps(pau);
            __m256 coef = _mm256_load_ps(pm);
            x2 = _mm256_mul_ps(x1, x2);         // x2 == g

            UpdateAdadelta(coef, ag, au, x2, dec, decc, c);

            _mm256_store_ps(pm, coef);
            _mm256_store_ps(pag, ag);
            _mm256_store_ps(pau, au);
        }
    }

    _vleave();
}

// For Adadelta, sparse matrix.
EXPORT_API(void) AddXYTranGradRX(_In_ const float * px, _In_ const float * py, _In_ const int * pstarts, _In_ const int * pindices,
    _Inout_ float * pcoefs, _Inout_ float * paccGrads, _Inout_ float * paccUpdates, float decay, float cond, int crow)
{
    const int * pii = pstarts + 1;
    const int * pi = pindices;
    float * pm = pcoefs;
    const float * pxLim = px + crow;
    float * pag = paccGrads;
    float * pau = paccUpdates;

    __m256 dec = _mm256_set1_ps(decay);
    __m256 decc = _mm256_set1_ps(1 - decay);
    __m256 c = _mm256_set1_ps(cond);

    for (; px < pxLim; px++)
    {
        const int * piLim = pindices + *pii++;
        float r = *px;

        __m256 x1 = _mm256_set1_ps(r);
        for (; pi + 8 <= piLim; pi += 8, pm += 8, pag += 8, pau += 8)
        {
            __m256 g = _mm256_mul_ps(x1, _load8(py, pi));
            __m256 ag = _mm256_loadu_ps(pag);
            __m256 au = _mm256_loadu_ps(pau);
            __m256 coef = _mm256_loadu_ps(pm);

            UpdateAdadelta(coef, ag, au, g, dec, decc, c);

            _mm256_storeu_ps(pm, coef);
            _mm256_storeu_ps(pag, ag);
            _mm256_storeu_ps(pau, au);
        }

        // REVIEW: Why is this so different than the SSE version?
        for (; pi < piLim; pi++, pm++, pag++, pau++)
        {
            float g = py[*pi] * r;
            float accGrad = decay * *pag + (1 - decay) * g * g;
            float accUpd = *pau;
            float newUpd = sqrtf((accUpd + cond) / (accGrad + cond)) * g;
            *pm += newUpd;
            *pag = accGrad;
            *pau = decay * accUpd + (1 - decay) * newUpd * newUpd;
        }
    }

    _vleave();
}

// For Adadelta, partial sparse source vector.
EXPORT_API(void) AddXYTranGradPX(_In_ const float * px, _In_ const int * pposY, _In_ const float * pvaluesY,
    int posMinY, int iposMinY, int iposLimY, _Inout_ float * pmat, _Inout_ float * paccGrads, _Inout_ float * paccUpdates,
    float decay, float cond, int crow, int ccol)
{
    const int * pposMin = pposY + iposMinY;
    const int * pposLim = pposY + iposLimY;
    const float * pxLim = px + crow;
    const float * py = pvaluesY - posMinY;
    float * pm0 = pmat - posMinY;
    float * pag0 = paccGrads - posMinY;
    float * pau0 = paccUpdates - posMinY;

    __m256 dec = _mm256_set1_ps(decay);
    __m256 decc = _mm256_set1_ps(1 - decay);
    __m256 c = _mm256_set1_ps(cond);
    for (; px < pxLim; px += 8, pm0 += 8 * ccol, pag0 += 8 * ccol, pau0 += 8 * ccol)
    {
        float * pm1 = pm0 + ccol;
        float * pm2 = pm1 + ccol;
        float * pm3 = pm2 + ccol;

        float * pag1 = pag0 + ccol;
        float * pag2 = pag1 + ccol;
        float * pag3 = pag2 + ccol;

        float * pau1 = pau0 + ccol;
        float * pau2 = pau1 + ccol;
        float * pau3 = pau2 + ccol;

        __m256 x1 = _mm256_load_ps(px);

        for (const int * ppos = pposMin; ppos < pposLim; ppos++)
        {
            int col1 = *ppos;
            int col2 = col1 + 4 * ccol;
            __m256 x2 = _mm256_set1_ps(py[col1]);
            __m256 ag = _mm256_setr_ps(
                pag0[col1], pag1[col1], pag2[col1], pag3[col1],
                pag0[col2], pag1[col2], pag2[col2], pag3[col2]);
            __m256 au = _mm256_setr_ps(
                pau0[col1], pau1[col1], pau2[col1], pau3[col1],
                pau0[col2], pau1[col2], pau2[col2], pau3[col2]);
            __m256 coef = _mm256_setr_ps(
                pm0[col1], pm1[col1], pm2[col1], pm3[col1],
                pm0[col2], pm1[col2], pm2[col2], pm3[col2]);
            x2 = _mm256_mul_ps(x2, x1);

            UpdateAdadelta(coef, ag, au, x2, dec, decc, c);

            __m128 t1 = _get_lo(coef);
            __m128 t2 = _get_hi(coef);
            _mm_store_ss(pm0 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pm1 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pm2 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pm3 + col1, t1);
            _mm_store_ss(pm0 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pm1 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pm2 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pm3 + col2, t2);

            t1 = _get_lo(ag);
            t2 = _get_hi(ag);
            _mm_store_ss(pag0 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pag1 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pag2 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pag3 + col1, t1);
            _mm_store_ss(pag0 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pag1 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pag2 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pag3 + col2, t2);

            t1 = _get_lo(au);
            t2 = _get_hi(au);
            _mm_store_ss(pau0 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pau1 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pau2 + col1, t1); t1 = _rotate(t1);
            _mm_store_ss(pau3 + col1, t1);
            _mm_store_ss(pau0 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pau1 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pau2 + col2, t2); t2 = _rotate(t2);
            _mm_store_ss(pau3 + col2, t2);
        }
    }

    _vleave();
}

EXPORT_API(void) ScaleX(float a, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m256 x1 = _mm256_set1_ps(a);
    for (; pd < pdLim; pd += 8)
    {
        __m256 x2 = _mm256_load_ps(pd);
        x2 = _mm256_mul_ps(x1, x2);
        _mm256_store_ps(pd, x2);
    }

    _vleave();
}

EXPORT_API(void) ScaleMaxNormX(float maxNorm, _Inout_ float * pmat, int crow, int ccol)
{
    float * pm = pmat;
    float maxNormSq = maxNorm * maxNorm;
    __m256 m = _mm256_set1_ps(maxNorm);
    for (int irow = 0; irow < crow; irow++)
    {
        __m256 rowNorm = _mm256_set1_ps(0);
        float * pms = pm;
        float * pmLim = pm + ccol;
        for (; pm < pmLim; pm += 8)
        {
            __m256 x1 = _mm256_load_ps(pm);
            x1 = _mm256_mul_ps(x1, x1);
            rowNorm = _mm256_add_ps(x1, rowNorm);
        }
        rowNorm = _mm256_hadd_ps(rowNorm, rowNorm);
        rowNorm = _mm256_hadd_ps(rowNorm, rowNorm);
        float rowNormRes = _mm_cvtss_f32(_mm_add_ss(_get_lo(rowNorm), _get_hi(rowNorm)));
        if (rowNormRes > maxNormSq)
        {
            __m256 scale = _mm256_set1_ps(rowNormRes);
#if 0
            // REVIEW: this is faster but it uses approximation so results differ significantly from CLR.
            scale = _mm256_rsqrt_ps(scale);
            scale = _mm256_mul_ps(scale, m);
#else
            scale = _mm256_sqrt_ps(scale);
            scale = _mm256_div_ps(m, scale);
#endif
            for (pm = pms; pm < pmLim; pm += 8)
            {
                __m256 x1 = _mm256_load_ps(pm);
                x1 = _mm256_mul_ps(x1, scale);
                _mm256_store_ps(pm, x1);
            }
        }
    }

    _vleave();
}

EXPORT_API(void) AddScaleX(float a, _In_ const float * ps, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m256 x1 = _mm256_set1_ps(a);
    for (; pd < pdLim; pd += 8, ps += 8)
    {
        __m256 x2 = _mm256_load_ps(ps);
        __m256 x3 = _mm256_load_ps(pd);
        x2 = _mm256_mul_ps(x1, x2);
        x3 = _mm256_add_ps(x2, x3);
        _mm256_store_ps(pd, x3);
    }

    _vleave();
}

EXPORT_API(void) AddScaleMomX(float a, _In_ const float * ps, _Inout_ float * pd, float momentum, _Inout_ float * pe, int c)
{
    float * pdLim = pd + c;

    __m256 x0 = _mm256_set1_ps(momentum);
    __m256 x1 = _mm256_set1_ps(a);
    for (; pd < pdLim; pd += 8, pe += 8, ps += 8)
    {
        __m256 x2 = _mm256_load_ps(ps);
        __m256 x3 = _mm256_load_ps(pe);
        __m256 x4 = _mm256_load_ps(pd);
        x2 = _mm256_mul_ps(x1, x2);
        x3 = _mm256_mul_ps(x0, x3);
        x3 = _mm256_add_ps(x2, x3);
        x4 = _mm256_add_ps(x3, x4);
        _mm256_store_ps(pe, x3);
        _mm256_store_ps(pd, x4);
    }

    _vleave();
}

EXPORT_API(void) AddScaleGradX(_In_ const float * ps, _Inout_ float * pd, _Inout_ float * paccGrads, _Inout_ float * paccUpdates,
    float decay, float cond, int c)
{
    float * pdLim = pd + c;

    __m256 dec = _mm256_set1_ps(decay);
    __m256 decc = _mm256_set1_ps(1 - decay);
    __m256 cnd = _mm256_set1_ps(cond);
    for (; pd < pdLim; pd += 8, ps += 8, paccGrads += 8, paccUpdates += 8)
    {
        __m256 g = _mm256_load_ps(ps);
        __m256 ag = _mm256_load_ps(paccGrads);
        __m256 au = _mm256_load_ps(paccUpdates);
        __m256 coef = _mm256_load_ps(pd);

        UpdateAdadelta(coef, ag, au, g, dec, decc, cnd);

        _mm256_store_ps(pd, coef);
        _mm256_store_ps(paccGrads, ag);
        _mm256_store_ps(paccUpdates, au);
    }

    _vleave();
}

EXPORT_API(void) AddX(_In_ const float * ps, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    for (; pd < pdLim; pd += 8, ps += 8)
    {
        __m256 x1 = _mm256_load_ps(ps);
        __m256 x2 = _mm256_load_ps(pd);
        x2 = _mm256_add_ps(x1, x2);
        _mm256_store_ps(pd, x2);
    }

    _vleave();
}

EXPORT_API(float) SumX(const float * ps, int c)
{
    const float * psLim = ps + c;

    __m256 res = _mm256_setzero_ps();
    for (; ps < psLim; ps += 8)
    {
        __m256 x1 = _mm256_load_ps(ps);
        res = _mm256_add_ps(res, x1);
    }
    res = _mm256_hadd_ps(res, res);
    res = _mm256_hadd_ps(res, res);
    __m128 r = _mm_add_ss(_get_lo(res), _get_hi(res));

    float ret = _mm_cvtss_f32(r);
    _vleave();
    return ret;
}

EXPORT_API(void) ScaleAdadeltaX(_Inout_ float * mat, _Inout_ float * accGrads, _Inout_ float * accUpdates, float decay, float cond, _In_ const float * grads, int size)
{
    float * pm = mat;
    float * pmLim = pm + size;
    float * pag = accGrads;
    float * pau = accUpdates;
    const float * pg = grads;

    __m256 dec = _mm256_set1_ps(decay);
    __m256 decc = _mm256_set1_ps(1 - decay);
    __m256 c = _mm256_set1_ps(cond);

    for (; pm + 8 <= pmLim; pm += 8, pag += 8, pau += 8, pg += 8)
    {
        __m256 g = _mm256_loadu_ps(pg);
        __m256 ag = _mm256_loadu_ps(pag);
        __m256 au = _mm256_loadu_ps(pau);
        __m256 coef = _mm256_loadu_ps(pm);

        UpdateAdadelta(coef, ag, au, g, dec, decc, c);

        _mm256_storeu_ps(pm, coef);
        _mm256_storeu_ps(pag, ag);
        _mm256_storeu_ps(pau, au);
    }

    for (; pm < pmLim; pm++, pag++, pau++, pg++)
    {
        float g = *pg;
        float accGrad = decay * *pag + (1 - decay) * g * g;
        float accUpd = *pau;
        float newUpd = sqrtf((accUpd + cond) / (accGrad + cond)) * g;
        *pm += newUpd;
        *pag = accGrad;
        *pau = decay * accUpd + (1 - decay) * newUpd * newUpd;
    }

    _vleave();
}
