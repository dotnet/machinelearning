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

#include "../Stdafx.h"
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <stdint.h>

#define _load1(ps, pi) \
    _mm_set_ss(ps[pi[0]])

#define _load4(ps, pi) \
    _mm_setr_ps(ps[pi[0]], ps[pi[1]], ps[pi[2]], ps[pi[3]])

#define _rotate(x) _mm_shuffle_ps(x, x, 0x39)

#define _rotate_reverse(x) _mm_shuffle_ps(x, x, 0x93)

#define _store1(x, pd, pi) \
    _mm_store_ss(pd + pi[0], x)

//Warning: this operation changes the value of x => do not reuse x
#define _store4(x, pd, pi) \
    _mm_store_ss(pd + pi[0], x); \
    x = _rotate(x); _mm_store_ss(pd + pi[1], x); \
    x = _rotate(x); _mm_store_ss(pd + pi[2], x); \
    x = _rotate(x); _mm_store_ss(pd + pi[3], x)

#ifndef _WIN32

typedef unsigned int DWORD; // NOTE: diff from  windows.h, for LP64 compat

// getcpuid and xmmYmmStateSupport are taken from 
// https://github.com/dotnet/coreclr/blob/b5f4d2df2e087401f2c3aab2c37021e326707915/src/vm/amd64/unixstubs.cpp#L14-L55

DWORD getcpuid(DWORD arg, unsigned char result[16])
{
	DWORD eax;
	__asm("  xor %%ecx, %%ecx\n" \
	"  cpuid\n" \
		"  mov %%eax, 0(%[result])\n" \
		"  mov %%ebx, 4(%[result])\n" \
		"  mov %%ecx, 8(%[result])\n" \
		"  mov %%edx, 12(%[result])\n" \
		: "=a"(eax) /*output in eax*/\
		: "a"(arg), [result]"r"(result) /*inputs - arg in eax, result in any register*/\
		: "rbx", "ecx", "edx", "memory" /* registers that are clobbered, *result is clobbered */
		);
	return eax;
}

DWORD xmmYmmStateSupport()
{
	DWORD eax;
	__asm("  xgetbv\n" \
	: "=a"(eax) /*output in eax*/\
		: "c"(0) /*inputs - 0 in ecx*/\
		: "edx" /* registers that are clobbered*/
		);
	// check OS has enabled both XMM and YMM state support
	return ((eax & 0x06) == 0x06) ? 1 : 0;
}

#endif

const unsigned int LeadingAlignmentMask[16] =
{
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000,
    0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
};

const unsigned int TrailingAlignmentMask[16] =
{
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF,
    0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF,
    0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
};

// Test whether Avx is available.
EXPORT_API(bool) ChkAvx()
{
#ifdef _WIN32
	int cpuInfo[4];
	__cpuid(cpuInfo, 1);

	// 28th bit of second integer of Cpu Info denotes whether the Avx is supported in CPU or not 
	// Reference https://msdn.microsoft.com/en-us/library/hskdteyh(v=vs.100).aspx
	return cpuInfo[2] & (1 << 28) || false;
#else
	unsigned char buffer[16];
	(void) getcpuid(1, buffer);

	// taken from https://github.com/dotnet/coreclr/blob/b5f4d2df2e087401f2c3aab2c37021e326707915/src/vm/codeman.cpp#L1381
	return ((buffer[11] & 0x18) == 0x18) && (xmmYmmStateSupport() == 1);
#endif
}

// Multiply matrix times vector into vector.
EXPORT_API(void) MatMulA(bool add, _In_ const float * pmat, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    const float * psLim = psrc + ccol;
    const float * pdLim = pdst + crow;
    const float * pm = pmat;
    for (float * pd = pdst; pd < pdLim; pd += 4, pm += 3 * ccol)
    {
        __m128 res0 = _mm_setzero_ps();
        __m128 res1 = res0;
        __m128 res2 = res0;
        __m128 res3 = res0;
        for (const float * ps = psrc; ps < psLim; ps += 4, pm += 4)
        {
            const float * pmTmp;
            __m128 x01 = _mm_load_ps(pmTmp = pm);
            __m128 x11 = _mm_load_ps(pmTmp += ccol);
            __m128 x21 = _mm_load_ps(pmTmp += ccol);
            __m128 x31 = _mm_load_ps(pmTmp += ccol);
            __m128 x02 = _mm_load_ps(ps);
            x01 = _mm_mul_ps(x01, x02);
            x11 = _mm_mul_ps(x11, x02);
            x21 = _mm_mul_ps(x21, x02);
            x31 = _mm_mul_ps(x31, x02);
            res0 = _mm_add_ps(res0, x01);
            res1 = _mm_add_ps(res1, x11);
            res2 = _mm_add_ps(res2, x21);
            res3 = _mm_add_ps(res3, x31);
        }

        // Add up the entries of each, with the 4 results in res0
        res0 = _mm_hadd_ps(res0, res1);
        res2 = _mm_hadd_ps(res2, res3);
        res0 = _mm_hadd_ps(res0, res2);

        if (add)
            res0 = _mm_add_ps(res0, _mm_load_ps(pd));
        _mm_store_ps(pd, res0);
    }
}

// Partial sparse source vector.
EXPORT_API(void) MatMulPA(bool add, _In_ const float * pmat, _In_ const int * pposSrc, _In_ const float * psrc,
    int posMin, int iposMin, int iposLim, _Inout_ float * pdst, int crow, int ccol)
{
    // REVIEW: For extremely sparse inputs, interchanging the loops would
    // likely be more efficient.
    const int * pposMin = pposSrc + iposMin;
    const int * pposLim = pposSrc + iposLim;
    const float * pdLim = pdst + crow;
    const float * pm0 = pmat - posMin;
    const float * ps = psrc - posMin;
    for (float * pd = pdst; pd < pdLim; pd += 4, pm0 += 4 * ccol)
    {
        const float * pm1 = pm0 + ccol;
        const float * pm2 = pm1 + ccol;
        const float * pm3 = pm2 + ccol;
        __m128 res = _mm_setzero_ps();
        for (const int * ppos = pposMin; ppos < pposLim; ppos++)
        {
            int col = *ppos;
            __m128 x1 = _mm_setr_ps(pm0[col], pm1[col], pm2[col], pm3[col]);
            __m128 x2 = _mm_set1_ps(ps[col]);
            x2 = _mm_mul_ps(x2, x1);
            res = _mm_add_ps(res, x2);
        }

        if (add)
            res = _mm_add_ps(res, _mm_load_ps(pd));
        _mm_store_ps(pd, res);
    }
}

// Sparse matrix.
EXPORT_API(void) MatMulRU(bool add, _In_ const int * pstarts, _In_ const int * pindices, _In_ const float * pcoefs,
    _In_ const float * ps, _Inout_ float * pdst, int crow)
{
    const int * pii = pstarts + 1;
    const int * pi = pindices;
    const float * pm = pcoefs;
    const float * pdLim = pdst + crow;
    for (float * pd = pdst; pd < pdLim; pd++)
    {
        const int * piLim = pindices + *pii++;

        __m128 res = _mm_setzero_ps();
        for (; pi + 4 <= piLim; pi += 4, pm += 4)
        {
            __m128 x = _mm_mul_ps(_load4(ps, pi), _mm_loadu_ps(pm));
            res = _mm_add_ps(res, x);
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
}

// Unpadded convolution.
EXPORT_API(void) MatMulCU(bool add, _In_ const int * pmprowiv, _In_ const int * pmprowcol,
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

        __m128 res = _mm_setzero_ps();
        for (; pi + 4 <= piLim; pi += 4, pm += 4)
        {
            __m128 x = _mm_mul_ps(_load4(ps, pi), _mm_loadu_ps(pm));
            res = _mm_add_ps(res, x);
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
}

// Padded convolution.
EXPORT_API(void) MatMulDU(bool add, _In_ const int * pmprowiv, _In_ const int * pmprowcol, _In_ const int * pmprowrun,
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
        __m128 res = _mm_setzero_ps();
        if (irun == 0)
        {
            // No masking needed.
            for (; pi + 4 <= piLim; pi += 4, pm += 4)
            {
                __m128 x = _mm_mul_ps(_load4(ps, pi), _mm_loadu_ps(pm));
                res = _mm_add_ps(res, x);
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
            for (; pi + 4 <= piLim; pi += 4, pm += 4, pmask += 4)
            {
                __m128 x = _mm_mul_ps(_load4(ps, pi), _mm_and_ps(_mm_loadu_ps(pmask), _mm_loadu_ps(pm)));
                res = _mm_add_ps(res, x);
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
}

// Mean pooling.
EXPORT_API(void) MeanU(bool add, _In_ const int * pmprowcol, _In_opt_ const int * pmprowindices, _In_ const int * pindices,
    _In_ const float * psrc, _Inout_ float * pdst, int crow)
{
    const int * pcol = pmprowcol;
    const float * pdLim = pdst + crow;

    if (pmprowindices == nullptr)
    {
        int size = pindices[0];
        __m128 x0 = _mm_set_ss((float)size);
        const int * piLim = pindices + 1 + size;
        for (float * pd = pdst; pd < pdLim; pd++)
        {
            const float * ps = psrc + *pcol++;
            const int * pi = pindices + 1;

            __m128 res = _mm_setzero_ps();
            for (; pi + 4 <= piLim; pi += 4)
                res = _mm_add_ps(res, _load4(ps, pi));
            for (; pi < piLim; pi++)
                res = _mm_add_ss(res, _load1(ps, pi));
            res = _mm_hadd_ps(res, res);
            res = _mm_hadd_ps(res, res);

            res = _mm_div_ss(res, x0);
            if (add)
                res = _mm_add_ss(res, _mm_set_ss(*pd));
            _mm_store_ss(pd, res);
        }
    }
    else
    {
        const int * pii = pmprowindices;
        for (float * pd = pdst; pd < pdLim; pd++)
        {
            const float * ps = psrc + *pcol++;
            int ii = *pii++;

            const int * pi = pindices + ii;
            int size = *pi++;
            const int * piLim = pi + size;
            __m128 res = _mm_setzero_ps();
            for (; pi + 4 <= piLim; pi += 4)
                res = _mm_add_ps(res, _load4(ps, pi));
            for (; pi < piLim; pi++)
                res = _mm_add_ss(res, _load1(ps, pi));
            res = _mm_hadd_ps(res, res);
            res = _mm_hadd_ps(res, res);

            res = _mm_div_ss(res, _mm_set_ss((float)size));
            if (add)
                res = _mm_add_ss(res, _mm_set_ss(*pd));
            _mm_store_ss(pd, res);
        }
    }
}

// Max pooling.
EXPORT_API(void) MaxU(bool add, _In_ const int * pmprowcol, _In_opt_ const int * pmprowindices, _In_ const int * pindices,
    _In_ const float * psrc, _Inout_ float * pdst, int crow)
{
    const int * pcol = pmprowcol;
    const float * pdLim = pdst + crow;
    __m128 min = _mm_set1_ps(-std::numeric_limits<float>::infinity());

    if (pmprowindices == nullptr)
    {
        int size = pindices[0];
        const int * piLim = pindices + 1 + size;
        for (float * pd = pdst; pd < pdLim; pd++)
        {
            const float * ps = psrc + *pcol++;
            const int * pi = pindices + 1;

            __m128 res = min;
            for (; pi + 4 <= piLim; pi += 4)
                res = _mm_max_ps(res, _load4(ps, pi));
            for (; pi < piLim; pi++)
                res = _mm_max_ss(res, _load1(ps, pi));
            __m128 x1 = _mm_shuffle_ps(res, res, 0xB1);
            res = _mm_max_ps(res, x1);
            x1 = _mm_shuffle_ps(res, res, 0x02);
            res = _mm_max_ss(res, x1);

            if (add)
                res = _mm_add_ss(res, _mm_set_ss(*pd));
            _mm_store_ss(pd, res);
        }
    }
    else
    {
        const int * pii = pmprowindices;
        for (float * pd = pdst; pd < pdLim; pd++)
        {
            const float * ps = psrc + *pcol++;
            int ii = *pii++;

            const int * pi = pindices + ii;
            int size = *pi++;
            const int * piLim = pi + size;
            __m128 res = min;
            for (; pi + 4 <= piLim; pi += 4)
                res = _mm_max_ps(res, _load4(ps, pi));
            for (; pi < piLim; pi++)
                res = _mm_max_ss(res, _load1(ps, pi));
            __m128 x1 = _mm_shuffle_ps(res, res, 0xB1);
            res = _mm_max_ps(res, x1);
            x1 = _mm_shuffle_ps(res, res, 0x02);
            res = _mm_max_ss(res, x1);

            if (add)
                res = _mm_add_ss(res, _mm_set_ss(*pd));
            _mm_store_ss(pd, res);
        }
    }
}

// REVIEW: Try out SSE/AVX after padding support is added. AVX math platform uses the same code below.
EXPORT_API(void) RespNormU(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
    _In_ const int * pmprowcol, _In_opt_ const int * pmprowindices, _In_ const int * pindices,
    _In_ const float * psrc, _Inout_ float * pdst, int crow)
{
    const int * pcol = pmprowcol;
    const float * pdLim = pdst + crow;

    if (pmprowindices == nullptr)
    {
        int size = pindices[0];
        float scale = alpha / size;
        const int * piLim = pindices + 1 + size;
        for (float * pd = pdst; pd < pdLim; pd++)
        {
            const float * ps = psrc + *pcol++;
            const int * pi = pindices + 1;
            float res = 0;
            for (; pi < piLim; pi++)
            {
                float cur = ps[*pi];
                res += cur * cur;
            }
            res = ps[0] * powf(offset + scale * res, -beta);
            *pd = add ? *pd + res : res;
        }
    }
    else
    {
        int kernelSize = pindices[0];
        const int * pii = pmprowindices;
        for (float * pd = pdst; pd < pdLim; pd++)
        {
            const float * ps = psrc + *pcol++;
            int ii = *pii++;
            const int * pi = pindices + ii;
            int size = *pi++;
            const int * piLim = pi + size;
            float res = 0;
            for (; pi < piLim; pi++)
            {
                float cur = ps[*pi];
                res += cur * cur;
            }
            int avgDenom = avgOverFullKernel ? kernelSize : size;
            res = ps[0] * powf(offset + alpha / avgDenom * res, -beta);
            *pd = add ? *pd + res : res;
        }
    }
}

EXPORT_API(void) MatMulTranA(bool add, _In_ const float * pmat, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    const float * psLim = psrc + ccol;
    const float * pdLim = pdst + crow;
    const float * pm = pmat;
    const float * ps = psrc;

    if (!add)
    {
        __m128 x01 = _mm_load_ps(ps);
        // Replicate each slot of x01 into its own register.
        __m128 x11 = _mm_shuffle_ps(x01, x01, 0x55);
        __m128 x21 = _mm_shuffle_ps(x01, x01, 0xAA);
        __m128 x31 = _mm_shuffle_ps(x01, x01, 0xFF);
        x01 = _mm_shuffle_ps(x01, x01, 0x00);
        ps += 4;
        for (float * pd = pdst; pd < pdLim; pd += 4, pm += 4)
        {
            const float * pmTmp;
            __m128 x02 = _mm_load_ps(pmTmp = pm);
            __m128 x12 = _mm_load_ps(pmTmp += crow);
            __m128 x22 = _mm_load_ps(pmTmp += crow);
            __m128 x32 = _mm_load_ps(pmTmp += crow);
            x02 = _mm_mul_ps(x01, x02);
            x12 = _mm_mul_ps(x11, x12);
            x22 = _mm_mul_ps(x21, x22);
            x32 = _mm_mul_ps(x31, x32);
            x02 = _mm_add_ps(x02, x12);
            x22 = _mm_add_ps(x22, x32);
            x02 = _mm_add_ps(x02, x22);
            _mm_store_ps(pd, x02);
        }

        pm += 3 * crow;
    }

    for (; ps < psLim; ps += 4)
    {
        __m128 x01 = _mm_load_ps(ps);
        // Replicate each slot of x01 into its own register.
        __m128 x11 = _mm_shuffle_ps(x01, x01, 0x55);
        __m128 x21 = _mm_shuffle_ps(x01, x01, 0xAA);
        __m128 x31 = _mm_shuffle_ps(x01, x01, 0xFF);
        x01 = _mm_shuffle_ps(x01, x01, 0x00);
        for (float * pd = pdst; pd < pdLim; pd += 4, pm += 4)
        {
            const float * pmTmp;
            __m128 x02 = _mm_load_ps(pmTmp = pm);
            __m128 x12 = _mm_load_ps(pmTmp += crow);
            __m128 x22 = _mm_load_ps(pmTmp += crow);
            __m128 x32 = _mm_load_ps(pmTmp += crow);
            __m128 x3 = _mm_load_ps(pd);
            x02 = _mm_mul_ps(x01, x02);
            x12 = _mm_mul_ps(x11, x12);
            x22 = _mm_mul_ps(x21, x22);
            x32 = _mm_mul_ps(x31, x32);
            x02 = _mm_add_ps(x02, x12);
            x22 = _mm_add_ps(x22, x32);
            x02 = _mm_add_ps(x02, x22);
            x3 = _mm_add_ps(x02, x3);
            _mm_store_ps(pd, x3);
        }

        pm += 3 * crow;
    }
}

// Partial sparse source vector.
EXPORT_API(void) MatMulTranPA(bool add, _In_ const float * pmat, _In_ const int * pposSrc, _In_ const float * psrc,
    int posMin, int iposMin, int iposLim, _Inout_ float * pdst, int crow)
{
    const int * ppos = pposSrc + iposMin;
    const int * pposLim = pposSrc + iposLim;
    const float * pdLim = pdst + crow;

    if (!add)
    {
        int col = *ppos++ - posMin;
        const float * pm = pmat + col * crow;
        __m128 x0 = _mm_set1_ps(psrc[col]);
        for (float * pd = pdst; pd < pdLim; pd += 4, pm += 4)
        {
            __m128 x1 = _mm_load_ps(pm);
            x1 = _mm_mul_ps(x1, x0);
            _mm_store_ps(pd, x1);
        }
    }

    // REVIEW: Should we explore unrolling the outer loop?
    for (; ppos < pposLim; ppos++)
    {
        int col = *ppos - posMin;
        __m128 x0 = _mm_set1_ps(psrc[col]);
        const float * pm = pmat + col * crow;
        for (float * pd = pdst; pd < pdLim; pd += 4, pm += 4)
        {
            __m128 x1 = _mm_load_ps(pm);
            __m128 x2 = _mm_load_ps(pd);
            x1 = _mm_mul_ps(x1, x0);
            x2 = _mm_add_ps(x2, x1);
            _mm_store_ps(pd, x2);
        }
    }
}

// Sparse matrix.
EXPORT_API(void) MatMulTranRU(bool add, _In_ const int * pstarts, _In_ const int * pindices, _In_ const float * pcoefs,
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

        __m128 x1 = _mm_set1_ps(x);
        for (; pi + 4 <= piLim; pi += 4, pm += 4)
        {
            __m128 x2 = _mm_mul_ps(x1, _mm_loadu_ps(pm));
            x2 = _mm_add_ps(x2, _load4(pd, pi));
            _store4(x2, pd, pi);
        }
        for (; pi < piLim; pi++, pm++)
        {
            __m128 x2 = _mm_mul_ss(x1, _mm_set_ss(*pm));
            x2 = _mm_add_ss(x2, _load1(pd, pi));
            _store1(x2, pd, pi);
        }
    }
}

// Unpadded convolution.
EXPORT_API(void) MatMulTranCU(bool add, _In_ const int * pmpcoliv, _In_ const int * pmpcolrow,
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
        __m128 x1 = _mm_set1_ps(x);
        for (; pi + 4 <= piLim; pm += 4, pi += 4)
        {
            __m128 x2 = _mm_mul_ps(x1, _mm_loadu_ps(pm));
            x2 = _mm_add_ps(x2, _load4(pd, pi));
            _store4(x2, pd, pi);
        }
        for (; pi < piLim; pi++, pm++)
        {
            __m128 x2 = _mm_mul_ss(x1, _mm_set_ss(*pm));
            x2 = _mm_add_ss(x2, _load1(pd, pi));
            _store1(x2, pd, pi);
        }
    }
}

// Padded convolution.
EXPORT_API(void) MatMulTranDU(bool add, _In_ const int * pmpcoliv, _In_ const int * pmpcolrow, _In_ const int * pmpcolrun,
    _In_ const int * pruns, _In_ const float * pcoefs, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    if (!add)
        memset(pdst, 0, crow * sizeof(float));

    const int * piv = pmpcoliv;
    const int * prow = pmpcolrow;
    const float * psLim = psrc + ccol;

    const int * pirun = pmpcolrun;
    for (const float * ps = psrc; ps < psLim; ps++)
    {
        const float * pm = pcoefs + *piv++;
        float * pd = pdst + *prow++;
        int irun = *pirun++;
        const int * pi = pruns + 2 + irun;
        const int * piLim = pi + pi[-1];

        float x = *ps;
        __m128 x1 = _mm_set1_ps(x);
        if (irun == 0)
        {
            // No masking needed.
            for (; pi + 4 <= piLim; pi += 4, pm += 4)
            {
                __m128 x2 = _mm_mul_ps(x1, _mm_loadu_ps(pm));
                x2 = _mm_add_ps(x2, _load4(pd, pi));
                _store4(x2, pd, pi);
            }
            for (; pi < piLim; pi++, pm++)
            {
                __m128 x2 = _mm_mul_ss(x1, _mm_set_ss(*pm));
                x2 = _mm_add_ss(x2, _load1(pd, pi));
                _store1(x2, pd, pi);
            }
        }
        else
        {
            // Need masking.
            pm += pi[-2];
            const float * pmask = reinterpret_cast<const float *>(piLim);
            for (; pi + 4 <= piLim; pi += 4, pm += 4, pmask += 4)
            {
                __m128 x2 = _mm_mul_ps(_mm_and_ps(_mm_loadu_ps(pmask), x1), _mm_loadu_ps(pm));
                x2 = _mm_add_ps(x2, _load4(pd, pi));
                _store4(x2, pd, pi);
            }
            for (; pi < piLim; pi++, pm++, pmask++)
            {
                __m128 x2 = _mm_mul_ss(_mm_and_ps(_mm_set_ss(*pmask), x1), _mm_set_ss(*pm));
                x2 = _mm_add_ss(x2, _load1(pd, pi));
                _store1(x2, pd, pi);
            }
        }
    }
}

// Mean pooling back prop.
EXPORT_API(void) MeanBackU(bool add, _In_ const int * pmpcolrow, _In_opt_ const int * pmpcolindices, _In_ const int * pindices,
    _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    if (!add)
        memset(pdst, 0, crow * sizeof(float));

    const int * prow = pmpcolrow;
    const float * psLim = psrc + ccol;
    if (pmpcolindices == nullptr)
    {
        int size = pindices[0];
        const int * piLim = pindices + 1 + size;
        for (const float * ps = psrc; ps < psLim; ps++)
        {
            float * pd = pdst + *prow++;
            const int * pi = pindices + 1;

            float x = *ps / size;
            __m128 x1 = _mm_set1_ps(x);
            for (; pi + 4 <= piLim; pi += 4)
            {
                __m128 x2 = _mm_add_ps(x1, _load4(pd, pi));
                _store4(x2, pd, pi);
            }
            for (; pi < piLim; pi++)
            {
                __m128 x2 = _mm_add_ss(x1, _load1(pd, pi));
                _store1(x2, pd, pi);
            }
        }
    }
    else
    {
        const int * pii = pmpcolindices;
        for (const float * ps = psrc; ps < psLim; ps++)
        {
            float * pd = pdst + *prow++;
            int ii = *pii++;

            const int * pi = pindices + ii;
            int size = *pi++;
            const int * piLim = pi + size;

            float x = *ps / size;
            __m128 x1 = _mm_set1_ps(x);
            for (; pi + 4 <= piLim; pi += 4)
            {
                __m128 x2 = _mm_add_ps(x1, _load4(pd, pi));
                _store4(x2, pd, pi);
            }
            for (; pi < piLim; pi++)
            {
                __m128 x2 = _mm_add_ss(x1, _load1(pd, pi));
                _store1(x2, pd, pi);
            }
        }
    }
}

// Max pooling back prop.
EXPORT_API(void) MaxBackU(bool add, _In_ const int * pmpcolrow, _In_opt_ const int * pmpcolindices, _In_ const int * pindices,
    _In_ const float * psrc, _Inout_ float * pdst, _In_ const float * pval, int crow, int ccol)
{
    if (!add)
        memset(pdst, 0, crow * sizeof(float));

    const int * prow = pmpcolrow;
    const float * psLim = psrc + ccol;
    if (pmpcolindices == nullptr)
    {
        const int * piLim = pindices + 1 + pindices[0];
        for (const float * ps = psrc; ps < psLim; ps++)
        {
            int rowBase = *prow++;
            float * pd = pdst + rowBase;
            const float * pv = pval + rowBase;
            const int * pi = pindices + 1;

            int j = *pi++;
            float m = pv[j];
            for (; pi < piLim; pi++)
            {
                if (m < pv[*pi])
                {
                    j = *pi;
                    m = pv[j];
                }
            }
            pd[j] += *ps;
        }
    }
    else
    {
        const int * pii = pmpcolindices;
        for (const float * ps = psrc; ps < psLim; ps++)
        {
            int rowBase = *prow++;
            int ii = *pii++;
            float * pd = pdst + rowBase;
            const float * pv = pval + rowBase;
            const int * pi = pindices + ii + 1;
            const int * piLim = pi + pi[-1];

            int j = *pi++;
            float m = pv[j];
            for (; pi < piLim; pi++)
            {
                if (m < pv[*pi])
                {
                    j = *pi;
                    m = pv[j];
                }
            }
            pd[j] += *ps;
        }
    }
}

// REVIEW: Try out SSE/AVX after padding support is added. AVX math platform uses the same code below.
EXPORT_API(void) RespNormBackU(bool add, float alpha, float beta, bool avgOverFullKernel, float offset,
    _In_ const int * pmpcolrow, _In_opt_ const int * pmpcolindices, _In_ const int * pindices,
    _In_ const float * perrors, _Inout_ float * perrorsPrev, _In_ const float * pvaluesPrev, int crow, int ccol)
{
    if (!add)
        memset(perrorsPrev, 0, crow * sizeof(float));

    const int * prow = pmpcolrow;
    const float * psLim = perrors + ccol;
    if (pmpcolindices == nullptr)
    {
        int size = pindices[0];
        float scale = alpha / size;
        const int * piMin = pindices + 1;
        const int * piLim = piMin + size;
        for (const float * ps = perrors; ps < psLim; ps++)
        {
            int rowBase = *prow++;
            // First compute denominator: denom = offset + scale * Sum(Xj^2)
            float denom = 0;
            const float * pv = pvaluesPrev + rowBase;

            for (const int * pi = piMin; pi < piLim; pi++)
            {
                float cur = pv[*pi];
                denom += cur * cur;
            }
            denom = offset + scale * denom;
            float denomPow = powf(denom, -beta);
            // The output.
            float y = pv[0] * denomPow;

            // The update logic:
            //     srcError(*ps) X the derivative.
            //     derivative at i wrt center point = powf(denom, -beta) - 2* scale * beta * X[i] * y / denom.
            //     derivative at i wrt other points = - 2* scale * beta * X[i] * y / denom.
            float commonUpdate = *ps * (-2 * scale * beta * y) / denom;

            float * pd = perrorsPrev + rowBase;
            for (const int * pi = piMin; pi < piLim; pi++)
                pd[*pi] += pv[*pi] * commonUpdate;

            // Additional update for the center point.
            pd[0] += *ps * denomPow;
        }
    }
    else
    {
        int kernelSize = pindices[0];
        const int * pii = pmpcolindices;
        for (const float * ps = perrors; ps < psLim; ps++)
        {
            int rowBase = *prow++;
            // First compute denominator: denom = 1 + scale * Sum(Xj^2)
            float denom = 0;
            const float * pv = pvaluesPrev + rowBase;
            int ii = *pii++;

            const int * piMin = pindices + ii;
            int size = *piMin++;
            const int * piLim = piMin + size;

            for (const int * pi = piMin; pi < piLim; pi++)
            {
                float cur = pv[*pi];
                denom += cur * cur;
            }
            float scale = alpha / (avgOverFullKernel ? kernelSize : size);
            denom = offset + scale * denom;
            float denomPow = powf(denom, -beta);
            // The output.
            float y = pv[0] * denomPow;

            // The update logic:
            //     srcError(*ps) X the derivative.
            //     derivative at i wrt center point = powf(denom, -beta) - 2* scale * beta * X[i] * y / denom.
            //     derivative at i wrt other points = - 2* scale * beta * X[i] * y / denom.
            float commonUpdate = *ps * (-2 * scale * beta * y) / denom;

            float * pd = perrorsPrev + rowBase;
            for (const int * pi = piMin; pi < piLim; pi++)
                pd[*pi] += pv[*pi] * commonUpdate;

            // Additional update for the center point.
            pd[0] += *ps * denomPow;
        }
    }
}

template <bool useDecay>
void AddXYTranACore(float a, _In_ const float * px, _In_ const float * py, _Inout_ float * pmat, int crow, int ccol, float decay)
{
    const float * pyBase = py;
    const float * pxLim = px + crow;
    const float * pyLim = py + ccol;
    float * pm = pmat;
    __m128 wd;
    if (useDecay)
        wd = _mm_set1_ps(1 - decay);
    for (; px < pxLim; px++)
    {
        float r = a * *px;
        py = pyBase;

        __m128 x1 = _mm_set1_ps(r);
        for (; py + 16 <= pyLim; py += 16, pm += 16)
        {
            __m128 x02 = _mm_load_ps(py);
            __m128 x12 = _mm_load_ps(py + 4);
            __m128 x22 = _mm_load_ps(py + 8);
            __m128 x32 = _mm_load_ps(py + 12);
            __m128 x03 = _mm_load_ps(pm);
            __m128 x13 = _mm_load_ps(pm + 4);
            __m128 x23 = _mm_load_ps(pm + 8);
            __m128 x33 = _mm_load_ps(pm + 12);
            x02 = _mm_mul_ps(x1, x02);
            x12 = _mm_mul_ps(x1, x12);
            x22 = _mm_mul_ps(x1, x22);
            x32 = _mm_mul_ps(x1, x32);
            if (useDecay)
            {
                x03 = _mm_mul_ps(wd, x03);
                x13 = _mm_mul_ps(wd, x13);
                x23 = _mm_mul_ps(wd, x23);
                x33 = _mm_mul_ps(wd, x33);
            }
            x03 = _mm_add_ps(x02, x03);
            x13 = _mm_add_ps(x12, x13);
            x23 = _mm_add_ps(x22, x23);
            x33 = _mm_add_ps(x32, x33);
            _mm_store_ps(pm, x03);
            _mm_store_ps(pm + 4, x13);
            _mm_store_ps(pm + 8, x23);
            _mm_store_ps(pm + 12, x33);
        }
        for (; py < pyLim; py += 4, pm += 4)
        {
            __m128 x02 = _mm_load_ps(py);
            __m128 x03 = _mm_load_ps(pm);
            x02 = _mm_mul_ps(x1, x02);
            if (useDecay)
                x03 = _mm_mul_ps(wd, x03);
            x03 = _mm_add_ps(x02, x03);
            _mm_store_ps(pm, x03);
        }
    }
}

EXPORT_API(void) AddXYTranA(float a, _In_ const float * px, _In_ const float * py, _Inout_ float * pmat, int crow, int ccol, float decay)
{
    if (decay == 0)
        AddXYTranACore<false>(a, px, py, pmat, crow, ccol, decay);
    else
        AddXYTranACore<true>(a, px, py, pmat, crow, ccol, decay);
}

// Partial sparse source vector.
EXPORT_API(void) AddXYTranPA(float a, _In_ const float * px, _In_ const int * pposY, _In_ const float * pvaluesY,
    int posMinY, int iposMinY, int iposLimY, _Inout_ float * pmat, int crow, int ccol)
{
#if 1
    // REVIEW: This is faster for MNIST, but the version below is faster for extremely sparse input.
    const int * pposMin = pposY + iposMinY;
    const int * pposLim = pposY + iposLimY;
    const float * pxLim = px + crow;
    float * pm0 = pmat - posMinY;
    const float * py = pvaluesY - posMinY;

    __m128 x0 = _mm_set1_ps(a);
    for (; px < pxLim; px += 4, pm0 += 4 * ccol)
    {
        float * pm1 = pm0 + ccol;
        float * pm2 = pm1 + ccol;
        float * pm3 = pm2 + ccol;

        __m128 x1 = _mm_load_ps(px);
        x1 = _mm_mul_ps(x1, x0);

        for (const int * ppos = pposMin; ppos < pposLim; ppos++)
        {
            int col = *ppos;
            __m128 x2 = _mm_set1_ps(py[col]);
            __m128 x3 = _mm_setr_ps(pm0[col], pm1[col], pm2[col], pm3[col]);
            x2 = _mm_mul_ps(x2, x1);
            x3 = _mm_add_ps(x3, x2);

            _mm_store_ss(pm0 + col, x3); x3 = _rotate(x3);
            _mm_store_ss(pm1 + col, x3); x3 = _rotate(x3);
            _mm_store_ss(pm2 + col, x3); x3 = _rotate(x3);
            _mm_store_ss(pm3 + col, x3);
        }
    }
#else
    const int * pposMin = pposY + iposMinY;
    const int * pposLim = pposY + iposLimY;
    const float * pxLim = px + crow;
    float * pm = pmat - posMinY;
    const float * py = pvaluesY - posMinY;

    __m128 x0 = _mm_set1_ps(a);
    int d1 = 1 * ccol;
    int d2 = 2 * ccol;
    int d3 = 3 * ccol;
    int d4 = 4 * ccol;
    for (const int * ppos = pposMin; ppos < pposLim; ppos++)
    {
        int col = *ppos;
        __m128 x2 = _mm_set1_ps(py[col]);
        x2 = _mm_mul_ps(x2, x0);

        float * pm0 = pm + col;
        for (const float * px0 = px; px0 < pxLim; px0 += 4, pm0 += d4)
        {
            __m128 x1 = _mm_load_ps(px0);
            __m128 x3 = _mm_setr_ps(pm0[0], pm0[d1], pm0[d2], pm0[d3]);
            x1 = _mm_mul_ps(x1, x2);
            x3 = _mm_add_ps(x3, x1);

            _mm_store_ss(pm0, x3); x3 = _rotate(x3);
            _mm_store_ss(pm0 + d1, x3); x3 = _rotate(x3);
            _mm_store_ss(pm0 + d2, x3); x3 = _rotate(x3);
            _mm_store_ss(pm0 + d3, x3);
        }
    }
#endif
}

template <bool useDecay>
void AddXYTranRUCore(float a, _In_ const float * px, _In_ const float * py,
    _In_ const int * pstarts, _In_ const int * pindices, _Inout_ float * pcoefs, int crow, float decay)
{
    const int * pii = pstarts + 1;
    const int * pi = pindices;
    float * pm = pcoefs;
    const float * pxLim = px + crow;
    __m128 wd;
    if (useDecay)
        wd = _mm_set1_ps(1 - decay);
    for (; px < pxLim; px++)
    {
        const int * piLim = pindices + *pii++;
        float r = a * *px;

        __m128 x1 = _mm_set1_ps(r);
        for (; pi + 4 <= piLim; pi += 4, pm += 4)
        {
            __m128 x2 = _mm_mul_ps(x1, _load4(py, pi));
            __m128 x3 = _mm_loadu_ps(pm);
            if (useDecay)
                x3 = _mm_mul_ps(x3, wd);
            x2 = _mm_add_ps(x2, x3);
            _mm_storeu_ps(pm, x2);
        }
        for (; pi < piLim; pi++, pm++)
            *pm = (useDecay ? (*pm * (1 - decay)) : *pm) + py[*pi] * r;
    }
}

// Sparse matrix.
EXPORT_API(void) AddXYTranRU(float a, _In_ const float * px, _In_ const float * py,
    _In_ const int * pstarts, _In_ const int * pindices, _Inout_ float * pcoefs, int crow, float decay)
{
    if (decay == 0)
        AddXYTranRUCore<false>(a, px, py, pstarts, pindices, pcoefs, crow, decay);
    else
        AddXYTranRUCore<true>(a, px, py, pstarts, pindices, pcoefs, crow, decay);
}

// Unpadded convolution.
EXPORT_API(void) AddXYTranCU(float a, _In_ const float * px, _In_ const float * py, _In_ const int * pmprowiv,
    _In_ const int * pmprowcol, _In_ const int * pruns, _Inout_ float * pcoefs, int crow)
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

        __m128 x1 = _mm_set1_ps(r);
        for (; pi + 4 <= piLim; pi += 4, pm += 4)
        {
            __m128 x2 = _mm_mul_ps(x1, _load4(ps, pi));
            x2 = _mm_add_ps(x2, _mm_loadu_ps(pm));
            _mm_storeu_ps(pm, x2);
        }
        for (; pi < piLim; pi++, pm++)
            *pm += ps[*pi] * r;
        // Update the bias.
        *pm += r;
    }
}

// Padded convolution.
EXPORT_API(void) AddXYTranDU(float a, _In_ const float * px, _In_ const float * py, _In_ const int * pmprowiv,
    _In_ const int * pmprowcol, _In_ const int * pmprowrun, _In_ const int * pruns, _Inout_ float * pcoefs, int crow)
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

        __m128 x1 = _mm_set1_ps(r);
        if (irun == 0)
        {
            // No masking needed.
            for (; pi + 4 <= piLim; pi += 4, pm += 4)
            {
                __m128 x2 = _mm_mul_ps(x1, _load4(ps, pi));
                x2 = _mm_add_ps(x2, _mm_loadu_ps(pm));
                _mm_storeu_ps(pm, x2);
            }
            for (; pi < piLim; pi++, pm++)
                *pm += ps[*pi] * r;
        }
        else
        {
            // Need masking.
            pm += pi[-2];
            const float * pmask = reinterpret_cast<const float *>(piLim);
            for (; pi + 4 <= piLim; pi += 4, pm += 4, pmask += 4)
            {
                __m128 x2 = _mm_mul_ps(_mm_and_ps(_mm_loadu_ps(pmask), x1), _load4(ps, pi));
                x2 = _mm_add_ps(x2, _mm_loadu_ps(pm));
                _mm_storeu_ps(pm, x2);
            }
            for (; pi < piLim; pi++, pm++, pmask++)
            {
                __m128 x2 = _mm_mul_ss(_mm_and_ps(_mm_set_ss(*pmask), x1), _load1(ps, pi));
                x2 = _mm_add_ss(x2, _mm_set_ss(*pm));
                _mm_store_ss(pm, x2);
            }
        }
    }
}

// With momentum.
EXPORT_API(void) AddXYTranMomA(float a, _In_ const float * px, _In_ const float * py, _Inout_ float * pmat, float momentum, _Inout_ float * pdel, int crow, int ccol)
{
    const float * pyBase = py;
    const float * pxLim = px + crow;
    const float * pyLim = py + ccol;
    float * pm = pmat;
    float * pd = pdel;

    __m128 x0 = _mm_set1_ps(momentum);
    for (; px < pxLim; px++)
    {
        float r = a * *px;

        __m128 x1 = _mm_set1_ps(r);
        for (py = pyBase; py < pyLim; pm += 4, pd += 4, py += 4)
        {
            __m128 x2 = _mm_load_ps(py);
            __m128 x3 = _mm_load_ps(pd);
            __m128 x4 = _mm_load_ps(pm);
            x2 = _mm_mul_ps(x1, x2);
            x3 = _mm_mul_ps(x0, x3);
            x3 = _mm_add_ps(x2, x3);
            x4 = _mm_add_ps(x3, x4);

            _mm_store_ps(pd, x3);
            _mm_store_ps(pm, x4);
        }
    }
}

// coef: coefs to update, ag: accumulated grads, au: accumulated updates, g: cur grads.
// Note: parameters coef, ag, au and g will be updated, do not reuse parameter g in calling code.
__forceinline void UpdateAdadelta(__m128& coef, __m128& ag, __m128& au, __m128& g, const __m128& dec, const __m128& decc, const __m128& c)
{
    __m128 x4 = _mm_mul_ps(g, g);   // x4 == g * g
    x4 = _mm_mul_ps(decc, x4);      // x4 == (1 - decay) * g * g
    ag = _mm_mul_ps(dec, ag);       // ag == decay * accG
    ag = _mm_add_ps(ag, x4);        // ag == decay * accG + (1 - decay) * g * g
    __m128 x41 = _mm_add_ps(ag, c); // x41 == ag + cond
    __m128 x51 = _mm_add_ps(au, c); // x51 == accU + cond
#if 0
    // naive version:
    x51 = _mm_div_ps(x51, x41);
    x41 = _mm_sqrt_ps(x51);         // x41 == rate
#else
    // faster (approximate) version:
    x41 = _mm_rsqrt_ps(x41);
    __m128 x52 = _mm_rsqrt_ps(x51);
    x51 = _mm_mul_ps(x51, x52);
    x41 = _mm_mul_ps(x41, x51);     // x41 == rate
#endif
    g = _mm_mul_ps(g, x41);         // g - current update
    coef = _mm_add_ps(coef, g);

    g = _mm_mul_ps(g, g);           // g  == newU * newU
    g = _mm_mul_ps(decc, g);        // g  == (1 - decay) * newU * newU
    au = _mm_mul_ps(dec, au);       // au == decay * accU
    au = _mm_add_ps(au, g);         // au == decay * accU + (1 - decay) * newU * newU
}

// For Adadelta.
EXPORT_API(void) AddXYTranGradA(_In_ const float * px, _In_ const float * py, _Inout_ float * pmat, _Inout_ float * paccGrads, _Inout_ float * paccUpdates,
    float decay, float cond, int crow, int ccol)
{
    const float * pyBase = py;
    const float * pxLim = px + crow;
    const float * pyLim = py + ccol;
    float * pm = pmat;
    float * pag = paccGrads;
    float * pau = paccUpdates;

    __m128 dec = _mm_set1_ps(decay);
    __m128 decc = _mm_set1_ps(1 - decay);
    __m128 c = _mm_set1_ps(cond);
    for (; px < pxLim; px++)
    {
        float r = *px;

        __m128 x1 = _mm_set1_ps(r);
        for (py = pyBase; py < pyLim; pm += 4, pag += 4, pau += 4, py += 4)
        {
            __m128 x2 = _mm_load_ps(py);
            __m128 ag = _mm_load_ps(pag);
            __m128 au = _mm_load_ps(pau);
            __m128 coef = _mm_load_ps(pm);
            x2 = _mm_mul_ps(x1, x2);        // x2 == g

            UpdateAdadelta(coef, ag, au, x2, dec, decc, c);

            _mm_store_ps(pm, coef);
            _mm_store_ps(pag, ag);
            _mm_store_ps(pau, au);
        }
    }
}

// For Adadelta, sparse matrix.
EXPORT_API(void) AddXYTranGradRU(_In_ const float * px, _In_ const float * py, _In_ const int * pstarts, _In_ const int * pindices,
    _Inout_ float * pcoefs, _Inout_ float * paccGrads, _Inout_ float * paccUpdates, float decay, float cond, int crow)
{
    const int * pii = pstarts + 1;
    const int * pi = pindices;
    float * pm = pcoefs;
    const float * pxLim = px + crow;
    float * pag = paccGrads;
    float * pau = paccUpdates;

    __m128 dec = _mm_set1_ps(decay);
    __m128 decc = _mm_set1_ps(1 - decay);
    __m128 c = _mm_set1_ps(cond);

    for (; px < pxLim; px++)
    {
        const int * piLim = pindices + *pii++;
        float r = *px;

        __m128 x1 = _mm_set1_ps(r);
        for (; pi + 4 <= piLim; pi += 4, pm += 4, pag += 4, pau += 4)
        {
            __m128 g = _mm_mul_ps(x1, _load4(py, pi));
            __m128 ag = _mm_loadu_ps(pag);
            __m128 au = _mm_loadu_ps(pau);
            __m128 coef = _mm_loadu_ps(pm);

            UpdateAdadelta(coef, ag, au, g, dec, decc, c);

            _mm_storeu_ps(pm, coef);
            _mm_storeu_ps(pag, ag);
            _mm_storeu_ps(pau, au);
        }

        if (pi < piLim)
        {
            size_t ctail = piLim - pi;
            __m128 g = _mm_mul_ss(_load1(py, pi++), x1);
            __m128 ag = _mm_load_ss(pag++);
            __m128 au = _mm_load_ss(pau++);
            __m128 coef = _mm_load_ss(pm++);
            for (; pi < piLim; pi++, pm++, pag++, pau++)
            {
                g = _mm_or_ps(_mm_mul_ss(_load1(py, pi), x1), _rotate(g));
                ag = _mm_or_ps(_mm_load_ss(pag), _rotate(ag));
                au = _mm_or_ps(_mm_load_ss(pau), _rotate(au));
                coef = _mm_or_ps(_mm_load_ss(pm), _rotate(coef));
            }
            UpdateAdadelta(coef, ag, au, g, dec, decc, c);
            for (int i = 0; i < ctail; i++)
            {
                _mm_store_ss(pm - i - 1, coef);
                coef = _rotate_reverse(coef);
                _mm_store_ss(pag - i - 1, ag);
                ag = _rotate_reverse(ag);
                _mm_store_ss(pau - i - 1, au);
                au = _rotate_reverse(au);
            }
        }
    }
}

// For Adadelta, partial sparse source vector.
EXPORT_API(void) AddXYTranGradPA(_In_ const float * px, _In_ const int * pposY, _In_ const float * pvaluesY,
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

    __m128 dec = _mm_set1_ps(decay);
    __m128 decc = _mm_set1_ps(1 - decay);
    __m128 c = _mm_set1_ps(cond);
    for (; px < pxLim; px += 4, pm0 += 4 * ccol, pag0 += 4 * ccol, pau0 += 4 * ccol)
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

        __m128 x1 = _mm_load_ps(px);

        for (const int * ppos = pposMin; ppos < pposLim; ppos++)
        {
            int col = *ppos;
            __m128 x2 = _mm_set1_ps(py[col]);
            __m128 ag = _mm_setr_ps(pag0[col], pag1[col], pag2[col], pag3[col]);
            __m128 au = _mm_setr_ps(pau0[col], pau1[col], pau2[col], pau3[col]);
            __m128 coef = _mm_setr_ps(pm0[col], pm1[col], pm2[col], pm3[col]);
            x2 = _mm_mul_ps(x2, x1);

            UpdateAdadelta(coef, ag, au, x2, dec, decc, c);

            _mm_store_ss(pm0 + col, coef); coef = _rotate(coef);
            _mm_store_ss(pm1 + col, coef); coef = _rotate(coef);
            _mm_store_ss(pm2 + col, coef); coef = _rotate(coef);
            _mm_store_ss(pm3 + col, coef);

            _mm_store_ss(pag0 + col, ag); ag = _rotate(ag);
            _mm_store_ss(pag1 + col, ag); ag = _rotate(ag);
            _mm_store_ss(pag2 + col, ag); ag = _rotate(ag);
            _mm_store_ss(pag3 + col, ag);

            _mm_store_ss(pau0 + col, au); au = _rotate(au);
            _mm_store_ss(pau1 + col, au); au = _rotate(au);
            _mm_store_ss(pau2 + col, au); au = _rotate(au);
            _mm_store_ss(pau3 + col, au);
        }
    }
}

// pd[i] += a
EXPORT_API(void) AddScalarU(float a, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m128 x1 = _mm_set1_ps(a);
    for (; pd + 4 <= pdLim; pd += 4)
    {
        __m128 x2 = _mm_loadu_ps(pd);
        x2 = _mm_add_ps(x2, x1);
        _mm_storeu_ps(pd, x2);
    }

    for (; pd < pdLim; pd++)
    {
        __m128 x2 = _mm_load_ss(pd);
        x2 = _mm_add_ss(x2, x1);
        _mm_store_ss(pd, x2);
    }
}

EXPORT_API(void) Scale(float a, _Inout_ float * pd, int c)
{
    __m128 x1 = _mm_set1_ps(a);
    
    if (c < 4)
    {
        switch(c)
        {
            case 3: pd[2] *= a;
            case 2: pd[1] *= a;
            case 1: pd[0] *= a;
        }
        return;           
    }

    uintptr_t address = (uintptr_t)(pd);
    uintptr_t misalignment = address % 16;
    int remainder = 0;

    if ((misalignment & 3) != 0)
    {
        // Handles cases where the data is not 32-bit aligned and we can't ever use aligned operations
        remainder = c % 4;
        
        for (const float* pEnd = pd + (c - remainder); pd < pEnd; pd += 4)
        {
            __m128 x2 = _mm_loadu_ps(pd);
            x2 = _mm_mul_ps(x1, x2);
            _mm_storeu_ps(pd, x2);
        }
    }
    else
    {
        if (misalignment != 0)
        {
            // Handle cases where the data is not 128-bit aligned by doing an unaligned read and then
            // masking any elements that will be included in the first aligned read
            misalignment >>= 2;
            misalignment = 4 - misalignment;
            
            __m128 result = _mm_loadu_ps(pd);            
            
            __m128 leadingMask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (misalignment * 4));
            __m128 trailingMask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + ((4 - misalignment) * 4));
            
            __m128 temp = _mm_and_ps(result, leadingMask);
            result = _mm_and_ps(result, trailingMask);
            
            temp = _mm_mul_ps(temp, x1);
            result = _mm_or_ps(temp, result);            
            
            _mm_storeu_ps(pd, result);
            
            pd += misalignment;
            c -= misalignment;            
        }

        if (c > 3)
        {
            // Handle all the 128-bit blocks that we can now that we have offset to an aligned address
            remainder = c % 4;
            for (const float* pEnd = pd + (c - remainder); pd < pEnd; pd += 4)
            {
                __m128 x2 = _mm_load_ps(pd);
                x2 = _mm_mul_ps(x1, x2);
                _mm_storeu_ps(pd, x2);
            }
        }
        else
        {
            // Handle the "worst-case" scenario, which is when we have 4-8 elements and the input is not
            // 128-bit aligned. This means we can't do any aligned loads and will just end up doing two
            // unaligned loads where we mask the input each time.
            remainder = c;
        }
    }

    if (remainder != 0)
    {
        // Handle any trailing elements that don't fit into a 128-bit block by moving back so that the next
        // unaligned load will read to the end of the array and then mask out any elements already processed
        
        pd -= (4 - remainder);
        __m128 result = _mm_loadu_ps(pd);            
            
        __m128 trailingMask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + (remainder * 4));
        __m128 leadingMask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + ((4 - remainder) * 4));
            
        __m128 temp = _mm_and_ps(result, trailingMask);
        result = _mm_and_ps(result, leadingMask);
            
        temp = _mm_mul_ps(temp, x1);
        result = _mm_or_ps(temp, result);            
            
        _mm_storeu_ps(pd, result);
    }
}

EXPORT_API(void) ScaleSrcU(float a, _In_ const float * ps, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m128 x1 = _mm_set1_ps(a);
    for (; pd + 4 <= pdLim; pd += 4, ps += 4)
    {
        __m128 x2 = _mm_loadu_ps(ps);
        x2 = _mm_mul_ps(x2, x1);
        _mm_storeu_ps(pd, x2);
    }

    for (; pd < pdLim; pd++, ps++)
    {
        __m128 x2 = _mm_load_ss(ps);
        x2 = _mm_mul_ss(x2, x1);
        _mm_store_ss(pd, x2);
    }
}

// pd[i] = a * (pd[i] + b)
EXPORT_API(void) ScaleAddU(float a, float b, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m128 x1 = _mm_set1_ps(a);
    __m128 x2 = _mm_set1_ps(b);
    for (; pd + 4 <= pdLim; pd += 4)
    {
        __m128 x3 = _mm_loadu_ps(pd);
        x3 = _mm_add_ps(x3, x2);
        x3 = _mm_mul_ps(x3, x1);
        _mm_storeu_ps(pd, x3);
    }

    for (; pd < pdLim; pd++)
    {
        __m128 x3 = _mm_load_ss(pd);
        x3 = _mm_add_ss(x3, x2);
        x3 = _mm_mul_ss(x3, x1);
        _mm_store_ss(pd, x3);
    }
}

EXPORT_API(void) ScaleMaxNormA(float maxNorm, _Inout_ float * pmat, int crow, int ccol)
{
    float * pm = pmat;
    float maxNormSq = maxNorm * maxNorm;
    __m128 m = _mm_set1_ps(maxNorm);
    for (int irow = 0; irow < crow; irow++)
    {
        __m128 rowNorm = _mm_set1_ps(0);
        float * pms = pm;
        float * pmLim = pm + ccol;
        for (; pm < pmLim; pm += 4)
        {
            __m128 x1 = _mm_load_ps(pm);
            x1 = _mm_mul_ps(x1, x1);
            rowNorm = _mm_add_ps(x1, rowNorm);
        }
        rowNorm = _mm_hadd_ps(rowNorm, rowNorm);
        rowNorm = _mm_hadd_ps(rowNorm, rowNorm);
        float rowNormRes = _mm_cvtss_f32(rowNorm);
        if (rowNormRes > maxNormSq)
        {
            __m128 scale = _mm_set1_ps(rowNormRes);
#if 0
            // REVIEW: this is faster but it uses approximation so results differ significantly from CLR.
            scale = _mm_rsqrt_ps(scale);
            scale = _mm_mul_ps(scale, m);
#else
            scale = _mm_sqrt_ps(scale);
            scale = _mm_div_ps(m, scale);
#endif
            for (pm = pms; pm < pmLim; pm += 4)
            {
                __m128 x1 = _mm_load_ps(pm);
                x1 = _mm_mul_ps(x1, scale);
                _mm_store_ps(pm, x1);
            }
        }
    }
}

EXPORT_API(void) ScaleMaxNormTranU(float maxNorm, _Inout_ float * pmat, int crow, int ccol)
{
    for (int icol = 0; icol < ccol; icol++)
    {
        float * pm = pmat + icol;
        float rowNorm = 0;
        for (int irow = 0; irow < crow; irow++)
        {
            rowNorm += *pm * *pm;
            pm += ccol;
        }
        if (rowNorm > maxNorm * maxNorm)
        {
            float scale = maxNorm / sqrtf(rowNorm);
            pm = pmat + icol;
            for (int irow = 0; irow < crow; irow++)
            {
                *pm *= scale;
                pm += ccol;
            }
        }
    }
}

// Sparse matrix.
EXPORT_API(void) ScaleMaxNormRU(float maxNorm, _In_ const int * pstarts, _Inout_ float * pmat, int crow)
{
    for (int irow = 0; irow < crow; irow++)
    {
        float rowNorm = 0;
        for (int idx = pstarts[irow]; idx < pstarts[irow + 1]; idx++)
        {
            rowNorm += pmat[idx] * pmat[idx];
        }
        if (rowNorm > maxNorm * maxNorm)
        {
            float scale = maxNorm / sqrtf(rowNorm);
            for (int idx = pstarts[irow]; idx < pstarts[irow + 1]; idx++)
            {
                pmat[idx] *= scale;
            }
        }
    }
}

// Convolution.
EXPORT_API(void) ScaleMaxNormCU(float maxNorm, int kernCount, int kernSize, _Inout_ float * pmat)
{
    float * pm = pmat;
    for (int irow = 0; irow < kernCount; irow++)
    {
        float rowNorm = 0;
        for (int icol = 0; icol < kernSize; icol++)
        {
            rowNorm += *pm * *pm;
            pm++;
        }
        if (rowNorm > maxNorm * maxNorm)
        {
            float scale = maxNorm / sqrtf(rowNorm);
            pm -= kernSize;
            for (int icol = 0; icol < kernSize; icol++)
            {
                *pm *= scale;
                pm++;
            }
        }
        // Skip bias.
        pm++;
    }
}

EXPORT_API(void) AddScaleA(float a, _In_ const float * ps, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m128 x1 = _mm_set1_ps(a);
    for (; pd < pdLim; pd += 4, ps += 4)
    {
        __m128 x2 = _mm_load_ps(ps);
        __m128 x3 = _mm_load_ps(pd);
        x2 = _mm_mul_ps(x1, x2);
        x3 = _mm_add_ps(x2, x3);
        _mm_store_ps(pd, x3);
    }
}

EXPORT_API(void) AddScaleU(float a, _In_ const float * ps, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m128 x1 = _mm_set1_ps(a);
    for (; pd + 4 <= pdLim; pd += 4, ps += 4)
    {
        __m128 x2 = _mm_loadu_ps(ps);
        __m128 x3 = _mm_loadu_ps(pd);
        x2 = _mm_mul_ps(x2, x1);
        x3 = _mm_add_ps(x3, x2);
        _mm_storeu_ps(pd, x3);
    }

    for (; pd < pdLim; pd++, ps++)
    {
        __m128 x2 = _mm_load_ss(ps);
        __m128 x3 = _mm_load_ss(pd);
        x2 = _mm_mul_ss(x2, x1);
        x3 = _mm_add_ss(x3, x2);
        _mm_store_ss(pd, x3);
    }
}

EXPORT_API(void) AddScaleCopyU(float a, _In_ const float * ps, _In_ const float * pd, _Inout_ float * pr, int c)
{
    float * prLim = pr + c;

    __m128 x1 = _mm_set1_ps(a);
    for (; pr + 4 <= prLim; pr += 4, pd += 4, ps += 4)
    {
        __m128 x2 = _mm_loadu_ps(ps);
        __m128 x3 = _mm_loadu_ps(pd);
        x2 = _mm_mul_ps(x2, x1);
        x3 = _mm_add_ps(x3, x2);
        _mm_storeu_ps(pr, x3);
    }

    for (; pr < prLim; pr++, pd++, ps++)
    {
        __m128 x2 = _mm_load_ss(ps);
        __m128 x3 = _mm_load_ss(pd);
        x2 = _mm_mul_ss(x2, x1);
        x3 = _mm_add_ss(x3, x2);
        _mm_store_ss(pr, x3);
    }
}

EXPORT_API(void) AddScaleSU(float a, _In_ const float * ps, _In_ const int * pi, _Inout_ float * pd, int c)
{
    const int * piLim = pi + c;

    __m128 x1 = _mm_set1_ps(a);
    for (; pi + 4 <= piLim; pi += 4, ps += 4)
    {
        __m128 x2 = _mm_loadu_ps(ps);
        __m128 x3 = _load4(pd, pi);
        x2 = _mm_mul_ps(x2, x1);
        x3 = _mm_add_ps(x3, x2);
        _store4(x3, pd, pi);
    }

    for (; pi < piLim; pi++, ps++)
        pd[*pi] += a * *ps;
}

EXPORT_API(void) AddScaleMomA(float a, _In_ const float * ps, _Inout_ float * pd, float momentum, _Inout_ float * pe, int c)
{
    float * pdLim = pd + c;

    __m128 x0 = _mm_set1_ps(momentum);
    __m128 x1 = _mm_set1_ps(a);
    for (; pd < pdLim; pd += 4, pe += 4, ps += 4)
    {
        __m128 x2 = _mm_load_ps(ps);
        __m128 x3 = _mm_load_ps(pe);
        __m128 x4 = _mm_load_ps(pd);
        x2 = _mm_mul_ps(x1, x2);
        x3 = _mm_mul_ps(x0, x3);
        x3 = _mm_add_ps(x2, x3);
        x4 = _mm_add_ps(x3, x4);
        _mm_store_ps(pe, x3);
        _mm_store_ps(pd, x4);
    }
}

EXPORT_API(void) AddScaleGradA(_In_ const float * ps, _Inout_ float * pd, _Inout_ float * paccGrads, _Inout_ float * paccUpdates,
    float decay, float cond, int c)
{
    float * pdLim = pd + c;

    __m128 dec = _mm_set1_ps(decay);
    __m128 decc = _mm_set1_ps(1 - decay);
    __m128 cnd = _mm_set1_ps(cond);
    for (; pd < pdLim; pd += 4, ps += 4, paccGrads += 4, paccUpdates += 4)
    {
        __m128 g = _mm_load_ps(ps);
        __m128 ag = _mm_load_ps(paccGrads);
        __m128 au = _mm_load_ps(paccUpdates);
        __m128 coef = _mm_load_ps(pd);

        UpdateAdadelta(coef, ag, au, g, dec, decc, cnd);

        _mm_store_ps(pd, coef);
        _mm_store_ps(paccGrads, ag);
        _mm_store_ps(paccUpdates, au);
    }
}

EXPORT_API(void) AddScaleMultiA(int count, _In_ const float * ps, _Inout_ float * pd, _Inout_ float * paccGrads, _Inout_ float * paccUpdates,
    float decay, float cond, int size)
{
    if (1 == count)
        AddScaleGradA(ps, pd, paccGrads, paccUpdates, decay, cond, size);
    else
    {
        float * pdLim = pd + size;

        __m128 dec = _mm_set1_ps(decay);
        __m128 decc = _mm_set1_ps(1 - decay);
        __m128 cnd = _mm_set1_ps(cond);
        for (; pd < pdLim; pd += 4, ps += 4, paccGrads += 4, paccUpdates += 4)
        {
            __m128 g = _mm_set1_ps(0);
            const float * ps1 = ps;
            // REVIEW: unroll?
            for (int i = 0; i < count; i++, ps1 += size)
            {
                __m128 x1 = _mm_load_ps(ps1);
                g = _mm_add_ps(x1, g);
            }
            __m128 ag = _mm_load_ps(paccGrads);
            __m128 au = _mm_load_ps(paccUpdates);
            __m128 coef = _mm_load_ps(pd);

            UpdateAdadelta(coef, ag, au, g, dec, decc, cnd);

            _mm_store_ps(pd, coef);
            _mm_store_ps(paccGrads, ag);
            _mm_store_ps(paccUpdates, au);
        }
    }
}

EXPORT_API(void) AddA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    for (; pd < pdLim; pd += 4, ps += 4)
    {
        __m128 x1 = _mm_load_ps(ps);
        __m128 x2 = _mm_load_ps(pd);
        x2 = _mm_add_ps(x1, x2);
        _mm_store_ps(pd, x2);
    }
}

EXPORT_API(void) AddU(_In_ const float * ps, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    for (; pd + 4 <= pdLim; pd += 4, ps += 4)
    {
        __m128 x1 = _mm_loadu_ps(ps);
        __m128 x2 = _mm_loadu_ps(pd);
        x2 = _mm_add_ps(x1, x2);
        _mm_storeu_ps(pd, x2);
    }

    for (; pd < pdLim; pd++, ps++)
    {
        __m128 x1 = _mm_load_ss(ps);
        __m128 x2 = _mm_load_ss(pd);
        x2 = _mm_add_ps(x1, x2);
        _mm_store_ss(pd, x2);
    }
}

EXPORT_API(void) AddSU(_In_ const float * ps, _In_ const int * pi, _Inout_ float * pd, int c)
{
    const int * piLim = pi + c;

    for (; pi + 4 <= piLim; pi += 4, ps += 4)
    {
        __m128 x1 = _load4(pd, pi);
        __m128 x2 = _mm_loadu_ps(ps);
        x1 = _mm_add_ps(x1, x2);
        _store4(x1, pd, pi);
    }

    for (; pi < piLim; pi++, ps++)
        pd[*pi] += *ps;
}

EXPORT_API(void) MulElementWiseU(_In_ const float * ps1, _In_ const float * ps2, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    for (; pd + 4 <= pdLim; pd += 4, ps1 += 4, ps2 += 4)
    {
        __m128 x1 = _mm_loadu_ps(ps1);
        __m128 x2 = _mm_loadu_ps(ps2);
        x2 = _mm_mul_ps(x1, x2);
        _mm_storeu_ps(pd, x2);
    }

    for (; pd < pdLim; pd++, ps1++, ps2++)
    {
        __m128 x1 = _mm_load_ss(ps1);
        __m128 x2 = _mm_load_ss(ps2);
        x2 = _mm_mul_ps(x1, x2);
        _mm_store_ss(pd, x2);
    }
}

EXPORT_API(void) MulElementWiseSU(_In_ const float * ps1, _In_ const float * ps2, _In_ const int * pi, _Inout_ float * pd, int c)
{
    const int * piLim = pi + c;

    for (; pi + 4 <= piLim; pi += 4)
    {
        __m128 x1 = _load4(ps1, pi);
        __m128 x2 = _load4(ps2, pi);
        x2 = _mm_mul_ps(x1, x2);
        _store4(x2, pd, pi);
    }

    for (; pi < piLim; pi++)   
        pd[*pi] = ps1[*pi] * ps2[*pi];    
}

EXPORT_API(float) SumA(const float * ps, int c)
{
    const float * psLim = ps + c;

    __m128 res = _mm_setzero_ps();
    for (; ps < psLim; ps += 4)
        res = _mm_add_ps(res, _mm_load_ps(ps));

    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) SumU(const float * ps, int c)
{
    const float * psLim = ps + c;

    __m128 res = _mm_setzero_ps();
    for (; ps + 4 <= psLim; ps += 4)
        res = _mm_add_ps(res, _mm_loadu_ps(ps));

    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);

    for (; ps < psLim; ps++)
        res = _mm_add_ss(res, _mm_load_ss(ps));

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) SumSqU(const float * ps, int c)
{
    const float * psLim = ps + c;

    __m128 res = _mm_setzero_ps();
    for (; ps + 4 <= psLim; ps += 4)
    {
        __m128 x = _mm_loadu_ps(ps);
        res = _mm_add_ps(res, _mm_mul_ps(x, x));
    }

    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);

    for (; ps < psLim; ps++)
    {
        __m128 x = _mm_load_ss(ps);
        res = _mm_add_ss(res, _mm_mul_ss(x, x));
    }

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) SumSqDiffU(float mean, const float * ps, int c)
{
    const float * psLim = ps + c;

    __m128 res = _mm_setzero_ps();
    __m128 m = _mm_set1_ps(mean);
    for (; ps + 4 <= psLim; ps += 4)
    {
        __m128 x = _mm_loadu_ps(ps);
        x = _mm_sub_ps(x, m);
        res = _mm_add_ps(res, _mm_mul_ps(x, x));
    }

    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);

    for (; ps < psLim; ps++)
    {
        __m128 x = _mm_load_ss(ps);
        x = _mm_sub_ss(x, m);
        res = _mm_add_ss(res, _mm_mul_ss(x, x));
    }

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) SumAbsU(const float * ps, int c)
{
    const float * psLim = ps + c;

    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    __m128 res = _mm_setzero_ps();
    for (; ps + 4 <= psLim; ps += 4)
        res = _mm_add_ps(res, _mm_and_ps(_mm_loadu_ps(ps), mask));

    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);

    for (; ps < psLim; ps++)
        res = _mm_add_ss(res, _mm_and_ps(_mm_load_ss(ps), mask));

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) SumAbsDiffU(float mean, const float * ps, int c)
{
    const float * psLim = ps + c;

    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    __m128 res = _mm_setzero_ps();
    __m128 m = _mm_set1_ps(mean);
    for (; ps + 4 <= psLim; ps += 4)
    {
        __m128 x = _mm_loadu_ps(ps);
        x = _mm_sub_ps(x, m);
        res = _mm_add_ps(res, _mm_and_ps(x, mask));
    }

    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);

    for (; ps < psLim; ps++)
    {
        __m128 x = _mm_load_ss(ps);
        x = _mm_sub_ss(x, m);
        res = _mm_add_ss(res, _mm_and_ps(x, mask));
    }

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) MaxAbsU(const float * ps, int c)
{
    const float * psLim = ps + c;

    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    __m128 res = _mm_setzero_ps();
    for (; ps + 4 <= psLim; ps += 4)
        res = _mm_max_ps(res, _mm_and_ps(_mm_loadu_ps(ps), mask));

    __m128 x1 = _mm_shuffle_ps(res, res, 0xB1);
    res = _mm_max_ps(res, x1);
    x1 = _mm_shuffle_ps(res, res, 0x02);
    res = _mm_max_ss(res, x1);

    for (; ps < psLim; ps++)
        res = _mm_max_ss(res, _mm_and_ps(_mm_load_ss(ps), mask));

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) MaxAbsDiffU(float mean, const float * ps, int c)
{
    const float * psLim = ps + c;

    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    __m128 res = _mm_setzero_ps();
    __m128 m = _mm_set1_ps(mean);
    for (; ps + 4 <= psLim; ps += 4)
    {
        __m128 x = _mm_loadu_ps(ps);
        x = _mm_sub_ps(x, m);
        res = _mm_max_ps(res, _mm_and_ps(x, mask));
    }

    __m128 x1 = _mm_shuffle_ps(res, res, 0xB1);
    res = _mm_max_ps(res, x1);
    x1 = _mm_shuffle_ps(res, res, 0x02);
    res = _mm_max_ss(res, x1);

    for (; ps < psLim; ps++)
    {
        __m128 x = _mm_load_ss(ps);
        x = _mm_sub_ss(x, m);
        res = _mm_max_ss(res, _mm_and_ps(x, mask));
    }

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) DotU(const float * pa, const float * pb, int c)
{
    const float * paLim = pa + c;

    __m128 res = _mm_setzero_ps();
    for (; pa + 4 <= paLim; pa += 4, pb += 4)
        res = _mm_add_ps(res, _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)));

    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);

    for (; pa < paLim; pa++, pb++)
        res = _mm_add_ss(res, _mm_mul_ss(_mm_load_ss(pa), _mm_load_ss(pb)));

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) DotSU(const float * pa, const float * pb, const int * pi, int c)
{
    const int * piLim = pi + c;

    __m128 res = _mm_setzero_ps();
    for (; pi + 4 <= piLim; pi += 4, pb += 4)
    {
        __m128 x = _mm_mul_ps(_load4(pa, pi), _mm_loadu_ps(pb));
        res = _mm_add_ps(res, x);
    }

    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);

    for (; pi < piLim; pi++, pb++)
    {
        __m128 x = _mm_mul_ss(_load1(pa, pi), _mm_load_ss(pb));
        res = _mm_add_ss(res, x);
    }

    return _mm_cvtss_f32(res);
}

EXPORT_API(float) Dist2(const float * px, const float * py, int c)
{
    const float * pxLim = px + c;
    __m128 norm2_4 = _mm_setzero_ps();
    for (; px + 4 <= pxLim; px += 4, py += 4)
    {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(px), _mm_loadu_ps(py));
        norm2_4 = _mm_add_ps(norm2_4, _mm_mul_ps(d, d));
    }
    norm2_4 = _mm_hadd_ps(norm2_4, norm2_4);
    norm2_4 = _mm_hadd_ps(norm2_4, norm2_4);

    float norm2 = _mm_cvtss_f32(norm2_4);
    for (; px < pxLim; ++px, ++py)
    {
        float d = *px - *py;
        norm2 += d * d;
    }

    return norm2;
}

// This is modeled after double-based SSE code

// 1 / ln(2).
const float RecipLn2 = (float)1.44269504088896340735992468100;

// Used for computing a 4th degree polynomial approximation of e^x.
const float Coef1 = (float)0.013555747234814917704030793;
const float Coef2 = (float)0.065588116243247810171479524;
const float Coef3 = (float)0.3069678791803394491901401;

const float ExpInf = 128;
const int ExpBias = 127;
const int ExpShift = 23;

float ExpFast(float arg)
{
    bool neg = false;
    if (arg < 0)
    {
        arg = -arg;
        neg = true;
    }

    arg *= RecipLn2;
    if (arg >= ExpInf)
        return neg ? 0.0f : std::numeric_limits<float>::infinity();

    int exp = (int)arg;
    arg -= exp;
    exp += ExpBias;
    exp <<= ExpShift;

    float res = (1 + arg) + (arg - 1) * arg * ((Coef1 * arg + Coef2) * arg + Coef3);
    res *= *(float *)&exp;

    if (neg)
        res = 1 / res;
    return res;
}

// Implements a fast approximation of sigmoid/tanh.
template<bool isTanh>
void ApplySigmoidCoreA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m128 cSign = _mm_set1_ps(-0.0f);
    __m128 cZero = _mm_set1_ps(0.0f);
    __m128 cOne = _mm_set1_ps(1.0f);

    __m128 cMax = _mm_set1_ps(ExpInf);
    __m128i cBias = _mm_set1_epi32(ExpBias);
    __m128 c0 = _mm_set1_ps(RecipLn2);
    __m128 c1 = _mm_set1_ps(Coef1);
    __m128 c2 = _mm_set1_ps(Coef2);
    __m128 c3 = _mm_set1_ps(Coef3);

    if (isTanh)
        c0 = _mm_add_ps(c0, c0);

    for (; pd < pdLim; ps += 4, pd += 4)
    {
        // Get the argument, capture its sign and take its absolute value.
        __m128 xArg = _mm_load_ps(ps);
        // maskNaN is set to zero if xArg is not NaN and set equal to xArg otherwise.
        __m128 maskNaN = _mm_and_ps(_mm_cmpneq_ps(xArg, xArg), xArg);
        __m128 xSign = _mm_and_ps(xArg, cSign);
        xArg = _mm_xor_ps(xArg, xSign);

        // Multiply by 1/ln(2) and check for out of bounds.
        xArg = _mm_mul_ps(xArg, c0);
        __m128 xGood = _mm_cmplt_ps(xArg, cMax);
        xArg = _mm_and_ps(xArg, xGood);

        // Get the integer and fractional parts.
        __m128i xInt = _mm_cvttps_epi32(xArg);
        xArg = _mm_sub_ps(xArg, _mm_cvtepi32_ps(xInt));

        // Add the exponent bias to xInt, then convert to a floating point
        // power of two by shifting past the mantissa bits.
        xInt = _mm_add_epi32(xInt, cBias);
        xInt = _mm_slli_epi32(xInt, ExpShift);

        // Approximate 2 raised to the fractional part.
        // (1 + f) + (f - 1) * f * ((c1 * f + c2) * f + c3)

        // x1 = (c1 * f + c2) * f + c3
        __m128 x1 = _mm_mul_ps(c1, xArg);
        x1 = _mm_add_ps(x1, c2);
        x1 = _mm_mul_ps(x1, xArg);
        x1 = _mm_add_ps(x1, c3);

        // x2 = f * (f - 1)
        __m128 x2 = _mm_sub_ps(xArg, cOne);
        x2 = _mm_mul_ps(xArg, x2);

        // Add (1 + f). Note that for tanh, we only add f, so we are approximating
        // 2^f - 1. This is necessary to preserve precision near zero. In particular,
        // near zero, tanh(x) ~ x.
        x1 = _mm_mul_ps(x2, x1);
        if (!isTanh)
            xArg = _mm_add_ps(xArg, cOne);
        x1 = _mm_add_ps(xArg, x1);

        // Multiply by 2^n, where n is the integer part.
        __m128 x3 = _mm_castsi128_ps(xInt);
        x1 = _mm_mul_ps(x1, x3);

        if (!isTanh)
        {
            // Add 1, and take the reciprocal.
            x1 = _mm_add_ps(x1, cOne);
            x1 = _mm_div_ps(cOne, x1);

            // Deal with out of bounds.
            x1 = _mm_and_ps(x1, xGood);
            // If the input was NaN, xGood is zero, so x1 is zero. So can simply or in maskNaN.
            x1 = _mm_or_ps(x1, maskNaN);

            // Deal with the sign. Set:
            // * x2 =     x1 if xSign is -0 (0x80000000)
            // * x2 = 1 - x1 if xSign is +0 (0x00000000).
            x1 = _mm_or_ps(x1, xSign);
            x2 = _mm_or_ps(xSign, cOne);
            x2 = _mm_max_ps(x2, cZero);
            x2 = _mm_sub_ps(x2, x1);
        }
        else
        {
            // [2^n(2^f - 1) + (2^n - 1)] / [2^n(2^f - 1) + (2^n + 1)]
            x2 = _mm_add_ps(x1, _mm_sub_ps(x3, cOne));
            x1 = _mm_add_ps(x1, _mm_add_ps(x3, cOne));
            x2 = _mm_div_ps(x2, x1);

            // Deal with out of bounds: x2 = (x2 & xGood) | ((1 + maskNaN) & ~xGood)
            x2 = _mm_and_ps(x2, xGood);
            x1 = _mm_andnot_ps(xGood, _mm_add_ps(maskNaN, cOne));
            x2 = _mm_or_ps(x2, x1);

            // Deal with the sign.
            x2 = _mm_or_ps(x2, xSign);
        }

        _mm_store_ps(pd, x2);
    }

    // If we overshot, back fill with zero! Since tanh(0) = 0, we only need to do this for sigmoid.
    if (!isTanh)
    {
        while (pd > pdLim)
            *--pd = 0.0f;
    }
}

EXPORT_API(void) ApplySigmoidA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    ApplySigmoidCoreA<false>(ps, pd, c);
}

EXPORT_API(void) ApplySoftMaxU(_In_ const float * ps, _Inout_ float * pd, int c)
{
    // REVIEW: Use SSE - do 4 at a time.

    const float * psLim = ps + c;

    // Compute max output.
    float maxOut = -std::numeric_limits<float>::infinity();
    for (const float * p = ps; p < psLim; p++)
    {
        float v = *p;
        if (maxOut < v)
            maxOut = v;
    }

    // Compute exp and sum.
    float sum = 0;
    const float * p = ps;
    for (float * q = pd; p < psLim; p++, q++)
    {
        float v = ExpFast(*p - maxOut);
        *q = v;
        sum += v;
    }

    // Normalize.
    for (float * q = pd; q < pd + c; q++)
        *q /= sum;
}

EXPORT_API(void) ApplyRectifiedLinearA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    const float * psLim = ps + c;

    __m128 cZero = _mm_set1_ps(0.0f);
    for (; ps < psLim; ps += 4, pd += 4)
    {
        __m128 x1 = _mm_load_ps(ps);
        x1 = _mm_max_ps(x1, cZero);
        _mm_store_ps(pd, x1);
    }
}

EXPORT_API(void) ApplySquareA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    const float * psLim = ps + c;

    for (; ps < psLim; ps += 4, pd += 4)
    {
        __m128 x1 = _mm_load_ps(ps);
        x1 = _mm_mul_ps(x1, x1);
        _mm_store_ps(pd, x1);
    }
}

EXPORT_API(void) ApplySqrtA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    const float * psLim = ps + c;

    __m128 cZero = _mm_set1_ps(0.0f);
    for (; ps < psLim; ps += 4, pd += 4)
    {
        __m128 x1 = _mm_load_ps(ps);
        x1 = _mm_max_ps(x1, cZero);
        x1 = _mm_sqrt_ps(x1);
        _mm_store_ps(pd, x1);
    }
}

EXPORT_API(void) ApplySoftRectifiedLinearU(_In_ const float * ps, _Inout_ float * pd, int c)
{
    const float * psLim = ps + c;

    // Apply: f(x) = log(1 + e^x). To avoid overflow for large x, we use the identity: f(x) = x + f(-x).
    // REVIEW: Should we implement a "LogFast"?
    // REVIEW: Do 4 at a time.
    const float * p = ps;
    for (float * q = pd; p < psLim; p++, q++)
    {
        float x = *p;
        if (x > 0)
            *q = x + log(1 + ExpFast(-x));
        else
            *q = log(1 + ExpFast(x));
    }
}

EXPORT_API(void) ApplyAbsA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    const float * psLim = ps + c;

    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    for (; ps < psLim; ps += 4, pd += 4)
    {
        __m128 x1 = _mm_load_ps(ps);
        x1 = _mm_and_ps(x1, mask);
        _mm_store_ps(pd, x1);
    }
}

EXPORT_API(void) ApplyTanhA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    ApplySigmoidCoreA<true>(ps, pd, c);
}

EXPORT_API(void) ApplyBoundedRectifiedLinearA(_In_ const float * ps, _Inout_ float * pd, int c)
{
    const float * psLim = ps + c;

    __m128 cZero = _mm_set1_ps(0.0f);
    __m128 cOne = _mm_set1_ps(1.0f);
    for (; ps < psLim; ps += 4, pd += 4)
    {
        __m128 x1 = _mm_load_ps(ps);
        x1 = _mm_max_ps(x1, cZero);
        x1 = _mm_min_ps(x1, cOne);
        _mm_store_ps(pd, x1);
    }
}

EXPORT_API(void) ApplySigmoidDerivativeA(_In_ const float * pv, _Inout_ float * pg, int c)
{
    float * pgLim = pg + c;

    // pg[i] *= pv[i] * (1 - pv[i])
    __m128 cOne = _mm_set1_ps(1.0f);
    for (; pg < pgLim; pg += 4, pv += 4)
    {
        __m128 x1 = _mm_load_ps(pv);
        __m128 x2 = _mm_load_ps(pg);
        __m128 x3 = _mm_sub_ps(cOne, x1);
        x1 = _mm_mul_ps(x1, x3);
        x2 = _mm_mul_ps(x2, x1);
        _mm_store_ps(pg, x2);
    }
}

EXPORT_API(void) ApplyRectifiedLinearDerivativeA(_In_ const float * pv, _Inout_ float * pg, int c)
{
    float * pgLim = pg + c;

    __m128 cZero = _mm_set1_ps(0.0f);
    for (; pg < pgLim; pg += 4, pv += 4)
    {
        __m128 x1 = _mm_load_ps(pv);
        __m128 x2 = _mm_load_ps(pg);
        x1 = _mm_cmpgt_ps(x1, cZero);
        x2 = _mm_and_ps(x2, x1);
        _mm_store_ps(pg, x2);
    }
}

EXPORT_API(void) ApplySquareDerivativeA(_In_ const float * px, _In_opt_ const float * py, _Inout_ float * pg, int c, bool drop)
{
    float * pgLim = pg + c;

    if (drop)
    {
        __m128 cZero = _mm_set1_ps(0.0f);
        for (; pg < pgLim; pg += 4, px += 4, py += 4)
        {
            __m128 x0 = _mm_cmpgt_ps(_mm_load_ps(py), cZero);
            __m128 x1 = _mm_load_ps(px);
            __m128 x2 = _mm_load_ps(pg);
            x1 = _mm_add_ps(x1, x1);
            x2 = _mm_mul_ps(x2, x1);
            x2 = _mm_and_ps(x2, x0);
            _mm_store_ps(pg, x2);
        }
    }
    else
    {
        for (; pg < pgLim; pg += 4, px += 4)
        {
            __m128 x1 = _mm_load_ps(px);
            __m128 x2 = _mm_load_ps(pg);
            x1 = _mm_add_ps(x1, x1);
            x2 = _mm_mul_ps(x2, x1);
            _mm_store_ps(pg, x2);
        }
    }
}

EXPORT_API(void) ApplySqrtDerivativeA(_In_ const float * pv, _Inout_ float * pg, int c)
{
    float * pgLim = pg + c;
    static const float smallValue = 1e-10F;

    __m128 cZero = _mm_set1_ps(0.0f);
    __m128 cSmall = _mm_set1_ps(smallValue);
    for (; pg < pgLim; pg += 4, pv += 4)
    {
        __m128 x1 = _mm_load_ps(pv);
        __m128 x2 = _mm_load_ps(pg);
        __m128 x3 = _mm_cmpgt_ps(x1, cZero);
        x1 = _mm_max_ps(x1, cSmall);
        x1 = _mm_add_ps(x1, x1);
        x2 = _mm_and_ps(x2, x3);
        x2 = _mm_div_ps(x2, x1);
        _mm_store_ps(pg, x2);
    }
}

EXPORT_API(void) ApplySoftRectifiedLinearDerivativeU(_In_opt_ const float * px, _In_ const float * py, _Inout_ float * pg, int c)
{
    UNUSED(px);

    float * pgLim = pg + c;

    // Use the identity: y' = 1 - e^(-y). This has a few nice properties:
    // * If x is large enough that x == y (after rounding), we'll compute y' as 1.
    // * If x is small enough that y == 0 (after rounding), we'll compute y' as 0.
    // * If y is zero because of drop out, we'll compute y' as 0.
    // REVIEW: Do 4 at a time.
    for (; pg < pgLim; pg++, py++)
        *pg *= 1 - ExpFast(-*py);
}

EXPORT_API(void) ApplyAbsDerivativeA(_In_ const float * px, _In_opt_ const float * py, _Inout_ float * pg, int c, bool drop)
{
    float * pgLim = pg + c;

    __m128 cZero = _mm_set1_ps(0.0f);
    __m128 cSign = _mm_set1_ps(-0.0f);
    if (drop)
    {
        for (; pg < pgLim; pg += 4, px += 4, py += 4)
        {
            __m128 x1 = _mm_and_ps(_mm_load_ps(px), cSign);
            __m128 x2 = _mm_cmpgt_ps(_mm_load_ps(py), cZero);
            __m128 x3 = _mm_load_ps(pg);
            x3 = _mm_xor_ps(x3, x1);
            x3 = _mm_and_ps(x3, x2);
            _mm_store_ps(pg, x3);
        }
    }
    else
    {
        for (; pg < pgLim; pg += 4, px += 4)
        {
            __m128 x0 = _mm_load_ps(px);
            __m128 x1 = _mm_and_ps(x0, cSign);
            __m128 x2 = _mm_cmpneq_ps(x0, cZero);
            __m128 x3 = _mm_load_ps(pg);
            x3 = _mm_xor_ps(x3, x1);
            x3 = _mm_and_ps(x3, x2);
            _mm_store_ps(pg, x3);
        }
    }
}

EXPORT_API(void) ApplyTanhDerivativeA(_In_ const float * pv, _Inout_ float * pg, int c)
{
    float * pgLim = pg + c;

    // pg[i] *= 1 - pv[i] * pv[i]
    __m128 cOne = _mm_set1_ps(1.0f);
    for (; pg < pgLim; pg += 4, pv += 4)
    {
        __m128 x1 = _mm_load_ps(pv);
        __m128 x2 = _mm_load_ps(pg);
        x1 = _mm_mul_ps(x1, x1);
        x1 = _mm_sub_ps(cOne, x1);
        x2 = _mm_mul_ps(x2, x1);
        _mm_store_ps(pg, x2);
    }
}

EXPORT_API(void) ApplyBoundedRectifiedLinearDerivativeA(_In_ const float * pv, _Inout_ float * pg, int c)
{
    float * pgLim = pg + c;

    __m128 cZero = _mm_set1_ps(0.0f);
    __m128 cOne = _mm_set1_ps(1.0f);
    for (; pg < pgLim; pg += 4, pv += 4)
    {
        __m128 x1 = _mm_load_ps(pv);
        __m128 x2 = _mm_load_ps(pg);
        x2 = _mm_and_ps(x2, _mm_cmpgt_ps(x1, cZero));
        x2 = _mm_and_ps(x2, _mm_cmplt_ps(x1, cOne));
        _mm_store_ps(pg, x2);
    }
}

EXPORT_API(void) ZeroItemsU(_Inout_ float * pd, int c, _In_ const int * pindices, int cindices)
{
    DEBUG_ONLY(c);
    for (int i = 0; i < cindices; ++i)
    {
        int iv = pindices[i];
        assert(0 <= iv && iv < c);
        pd[iv] = 0;
    }
}

EXPORT_API(void) ZeroMatrixItemsCore(_Inout_ float * pd, int c, int ccol, int cfltRow, _In_ const int * pindices, int cindices)
{
    DEBUG_ONLY(c);
    int ivLogMin = 0;
    int ivLogLim = ccol;
    int ivPhyMin = 0;
    for (int i = 0; i < cindices; ++i)
    {
        int iv = pindices[i];
        assert(0 <= iv && iv < c);

        int col = iv - ivLogMin;
        if ((unsigned int)col >= (unsigned int)ccol)
        {
            assert(ivLogMin > iv || iv >= ivLogLim);
            int row = iv / ccol;
            ivLogMin = row * ccol;
            ivLogLim = ivLogMin + ccol;
            ivPhyMin = row * cfltRow;
            assert(ivLogMin <= iv && iv < ivLogLim);
            col = iv - ivLogMin;
        }
        pd[ivPhyMin + col] = 0;
    }
}

EXPORT_API(void) SdcaL1UpdateU(float primalUpdate, _In_ const float * ps, float threshold, _Inout_ float *pd1, _Inout_ float * pd2, int c)
{
    const float * psLim = ps + c;

    __m128 xPrimal = _mm_set1_ps(primalUpdate);

    __m128 signMask = _mm_set1_ps(-0.0f); // 1000 0000 ...
    __m128 xThreshold = _mm_set1_ps(threshold);
    for (; ps + 4 <= psLim; ps += 4, pd1 += 4, pd2 += 4)
    {
        __m128 xs = _mm_loadu_ps(ps);
        __m128 xd1 = _mm_loadu_ps(pd1);
        xd1 = _mm_add_ps(xd1, _mm_mul_ps(xs, xPrimal));
        _mm_storeu_ps(pd1, xd1);

        __m128 xSign = _mm_and_ps(xd1, signMask); // result = 10000... if xd1 is negative or 00000 otherwise
        __m128 xd1Abs = _mm_xor_ps(xd1, xSign);
        __m128 xCond = _mm_cmpgt_ps(xd1Abs, xThreshold); // all 1's if true                        
        __m128 x2 = _mm_xor_ps(xSign, xThreshold); // -threshold if xd1 is negative and +threshold otherwise
        __m128 xd2 = _mm_and_ps(_mm_sub_ps(xd1, x2), xCond);
        _mm_storeu_ps(pd2, xd2);
    }

    for (; ps < psLim; ps++, pd1++, pd2++)
    {
        *pd1 += *ps * primalUpdate;
        float d1 = *pd1;
        *pd2 = std::abs(d1) > threshold ? (d1 > 0 ? d1 - threshold : d1 + threshold) : 0;
    }
}

EXPORT_API(void) SdcaL1UpdateSU(float primalUpdate, _In_ const float * ps, _In_ const int *pi, float threshold, _Inout_ float *pd1, _Inout_ float * pd2, int c)
{
    const int * piLim = pi + c;

    __m128 xPrimal = _mm_set1_ps(primalUpdate);

    __m128 signMask = _mm_set1_ps(-0.0f); // 1000 0000 ...
    __m128 xThreshold = _mm_set1_ps(threshold);
    for (; pi + 4 <= piLim; pi += 4, ps += 4)
    {
        __m128 xs = _mm_loadu_ps(ps);

        __m128 xd1 = _load4(pd1, pi);
        xd1 = _mm_add_ps(xd1, _mm_mul_ps(xs, xPrimal));

        __m128 xSign = _mm_and_ps(xd1, signMask); // result = 10000... if xd1 is negative or 00000 otherwise
        __m128 xd1Abs = _mm_xor_ps(xd1, xSign);
        __m128 xCond = _mm_cmpgt_ps(xd1Abs, xThreshold); // all 1's if true
        __m128 x2 = _mm_xor_ps(xSign, xThreshold); // -threshold if xd1 is negative and +threshold otherwise
        __m128 xd2 = _mm_and_ps(_mm_sub_ps(xd1, x2), xCond);

        _store4(xd1, pd1, pi);
        _store4(xd2, pd2, pi);
    }

    for (; pi < piLim; pi++, ps++)
    {
        int i = *pi;
        pd1[i] += *ps * primalUpdate;
        float d1 = pd1[i];
        pd2[i] = std::abs(d1) > threshold ? (d1 > 0 ? d1 - threshold : d1 + threshold) : 0;
    }
}

EXPORT_API(void) ScaleAdadeltaU(_Inout_ float * mat, _Inout_ float * accGrads, _Inout_ float * accUpdates, float decay, float cond, _In_ const float * grads, int size)
{
    float * pm = mat;
    float * pmLim = pm + size;
    float * pag = accGrads;
    float * pau = accUpdates;
    const float * pg = grads;

    __m128 dec = _mm_set1_ps(decay);
    __m128 decc = _mm_set1_ps(1 - decay);
    __m128 c = _mm_set1_ps(cond);

    for (; pm + 4 <= pmLim; pm += 4, pag += 4, pau += 4, pg += 4)
    {
        __m128 g = _mm_loadu_ps(pg);
        __m128 ag = _mm_loadu_ps(pag);
        __m128 au = _mm_loadu_ps(pau);
        __m128 coef = _mm_loadu_ps(pm);

        UpdateAdadelta(coef, ag, au, g, dec, decc, c);

        _mm_storeu_ps(pm, coef);
        _mm_storeu_ps(pag, ag);
        _mm_storeu_ps(pau, au);
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
}

EXPORT_API(void) ScaleAdadeltaA(_Inout_ float * mat, _Inout_ float * accGrads, _Inout_ float * accUpdates, float decay, float cond, _Inout_ float * grads, int size)
{
    float * pm = mat;
    float * pmLim = pm + size;
    float * pag = accGrads;
    float * pau = accUpdates;
    float * pg = grads;

    __m128 dec = _mm_set1_ps(decay);
    __m128 decc = _mm_set1_ps(1 - decay);
    __m128 c = _mm_set1_ps(cond);

    for (; pm < pmLim; pm += 4, pag += 4, pau += 4, pg += 4)
    {
        __m128 g = _mm_load_ps(pg);
        __m128 ag = _mm_load_ps(pag);
        __m128 au = _mm_load_ps(pau);
        __m128 coef = _mm_load_ps(pm);

        UpdateAdadelta(coef, ag, au, g, dec, decc, c);

        _mm_store_ps(pm, coef);
        _mm_store_ps(pag, ag);
        _mm_store_ps(pau, au);
    }
}
