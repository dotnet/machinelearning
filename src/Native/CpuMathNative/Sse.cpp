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

// Multiply matrix times vector into vector.
EXPORT_API(void) MatMul(bool add, _In_ const float * pmat, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
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
            __m128 x01 = _mm_loadu_ps(pmTmp = pm);
            __m128 x11 = _mm_loadu_ps(pmTmp += ccol);
            __m128 x21 = _mm_loadu_ps(pmTmp += ccol);
            __m128 x31 = _mm_loadu_ps(pmTmp += ccol);
            __m128 x02 = _mm_loadu_ps(ps);
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
            res0 = _mm_add_ps(res0, _mm_loadu_ps(pd));
        _mm_storeu_ps(pd, res0);
    }
}

// Partial sparse source vector.
EXPORT_API(void) MatMulP(bool add, _In_ const float * pmat, _In_ const int * pposSrc, _In_ const float * psrc,
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
            res = _mm_add_ps(res, _mm_loadu_ps(pd));
        _mm_storeu_ps(pd, res);
    }
}

EXPORT_API(void) MatMulTran(bool add, _In_ const float * pmat, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    const float * psLim = psrc + ccol;
    const float * pdLim = pdst + crow;
    const float * pm = pmat;
    const float * ps = psrc;

    if (!add)
    {
        __m128 x01 = _mm_loadu_ps(ps);
        // Replicate each slot of x01 into its own register.
        __m128 x11 = _mm_shuffle_ps(x01, x01, 0x55);
        __m128 x21 = _mm_shuffle_ps(x01, x01, 0xAA);
        __m128 x31 = _mm_shuffle_ps(x01, x01, 0xFF);
        x01 = _mm_shuffle_ps(x01, x01, 0x00);
        ps += 4;
        for (float * pd = pdst; pd < pdLim; pd += 4, pm += 4)
        {
            const float * pmTmp;
            __m128 x02 = _mm_loadu_ps(pmTmp = pm);
            __m128 x12 = _mm_loadu_ps(pmTmp += crow);
            __m128 x22 = _mm_loadu_ps(pmTmp += crow);
            __m128 x32 = _mm_loadu_ps(pmTmp += crow);
            x02 = _mm_mul_ps(x01, x02);
            x12 = _mm_mul_ps(x11, x12);
            x22 = _mm_mul_ps(x21, x22);
            x32 = _mm_mul_ps(x31, x32);
            x02 = _mm_add_ps(x02, x12);
            x22 = _mm_add_ps(x22, x32);
            x02 = _mm_add_ps(x02, x22);
            _mm_storeu_ps(pd, x02);
        }

        pm += 3 * crow;
    }

    for (; ps < psLim; ps += 4)
    {
        __m128 x01 = _mm_loadu_ps(ps);
        // Replicate each slot of x01 into its own register.
        __m128 x11 = _mm_shuffle_ps(x01, x01, 0x55);
        __m128 x21 = _mm_shuffle_ps(x01, x01, 0xAA);
        __m128 x31 = _mm_shuffle_ps(x01, x01, 0xFF);
        x01 = _mm_shuffle_ps(x01, x01, 0x00);
        for (float * pd = pdst; pd < pdLim; pd += 4, pm += 4)
        {
            const float * pmTmp;
            __m128 x02 = _mm_loadu_ps(pmTmp = pm);
            __m128 x12 = _mm_loadu_ps(pmTmp += crow);
            __m128 x22 = _mm_loadu_ps(pmTmp += crow);
            __m128 x32 = _mm_loadu_ps(pmTmp += crow);
            __m128 x3 = _mm_loadu_ps(pd);
            x02 = _mm_mul_ps(x01, x02);
            x12 = _mm_mul_ps(x11, x12);
            x22 = _mm_mul_ps(x21, x22);
            x32 = _mm_mul_ps(x31, x32);
            x02 = _mm_add_ps(x02, x12);
            x22 = _mm_add_ps(x22, x32);
            x02 = _mm_add_ps(x02, x22);
            x3 = _mm_add_ps(x02, x3);
            _mm_storeu_ps(pd, x3);
        }

        pm += 3 * crow;
    }
}

// Partial sparse source vector.
EXPORT_API(void) MatMulTranP(bool add, _In_ const float * pmat, _In_ const int * pposSrc, _In_ const float * psrc,
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
            __m128 x1 = _mm_loadu_ps(pm);
            x1 = _mm_mul_ps(x1, x0);
            _mm_storeu_ps(pd, x1);
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
            __m128 x1 = _mm_loadu_ps(pm);
            __m128 x2 = _mm_loadu_ps(pd);
            x1 = _mm_mul_ps(x1, x0);
            x2 = _mm_add_ps(x2, x1);
            _mm_storeu_ps(pd, x2);
        }
    }
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

EXPORT_API(void) ScaleU(float a, _Inout_ float * pd, int c)
{
    float * pdLim = pd + c;

    __m128 x1 = _mm_set1_ps(a);
    for (; pd + 4 <= pdLim; pd += 4)
    {
        __m128 x2 = _mm_loadu_ps(pd);
        x2 = _mm_mul_ps(x1, x2);
        _mm_storeu_ps(pd, x2);
    }

    for (; pd < pdLim; pd++)
    {
        __m128 x2 = _mm_load_ss(pd);
        x2 = _mm_mul_ss(x1, x2);
        _mm_store_ss(pd, x2);
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