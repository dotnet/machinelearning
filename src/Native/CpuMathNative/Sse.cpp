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
// * Tran means the matrix is transposed.

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

// Multiply matrix times vector into vector.
EXPORT_API(void) MatMul(_In_ const float * pmat, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    const float * pSrcEnd = psrc + ccol;
    const float * pDstEnd = pdst + crow;
    float* pDstCurrent = pdst;
    const float* pMatCurrent = pmat;

    while (pDstCurrent < pDstEnd)
    {
        __m128 res0 = _mm_setzero_ps();
        __m128 res1 = res0;
        __m128 res2 = res0;
        __m128 res3 = res0;

        int length = ccol;
        const float* pSrcCurrent = psrc;

        uintptr_t address = (uintptr_t)(pMatCurrent);
        uintptr_t misalignment = address % 16;
        int remainder = 0;
  
        if ((misalignment & 3) != 0)
        {
            while (pSrcCurrent < pSrcEnd)
            {
                __m128 vector = _mm_loadu_ps(pSrcCurrent);
                
                const float* pMatTemp = pMatCurrent;
                __m128 x01 = _mm_mul_ps(vector, _mm_loadu_ps(pMatTemp));
                __m128 x11 = _mm_mul_ps(vector, _mm_loadu_ps(pMatTemp += ccol));
                __m128 x21 = _mm_mul_ps(vector, _mm_loadu_ps(pMatTemp += ccol));
                __m128 x31 = _mm_mul_ps(vector, _mm_loadu_ps(pMatTemp += ccol));

                res0 = _mm_add_ps(res0, x01);
                res1 = _mm_add_ps(res1, x11);
                res2 = _mm_add_ps(res2, x21);
                res3 = _mm_add_ps(res3, x31);
                
                pSrcCurrent += 4;
                pMatCurrent += 4;
            }    
        }
        else
        {
            if (misalignment != 0)
            {
                misalignment >>= 2;
                misalignment = 4 - misalignment;

                __m128 mask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (misalignment * 4));

                // We only align pMat since it has significantly more reads.
                const float* pMatTemp = pMatCurrent;
                __m128 x01 = _mm_and_ps(mask, _mm_loadu_ps(pMatTemp));
                __m128 x11 = _mm_and_ps(mask, _mm_loadu_ps(pMatTemp += ccol));
                __m128 x21 = _mm_and_ps(mask, _mm_loadu_ps(pMatTemp += ccol));
                __m128 x31 = _mm_and_ps(mask, _mm_loadu_ps(pMatTemp += ccol));                
                __m128 vector = _mm_and_ps(mask, _mm_loadu_ps(pSrcCurrent));

                res0 = _mm_mul_ps(x01, vector);
                res1 = _mm_mul_ps(x11, vector);
                res2 = _mm_mul_ps(x21, vector);
                res3 = _mm_mul_ps(x31, vector);
                                
                pMatCurrent += misalignment;
                pSrcCurrent += misalignment;
                length -= misalignment;
            }

            if (length > 3)
            {
                remainder = length % 4;
                while(pSrcCurrent < pSrcEnd)
                {
                    __m128 vector = _mm_loadu_ps(pSrcCurrent);
                
                    const float* pMatTemp = pMatCurrent;
                    __m128 x01 = _mm_mul_ps(vector, _mm_load_ps(pMatTemp));
                    __m128 x11 = _mm_mul_ps(vector, _mm_load_ps(pMatTemp += ccol));
                    __m128 x21 = _mm_mul_ps(vector, _mm_load_ps(pMatTemp += ccol));
                    __m128 x31 = _mm_mul_ps(vector, _mm_load_ps(pMatTemp += ccol));
                
                    res0 = _mm_add_ps(res0, x01);
                    res1 = _mm_add_ps(res1, x11);
                    res2 = _mm_add_ps(res2, x21);
                    res3 = _mm_add_ps(res3, x31);
                
                    pSrcCurrent += 4;
                    pMatCurrent += 4;
                } 
            }
            else
            {
                remainder = length;
            }

            if (remainder != 0)
            {
                pMatCurrent -= (4 - remainder);
                pSrcCurrent -= (4 - remainder);

                __m128 mask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + (remainder * 4));

                const float* pMatTemp = pMatCurrent;
                __m128 x01 = _mm_and_ps(mask, _mm_loadu_ps(pMatTemp));
                __m128 x11 = _mm_and_ps(mask, _mm_loadu_ps(pMatTemp += ccol));
                __m128 x21 = _mm_and_ps(mask, _mm_loadu_ps(pMatTemp += ccol));
                __m128 x31 = _mm_and_ps(mask, _mm_loadu_ps(pMatTemp += ccol));                
                __m128 vector = _mm_and_ps(mask, _mm_loadu_ps(pSrcCurrent));

                res0 = _mm_add_ps(x01, _mm_mul_ps(x01, vector));
                res1 = _mm_add_ps(x11, _mm_mul_ps(x11, vector));
                res2 = _mm_add_ps(x21, _mm_mul_ps(x21, vector));
                res3 = _mm_add_ps(x31, _mm_mul_ps(x31, vector));
              
                pMatCurrent += 4;
                pSrcCurrent += 4;
            }
        }

        // Add up the entries of each, with the 4 results in res0
        res0 = _mm_hadd_ps(res0, res1);
        res2 = _mm_hadd_ps(res2, res3);
        res0 = _mm_hadd_ps(res0, res2);

        _mm_storeu_ps(pDstCurrent, res0);

        pDstCurrent += 4;
        pMatCurrent += 3 * ccol;  
    }
}

// Partial sparse source vector.
EXPORT_API(void) MatMulP(_In_ const float * pmat, _In_ const int * pposSrc, _In_ const float * psrc,
    int posMin, int iposMin, int iposLim, _Inout_ float * pdst, int crow, int ccol)
{
    // REVIEW: For extremely sparse inputs, interchanging the loops would
    // likely be more efficient.
    const int * pposMin = pposSrc + iposMin;
    const int * pposEnd = pposSrc + iposLim;
    const float * pDstEnd = pdst + crow;
    const float * pm0 = pmat - posMin;
    const float * pSrcCurrent = psrc - posMin;
    float* pDstCurrent = pdst;
    
    uintptr_t address = (uintptr_t)(pDstCurrent);
    uintptr_t misalignment = address % 16;
    int length = crow;
    int remainder = 0;
    
    if ((misalignment & 3) != 0)
    {
        while (pDstCurrent < pDstEnd)
        {
            const float* pm1 = pm0 + ccol;
            const float* pm2 = pm1 + ccol;
            const float* pm3 = pm2 + ccol;

            __m128 res = _mm_setzero_ps();
            const int* ppos = pposMin;

            while (ppos < pposEnd)
            {
                int col = *ppos;
                __m128 x1 = _mm_setr_ps(pm0[col], pm1[col], pm2[col], pm3[col]);
                __m128 x2 = _mm_set1_ps(pSrcCurrent[col]);
                x2 = _mm_mul_ps(x2, x1);
                res = _mm_add_ps(res, x2);
                ppos++;
            }

            _mm_storeu_ps(pDstCurrent, res);
            pDstCurrent += 4;
            pm0 += 4 * ccol;
        }
    }
    else
    {
        if (misalignment != 0)
        {
            misalignment >>= 2;
            misalignment = 4 - misalignment;

            __m128 mask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (misalignment * 4));       

            const float* pm1 = pm0 + ccol;
            const float* pm2 = pm1 + ccol;
            const float* pm3 = pm2 + ccol;

            __m128 res = _mm_setzero_ps();
            const int* ppos = pposMin;

            while (ppos < pposEnd)
            {
                int col = *ppos;
                __m128 x1 = _mm_setr_ps(pm0[col], pm1[col], pm2[col], pm3[col]);
                x1 = _mm_and_ps(mask, x1);
                
                __m128 x2 = _mm_set1_ps(pSrcCurrent[col]);
                x2 = _mm_mul_ps(x2, x1);
                res = _mm_add_ps(res, x2);
                ppos++;
            }

            _mm_storeu_ps(pDstCurrent, res);
            pDstCurrent += misalignment;
            pm0 += misalignment * ccol;
            length -= misalignment;
        }

        if (length > 3)
        {
            remainder = length % 4;
            while (pDstCurrent < pDstEnd)
            {
                const float* pm1 = pm0 + ccol;
                const float* pm2 = pm1 + ccol;
                const float* pm3 = pm2 + ccol;

                const int* ppos = pposMin;
                __m128 res = _mm_setzero_ps();

                while (ppos < pposEnd)
                {
                    int col = *ppos;
                    __m128 x1 = _mm_setr_ps(pm0[col], pm1[col], pm2[col], pm3[col]);
                    __m128 x2 = _mm_set1_ps(pSrcCurrent[col]);
                    x2 = _mm_mul_ps(x2, x1);
                    res = _mm_add_ps(res, x2);
                    ppos++;
                }

                _mm_store_ps(pDstCurrent, res);
                pDstCurrent += 4;
                pm0 += 4 * ccol;
            }
        }
        else
        {
            length = remainder;
        }

        if (remainder != 0)
        {
            pDstCurrent -= (4 - remainder);
            pm0 -= (4 - remainder) * ccol;

            __m128 trailingMask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + (remainder * 4));     
            __m128 leadingMask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (( 4 - remainder) * 4));     
            
            const float* pm1 = pm0 + ccol;
            const float* pm2 = pm1 + ccol;
            const float* pm3 = pm2 + ccol;

            const int* ppos = pposMin;
            __m128 res = _mm_setzero_ps();
            
            while (ppos < pposEnd)
            {
                int col = *ppos;
                __m128 x1 = _mm_setr_ps(pm0[col], pm1[col], pm2[col], pm3[col]);
                x1 = _mm_and_ps(x1, trailingMask);

                __m128 x2 = _mm_set1_ps(pSrcCurrent[col]);
                x2 = _mm_mul_ps(x2, x1);
                res = _mm_add_ps(res, x2);
                ppos++;
            }

            res = _mm_add_ps(res, _mm_and_ps(leadingMask, _mm_loadu_ps(pDstCurrent)));
            _mm_storeu_ps(pDstCurrent, res);
            pDstCurrent += 4;
            pm0 += 4 * ccol;
        }
    }
}

EXPORT_API(void) MatMulTran(_In_ const float * pmat, _In_ const float * psrc, _Inout_ float * pdst, int crow, int ccol)
{
    const float * pSrcEnd = psrc + ccol;
    const float * pDstEnd = pdst + crow;
    
    const float* pMatCurrent = pmat;
    const float* pSrcCurrent = psrc;

    if (pSrcCurrent < pSrcEnd)
    {
         __m128 x01 = _mm_loadu_ps(pSrcCurrent);
        // Replicate each slot of x01 into its own register.
        __m128 x11 = _mm_shuffle_ps(x01, x01, 0x55);
        __m128 x21 = _mm_shuffle_ps(x01, x01, 0xAA);
        __m128 x31 = _mm_shuffle_ps(x01, x01, 0xFF);
        x01 = _mm_shuffle_ps(x01, x01, 0x00);

        int length = crow;
        float* pDstCurrent = pdst;

        uintptr_t address = (uintptr_t)(pMatCurrent);
        uintptr_t misalignment = address % 16;
        int remainder = 0;

        if ((misalignment & 3) != 0)
        {
            while (pDstCurrent < pDstEnd)
            {
                const float* pMatTemp = pMatCurrent;
                __m128 x02 = _mm_mul_ps(x01, _mm_loadu_ps(pMatTemp));
                __m128 x12 = _mm_mul_ps(x11, _mm_loadu_ps(pMatTemp += crow));
                __m128 x22 = _mm_mul_ps(x21, _mm_loadu_ps(pMatTemp += crow));
                __m128 x32 = _mm_mul_ps(x31, _mm_loadu_ps(pMatTemp += crow));

                x02 = _mm_add_ps(x02, x12);
                x22 = _mm_add_ps(x22, x32);
                x02 = _mm_add_ps(x02, x22);
                
                _mm_storeu_ps(pDstCurrent, x02);
                pDstCurrent += 4;
                pMatCurrent += 4;
            }
        }
        else
        {
            int remainder = 0;
            if (misalignment != 0)
            {
                misalignment >>= 2;
                misalignment = 4 - misalignment;
                
                __m128 leadingMask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (misalignment * 4));

                // We only align pMat since it has significantly more reads.
                const float* pMatTemp = pMatCurrent;
                __m128 x02 = _mm_and_ps(leadingMask, _mm_loadu_ps(pMatTemp));
                __m128 x12 = _mm_and_ps(leadingMask, _mm_loadu_ps(pMatTemp += crow));
                __m128 x22 = _mm_and_ps(leadingMask, _mm_loadu_ps(pMatTemp += crow));
                __m128 x32 = _mm_and_ps(leadingMask, _mm_loadu_ps(pMatTemp += crow));

                x02 = _mm_mul_ps(x01, x02);
                x12 = _mm_mul_ps(x11, x12);
                x22 = _mm_mul_ps(x21, x22);
                x32 = _mm_mul_ps(x31, x32);                                                

                x02 = _mm_add_ps(x02, x12);
                x22 = _mm_add_ps(x22, x32);
                x02 = _mm_add_ps(x02, x22);

                __m128 trailingMask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + (( 4 - misalignment) * 4));
                __m128 x3 = _mm_loadu_ps(pDstCurrent);
                x02 = _mm_or_ps(x02, _mm_and_ps(x3, trailingMask));

                _mm_storeu_ps(pDstCurrent, x02);
                pMatCurrent += misalignment;
                pDstCurrent += misalignment;
                length -= misalignment;
            }

            if(length > 3)
            {
                remainder = length % 4;
                while (pDstCurrent < pDstEnd)
                {
                    const float* pMatTemp = pMatCurrent;
                    __m128 x02 = _mm_mul_ps(x01, _mm_load_ps(pMatTemp));
                    __m128 x12 = _mm_mul_ps(x11, _mm_load_ps(pMatTemp += crow));
                    __m128 x22 = _mm_mul_ps(x21, _mm_load_ps(pMatTemp += crow));
                    __m128 x32 = _mm_mul_ps(x31, _mm_load_ps(pMatTemp += crow));

                    x02 = _mm_add_ps(x02, x12);
                    x22 = _mm_add_ps(x22, x32);
                    x02 = _mm_add_ps(x02, x22);

                    _mm_storeu_ps(pDstCurrent, x02);
                
                    pDstCurrent += 4;
                    pMatCurrent += 4;               
                }
            }
            else
            {
                length = remainder;
            }

            if (remainder != 0)
            {
                pMatCurrent -= (4 - remainder);
                pDstCurrent -= (4 - remainder);

                __m128 trailingMask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + (remainder * 4));             
                
                const float* pMatTemp = pMatCurrent;
                __m128 x02 = _mm_and_ps(trailingMask, _mm_loadu_ps(pMatTemp));
                __m128 x12 = _mm_and_ps(trailingMask, _mm_loadu_ps(pMatTemp += crow));
                __m128 x22 = _mm_and_ps(trailingMask, _mm_loadu_ps(pMatTemp += crow));
                __m128 x32 = _mm_and_ps(trailingMask, _mm_loadu_ps(pMatTemp += crow));

                x02 = _mm_mul_ps(x01, x02);
                x12 = _mm_mul_ps(x11, x12);
                x22 = _mm_mul_ps(x21, x22);
                x32 = _mm_mul_ps(x31, x32);                                                

                x02 = _mm_add_ps(x02, x12);
                x22 = _mm_add_ps(x22, x32);
                x02 = _mm_add_ps(x02, x22);

                __m128 leadingMask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (( 4 - remainder) * 4));
                __m128 x3 = _mm_loadu_ps(pDstCurrent);
                x02 = _mm_or_ps(x02, _mm_and_ps(x3, leadingMask));

                _mm_storeu_ps(pDstCurrent, x02);
                pMatCurrent += 4;
                pDstCurrent += 4;
            }
        }

        pMatCurrent += 3 * crow;
        pSrcCurrent += 4;
    }

    while (pSrcCurrent < pSrcEnd)
    {
        __m128 x01 = _mm_loadu_ps(pSrcCurrent);
        // Replicate each slot of x01 into its own register.
        __m128 x11 = _mm_shuffle_ps(x01, x01, 0x55);
        __m128 x21 = _mm_shuffle_ps(x01, x01, 0xAA);
        __m128 x31 = _mm_shuffle_ps(x01, x01, 0xFF);
        x01 = _mm_shuffle_ps(x01, x01, 0x00);

        int length = crow;
        float* pDstCurrent = pdst;

        uintptr_t address = (uintptr_t)(pMatCurrent);
        uintptr_t misalignment = address % 16;
        int remainder = 0;

        if ((misalignment & 3) != 0)
        {
            while (pDstCurrent < pDstEnd)
            {
                const float* pMatTemp = pMatCurrent;
                __m128 x02 = _mm_mul_ps(x01, _mm_loadu_ps(pMatTemp));
                __m128 x12 = _mm_mul_ps(x11, _mm_loadu_ps(pMatTemp += crow));
                __m128 x22 = _mm_mul_ps(x21, _mm_loadu_ps(pMatTemp += crow));
                __m128 x32 = _mm_mul_ps(x31, _mm_loadu_ps(pMatTemp += crow));

                x02 = _mm_add_ps(x02, x12);
                x22 = _mm_add_ps(x22, x32);
                x02 = _mm_add_ps(x02, x22);

                x02 = _mm_add_ps(x02, _mm_loadu_ps(pDstCurrent));
            
                _mm_storeu_ps(pDstCurrent, x02);
                pDstCurrent += 4;
                pMatCurrent += 4;
            }
        }
        else
        {
            int remainder = 0;
            if (misalignment != 0)
            {
                misalignment >>= 2;
                misalignment = 4 - misalignment;
                
                __m128 leadingMask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (misalignment * 4));

                // We only align pMat since it has significantly more reads.
                const float* pMatTemp = pMatCurrent;
                __m128 x02 = _mm_and_ps(leadingMask, _mm_loadu_ps(pMatTemp));
                __m128 x12 = _mm_and_ps(leadingMask, _mm_loadu_ps(pMatTemp += crow));
                __m128 x22 = _mm_and_ps(leadingMask, _mm_loadu_ps(pMatTemp += crow));
                __m128 x32 = _mm_and_ps(leadingMask, _mm_loadu_ps(pMatTemp += crow));

                x02 = _mm_mul_ps(x01, x02);
                x12 = _mm_mul_ps(x11, x12);
                x22 = _mm_mul_ps(x21, x22);
                x32 = _mm_mul_ps(x31, x32);                                                

                x02 = _mm_add_ps(x02, x12);
                x22 = _mm_add_ps(x22, x32);
                x02 = _mm_add_ps(x02, x22);

                __m128 trailingMask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + (( 4 - misalignment) * 4));
                __m128 x3 = _mm_loadu_ps(pDstCurrent);
                x02 = _mm_or_ps(x02, _mm_and_ps(x3, trailingMask));
                x02 = _mm_add_ps(x02, _mm_and_ps(x3, leadingMask));

                _mm_storeu_ps(pDstCurrent, x02);
                pMatCurrent += misalignment;
                pDstCurrent += misalignment;
                length -= misalignment;
            }

            if(length > 3)
            {
                remainder = length % 4;
                while (pDstCurrent < pDstEnd)
                {
                    const float* pMatTemp = pMatCurrent;
                    __m128 x02 = _mm_mul_ps(x01, _mm_load_ps(pMatTemp));
                    __m128 x12 = _mm_mul_ps(x11, _mm_load_ps(pMatTemp += crow));
                    __m128 x22 = _mm_mul_ps(x21, _mm_load_ps(pMatTemp += crow));
                    __m128 x32 = _mm_mul_ps(x31, _mm_load_ps(pMatTemp += crow));

                    x02 = _mm_add_ps(x02, x12);
                    x22 = _mm_add_ps(x22, x32);
                    x02 = _mm_add_ps(x02, x22);

                    x02 = _mm_add_ps(x02,  _mm_loadu_ps(pDstCurrent));

                    _mm_storeu_ps(pDstCurrent, x02);
                
                    pDstCurrent += 4;
                    pMatCurrent += 4;               
                }
            }
            else
            {
                length = remainder;
            }

            if (remainder != 0)
            {
                pMatCurrent -= (4 - remainder);
                pDstCurrent -= (4 - remainder);

                __m128 trailingMask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + (remainder * 4));             
                
                const float* pMatTemp = pMatCurrent;
                __m128 x02 = _mm_and_ps(trailingMask, _mm_loadu_ps(pMatTemp));
                __m128 x12 = _mm_and_ps(trailingMask, _mm_loadu_ps(pMatTemp += crow));
                __m128 x22 = _mm_and_ps(trailingMask, _mm_loadu_ps(pMatTemp += crow));
                __m128 x32 = _mm_and_ps(trailingMask, _mm_loadu_ps(pMatTemp += crow));

                x02 = _mm_mul_ps(x01, x02);
                x12 = _mm_mul_ps(x11, x12);
                x22 = _mm_mul_ps(x21, x22);
                x32 = _mm_mul_ps(x31, x32);                                                

                x02 = _mm_add_ps(x02, x12);
                x22 = _mm_add_ps(x22, x32);
                x02 = _mm_add_ps(x02, x22);

                __m128 leadingMask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (( 4 - remainder) * 4));
                __m128 x3 = _mm_loadu_ps(pDstCurrent);
                x02 = _mm_or_ps(x02, _mm_and_ps(x3, leadingMask));

                x02 = _mm_add_ps(x02, _mm_and_ps(x3, trailingMask));
                _mm_storeu_ps(pDstCurrent, x02);
                pMatCurrent += 4;
                pDstCurrent += 4;
            }
        }

        pMatCurrent += 3 * crow;
        pSrcCurrent += 4;
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

EXPORT_API(float) Sum(const float* pValues, int length)
{
    if (length < 4)
    {
        // Handle cases where we have less than 128-bits total and can't ever use SIMD acceleration.

        float result = 0;

        switch (length)
        {
            case 3: result += pValues[2];
            case 2: result += pValues[1];
            case 1: result += pValues[0];
        }

        return result;
    }

    __m128 result = _mm_setzero_ps();

    uintptr_t address = (uintptr_t)(pValues);
    uintptr_t misalignment = address % 16;

    int remainder = 0;

    if ((misalignment & 3) != 0)
    {
        // Handles cases where the data is not 32-bit aligned and we can't ever use aligned operations

        remainder = length % 4;

        for (const float* pEnd = pValues + (length - remainder); pValues < pEnd; pValues += 4)
        {
            __m128 temp = _mm_loadu_ps(pValues);
            result = _mm_add_ps(result, temp);
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

            __m128 temp = _mm_loadu_ps(pValues);
            __m128 mask = _mm_loadu_ps(((float*)(&LeadingAlignmentMask)) + (misalignment * 4));
            temp = _mm_and_ps(temp, mask);
            result = _mm_add_ps(result, temp);

            pValues += misalignment;
            length -= misalignment;
        }

        if (length > 3)
        {
            // Handle all the 128-bit blocks that we can now that we have offset to an aligned address

            remainder = length % 4;

            for (const float* pEnd = pValues + (length - remainder); pValues < pEnd; pValues += 4)
            {
                __m128 temp = _mm_load_ps(pValues);
                result = _mm_add_ps(result, temp);
            }
        }
        else
        {
            // Handle the "worst-case" scenario, which is when we have 4-8 elements and the input is not
            // 128-bit aligned. This means we can't do any aligned loads and will just end up doing two
            // unaligned loads where we mask the input each time.
            remainder = length;
        }
    }

    if (remainder != 0)
    {
        // Handle any trailing elements that don't fit into a 128-bit block by moving back so that the next
        // unaligned load will read to the end of the array and then mask out any elements already processed

        pValues -= (4 - remainder);

        __m128 temp = _mm_loadu_ps(pValues);
        __m128 mask = _mm_loadu_ps(((float*)(&TrailingAlignmentMask)) + (remainder * 4));
        temp = _mm_and_ps(temp, mask);
        result = _mm_add_ps(result, temp);
    }

    // Sum all the elements together and return the result

    result = _mm_add_ps(result, _mm_movehl_ps(result, result));
    result = _mm_add_ps(result, _mm_shuffle_ps(result, result, 0xB1));

    return _mm_cvtss_f32(result);
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
