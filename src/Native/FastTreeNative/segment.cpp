//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#include "stdafx.h"
#include <iostream> // for qsort
#include <stdio.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>

EXPORT_API(void) C_SegmentFindOptimalPath15(_In_reads_(valc) unsigned long* valv, _In_ long valc, _Inout_ long long* pBits, _Inout_ long* pTransitions)
{
    unsigned long transmap, bitindex = 0, bestindex = 0;
    short bestcost;
    long long bits = 40;

    // Get the preallocated memory.
    __m128i masksu[16];
    {
        __m128i imask = _mm_set1_epi16(~0);
        for (int i = 15; i >= 0; --i)
        {
            imask = _mm_srli_si128(imask, 1);
            _mm_storeu_si128(masksu + i, imask);
        }
    }

    __m128i state0f = _mm_setzero_si128();

    for (int i = 0; i < valc; ++i)
    {
        // The bit ops are a little goofy, but it's a lot better than running conditional tests.
        _BitScanReverse((unsigned long*)&bitindex, (((unsigned long)valv[i]) << 1) | 1);
        __m128i mask = _mm_load_si128(masksu + bitindex);
        __m128i staycost = _mm_adds_epu8(state0f, _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
        __m128i unmasked = _mm_min_epu8(staycost, _mm_setr_epi8(
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55));
        // Find the transition map.
        transmap = _mm_movemask_epi8(_mm_cmpeq_epi8(unmasked, staycost));
        // Calculate the next base state of state0f.
        state0f = _mm_or_si128(mask, unmasked);
        // Calculate the min cost position in the current iteration.
        bestcost = _mm_extract_epi16(_mm_minpos_epu16(_mm_min_epu16(
            _mm_cvtepi8_epi16(state0f), _mm_cvtepi8_epi16(_mm_srli_si128(state0f, 8)))), 0);
        // Keep the invariant that the min cost position has 0 cost.
        state0f = _mm_or_si128(mask, _mm_sub_epi8(state0f, _mm_set1_epi8((char)bestcost)));
        // Find the bit index of the best position.
        _BitScanReverse((unsigned long*)&bestindex, (long)(_mm_movemask_epi8(_mm_cmpeq_epi8(state0f, _mm_setzero_si128()))));
        // Store the vital statistics.
        // [31,27]: best bit index
        // [26,22]: min bits to encoded current value (m)
        // [21, m]: Transition map (implicitly starts at bit 0)
        // ( m, 0]: The value
        valv[i] |= ((bestindex << 27) | (bitindex << 22) | ((((unsigned int)transmap) >> bitindex) << bitindex));
        bits += bestcost;
    }

    long back = 0, bitness = 0, transitions = 0;
    for (int i = valc - 1; i >= 0; --i)
    {
        bitness = back ? bitness : (valv[i] >> 27);
        transitions += 1 - back;
        back = (valv[i] >> bitness) & 1;
        valv[i] &= (1 << (valv[i] >> 22)) - 1;
        valv[i] |= (bitness << 27);
    }

    *pBits = bits;
    *pTransitions = transitions;
}

EXPORT_API(void) C_SegmentFindOptimalPath21(_In_reads_(valc) unsigned long* valv, _In_ long valc, _Inout_ long long* pBits, _Inout_ long* pTransitions)
{
    unsigned long transmap, bitindex = 0, bestindex = 0;
    short bestcost;
    long long bits = 40;
    unsigned long* end = valv + valc;

    __m128i statelo = _mm_setzero_si128(), statehi = _mm_setzero_si128();

    // Get the preallocated memory.
    __m128i masksu[32 + 16];
    {
        __m128i imask = _mm_set1_epi16(~0);
        for (int i = 47; i >= 32; --i)
        {
            _mm_storeu_si128(masksu + i, imask);
        }
        for (int i = 31; i >= 16; --i)
        {
            imask = _mm_srli_si128(imask, 1);
            _mm_storeu_si128(masksu + i, imask);
        }
        for (int i = 15; i >= 0; --i)
        {
            _mm_storeu_si128(masksu + i, _mm_setzero_si128());
        }
    }

    for (int i = 0; i < valc; ++i)
    {
        // The bit ops are a little goofy, but it's a lot better than running conditional tests.
        _BitScanReverse((unsigned long*)&bitindex, (((unsigned long)valv[i]) << 1) | 1);
        __m128i masklo = _mm_load_si128(masksu + bitindex + 16);

        // Low order bit calculation.
        __m128i staycost = _mm_adds_epu8(statelo, _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
        __m128i unmasked = _mm_min_epu8(staycost, _mm_setr_epi8(
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55));
        // Find the transition map.
        transmap = _mm_movemask_epi8(_mm_cmpeq_epi8(unmasked, staycost));
        // Calculate the next base state.
        statelo = _mm_or_si128(masklo, unmasked);

        // High order bit calculation.
        __m128i maskhi = _mm_load_si128(masksu + bitindex);
        // Low order bit calculation.
        staycost = _mm_adds_epu8(statehi, _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31));
        unmasked = _mm_min_epu8(staycost, _mm_setr_epi8(
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71));
        // Find the transition map.
        transmap |= _mm_movemask_epi8(_mm_cmpeq_epi8(unmasked, staycost)) << 16;
        // Calculate the next base state.
        statehi = _mm_or_si128(maskhi, unmasked);

        // Calculate the min cost position in the current iteration.
        bestcost = _mm_extract_epi16(_mm_minpos_epu16(_mm_min_epu16(
            _mm_min_epu16(_mm_cvtepi8_epi16(statelo), _mm_cvtepi8_epi16(_mm_srli_si128(statelo, 8))),
            _mm_min_epu16(_mm_cvtepi8_epi16(statehi), _mm_cvtepi8_epi16(_mm_srli_si128(statehi, 8))))), 0);
        // Keep the invariant that the min cost position has 0 cost.
        statelo = _mm_or_si128(masklo, _mm_sub_epi8(statelo, _mm_set1_epi8((char)bestcost)));
        statehi = _mm_or_si128(maskhi, _mm_sub_epi8(statehi, _mm_set1_epi8((char)bestcost)));
        // Find the bit index of the best position.
        _BitScanReverse((unsigned long*)&bestindex, (long)(_mm_movemask_epi8(_mm_cmpeq_epi8(statelo, _mm_setzero_si128())) | (_mm_movemask_epi8(_mm_cmpeq_epi8(statehi, _mm_setzero_si128())) << 16)));
        // Store the vital statistics.
        // [31,27]: best bit index
        // [26,22]: min bits to encoded current value (m)
        // [21, m]: Transition map (implicitly starts at bit 0)
        // ( m, 0]: The value
        valv[i] |= ((bestindex << 27) | (bitindex << 22) | ((((unsigned int)transmap & 0x003fffff) >> bitindex) << bitindex));
        bits += bestcost;
    }

    long back = 0, bitness = 0, transitions = 0;
    for (int i = valc - 1; i >= 0; --i)
    {
        bitness = back ? bitness : (valv[i] >> 27);
        transitions += 1 - back;
        back = (valv[i] >> bitness) & 1;
        valv[i] &= (1 << (valv[i] >> 22)) - 1;
        valv[i] |= (bitness << 27);
    }

    *pBits = bits;
    *pTransitions = transitions;
}

EXPORT_API(void) C_SegmentFindOptimalPath7(_In_reads_(valc) unsigned long* valv, _In_ long valc, _Inout_ long long* pBits, _Inout_ long* pTransitions)
{
    // TODO: Speed this up.
    // In principle it seems like, with fewer and simpler operations,
    // and fewer conversions from 8-bit to 16-bit, that this should be
    // faster than the 15-bit version of this function, but this does
    // not seem to actually be the case?  This should be improved.
    unsigned long transmap, bitindex = 0, bestindex = 0;
    short bestcost;
    long long bits = 40;
    unsigned long* end = valv + valc;

    __m128i state0f, stay, transition;

    state0f = _mm_setzero_si128();
    stay = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
    transition = _mm_add_epi16(_mm_set1_epi16(40), stay);

    // Get the preallocated memory.
    __m128i masksu[8];
    {
        __m128i imask = _mm_set1_epi8(~0);
        for (int i = 7; i >= 0; --i)
        {
            imask = _mm_srli_si128(imask, 2);
            _mm_storeu_si128(masksu + i, imask);
        }
    }

    for (int i = 0; i < valc; ++i)
    {
        // The bit ops are a little goofy, but it's a lot better than running conditional tests.
        _BitScanReverse((unsigned long*)&bitindex, (((unsigned long)valv[i]) << 1) | 1);
        __m128i mask = _mm_load_si128(masksu + bitindex);
        __m128i staycost = _mm_adds_epu16(state0f, stay);
        __m128i unmasked = _mm_min_epu16(staycost, transition);
        // Find the transition map.
        transmap = _mm_movemask_epi8(_mm_packus_epi16(_mm_cmpeq_epi16(unmasked, staycost), _mm_setzero_si128()));
        // Calculate the next base state of state0f.
        state0f = _mm_or_si128(mask, unmasked);
        // Calculate the min cost position in the current iteration.
        __m128i min = _mm_minpos_epu16(state0f);
        bestcost = _mm_extract_epi16(min, 0);
        bestindex = _mm_extract_epi16(min, 1);
        // Keep the invariant that the min cost position has 0 cost.
        state0f = _mm_or_si128(mask, _mm_sub_epi8(state0f, _mm_set1_epi16(bestcost)));
        // Store the vital statistics.
        // [31,27]: best bit index
        // [26,22]: min bits to encoded current value (m)
        // [21, m]: Transition map (implicitly starts at bit 0)
        // ( m, 0]: The value
        valv[i] |= ((bestindex << 27) | (bitindex << 22) | ((((unsigned int)transmap) >> bitindex) << bitindex));
        bits += bestcost;
    }

    long back = 0, bitness = 0, transitions = 0;
    for (int i = valc - 1; i >= 0; --i)
    {
        bitness = back ? bitness : (valv[i] >> 27);
        transitions += 1 - back;
        back = (valv[i] >> bitness) & 1;
        valv[i] &= (1 << (valv[i] >> 22)) - 1;
        valv[i] |= (bitness << 27);
    }

    *pBits = bits;
    *pTransitions = transitions;
}

EXPORT_API(void) C_SegmentFindOptimalCost15(_In_reads_(valc) unsigned int* valv, _In_ const int valc, _Inout_ long* pBits)
{
    unsigned int val, numbits, transitions = 0;
    __m128i state0f, stay, transition;
    long bits = 40;
    unsigned long bitindex = 0;

    // Get the preallocated masks.
    __m128i masksu[16];
    stay = _mm_set1_epi16(~0);
    for (int i = 15; i >= 0; --i)
    {
        stay = _mm_srli_si128(stay, 1);
        _mm_storeu_si128(masksu + i, stay);
    }

    state0f = _mm_setzero_si128();
    stay = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    transition = _mm_add_epi8(_mm_set1_epi8(40), stay);

    for (int i = 0; i < valc; ++i)
    {
        val = valv[i];
        bitindex = 0;
        _BitScanReverse((unsigned long*)&bitindex, (((unsigned long)valv[i]) << 1) | 1);
        numbits = bitindex;

        __m128i mask = _mm_load_si128(masksu + numbits);
        // Calculate the next base state of state0f.
        state0f = _mm_or_si128(mask, _mm_min_epu8(_mm_adds_epu8(state0f, stay), transition));
        // Calculate the min cost position in the current iteration.
        short bestcost = _mm_extract_epi16(_mm_minpos_epu16(_mm_min_epu16(
            _mm_cvtepi8_epi16(state0f), _mm_cvtepi8_epi16(_mm_srli_si128(state0f, 8)))), 0);
        // Keep the invariant that the min cost position has 0 cost.
        state0f = _mm_or_si128(mask, _mm_sub_epi8(state0f, _mm_set1_epi8((char)bestcost)));
        // Find the position of the best position.
        bits += bestcost;
    }
    *pBits = bits;
}

EXPORT_API(void) C_SegmentFindOptimalCost31(_In_reads_(valc) unsigned long* valv, _In_ long valc, _Inout_ long long* pBits)
{
    unsigned long bitindex = 0, bestindex = 0;
    short bestcost;
    long long bits = 40;
    unsigned long* end = valv + valc;

    __m128i statelo = _mm_setzero_si128(), statehi = _mm_setzero_si128();

    // Get the preallocated memory.
    __m128i masksu[32 + 16];
    {
        __m128i imask = _mm_set1_epi16(~0);
        for (int i = 47; i >= 32; --i)
        {
            _mm_storeu_si128(masksu + i, imask);
        }
        for (int i = 31; i >= 16; --i)
        {
            imask = _mm_srli_si128(imask, 1);
            _mm_storeu_si128(masksu + i, imask);
        }
        for (int i = 15; i >= 0; --i)
        {
            _mm_storeu_si128(masksu + i, _mm_setzero_si128());
        }
    }

    for (int i = 0; i < valc; ++i)
    {
        // The bit ops are a little goofy, but it's a lot better than running conditional tests.
        _BitScanReverse((unsigned long*)&bitindex, (((unsigned long)valv[i]) << 1) | 1);
        __m128i masklo = _mm_load_si128(masksu + bitindex + 16);
        statelo = _mm_or_si128(masklo, _mm_min_epu8(
            _mm_adds_epu8(statelo, _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
            _mm_setr_epi8(40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55)));

        // High order bit calculation.
        __m128i maskhi = _mm_load_si128(masksu + bitindex);
        statehi = _mm_or_si128(maskhi, _mm_min_epu8(
            _mm_adds_epu8(statehi, _mm_setr_epi8(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)),
            _mm_setr_epi8(56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71)));

        // Calculate the min cost position in the current iteration.
        bestcost = _mm_extract_epi16(_mm_minpos_epu16(_mm_min_epu16(
            _mm_min_epu16(_mm_cvtepi8_epi16(statelo), _mm_cvtepi8_epi16(_mm_srli_si128(statelo, 8))),
            _mm_min_epu16(_mm_cvtepi8_epi16(statehi), _mm_cvtepi8_epi16(_mm_srli_si128(statehi, 8))))), 0);
        // Keep the invariant that the min cost position has 0 cost.
        statelo = _mm_or_si128(masklo, _mm_sub_epi8(statelo, _mm_set1_epi8((char)bestcost)));
        statehi = _mm_or_si128(maskhi, _mm_sub_epi8(statehi, _mm_set1_epi8((char)bestcost)));
        bits += bestcost;
    }

    *pBits = bits;
}