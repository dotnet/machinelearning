//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

template<class FloatT, class FloatT2>
void SumupSegment_noindices(_In_ uint32_t* pData, _In_ uint8_t* pSegType, _In_ int32_t* pSegLength, _In_reads_(origSampleSize) FloatT* pSampleOutputs,
#ifdef IsWeighted
    _In_reads_(origSampleSize) FloatT2* pSampleOutputWeights,
#endif
    _Inout_ FloatT* pSumOutputsByBin,
#ifdef IsWeighted
    _Inout_ FloatT2* pSumWeightsByBin,
#endif
    _Inout_ int32_t* pCountByBin, _In_ int32_t origSampleSize)
{
    // Sumup over all values.
    uint64_t workingBits = pData[0] | ((uint64_t)pData[1] << 32);

    int bitsOffset = 0;
    pData += 2;
    FloatT* pSampleEnd = pSampleOutputs + origSampleSize;

    while (pSampleOutputs < pSampleEnd)
    {
        int32_t segEnd = *(pSegLength++);
        int8_t segType = *(pSegType++);
        uint32_t mask = ~((-1) << segType);

        while (segEnd-- > 0)
        {
            int32_t featureBin = (int32_t)((workingBits >> bitsOffset)&mask);
            pSumOutputsByBin[featureBin] += *(pSampleOutputs++);
#ifdef IsWeighted
            pSumWeightsByBin[featureBin] += *(pSampleOutputWeights++);
#endif
            pCountByBin[featureBin]++;
            bitsOffset += segType;
            if (bitsOffset >= 32)
            {
                workingBits = (workingBits >> 32) | (((uint64_t)*(pData++)) << 32);
                bitsOffset &= 31;
            }
        }
    }
}

template<class FloatT, class FloatT2>
void SumupSegment(_In_ uint32_t* pData, _In_ uint8_t* pSegType, _In_ int32_t* pSegLength, _In_reads_(origSampleSize) int32_t* pIndices, _In_reads_(origSampleSize) FloatT* pSampleOutputs,
#ifdef IsWeighted
    _In_reads_(origSampleSize) FloatT2* pSampleOutputWeights,
#endif
    _Inout_ FloatT* pSumOutputsByBin,
#ifdef IsWeighted
    _Inout_ FloatT2* pSumWeightsByBin,
#endif
    _Inout_ int32_t* pCountByBin, _In_ int32_t origSampleSize)
{
    if (pIndices == nullptr)
    {
        SumupSegment_noindices<FloatT, double>
#ifdef IsWeighted
            (pData, pSegType, pSegLength, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, origSampleSize);
#else
            (pData, pSegType, pSegLength, pSampleOutputs, pSumOutputsByBin, pCountByBin, origSampleSize);
#endif
        return;
    }

    int64_t globalBitOffset = 0;
    int32_t currIndex = 0, segEnd = *(pSegLength++);
    int32_t nextIndex = segEnd;
    int8_t segType = *(pSegType++);
    uint32_t mask = ~((-1) << segType);

    FloatT* pSampleEnd = pSampleOutputs + origSampleSize;

    while (pSampleOutputs < pSampleEnd)
    {
        int index = *(pIndices++);
        while (index >= nextIndex)
        {
            globalBitOffset += __emulu((unsigned int)segEnd, (unsigned int)segType);
            currIndex = nextIndex;
            nextIndex += (segEnd = *(pSegLength++));
            mask = ~((-1) << (segType = *(pSegType++)));
        }
        int64_t bitOffset = globalBitOffset + __emul(index - currIndex, (int)segType);
        int32_t major = (int32_t)(bitOffset >> 5), minor = (int32_t)(bitOffset & 0x1f);
        int32_t featureBin = (((uint64_t)pData[major] >> minor) | ((uint64_t)pData[major + 1] << (32 - minor)))&mask;
        pSumOutputsByBin[featureBin] += *(pSampleOutputs++);
#ifdef IsWeighted
        pSumWeightsByBin[featureBin] += *(pSampleOutputWeights++);
#endif
        pCountByBin[featureBin]++;
    }
}
