//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#ifdef IsWeighted
template<class FloatT, class FloatT2, class T>
#else
template<class FloatT, class T>
#endif
void SumupNibbles_noindices
(_In_reads_(origSampleSize) T* pData, _In_reads_(origSampleSize) FloatT* pSampleOutputs,
#ifdef IsWeighted
    _In_reads_(origSampleSize) FloatT2* pSampleOutputWeights,
#endif
    _Inout_ FloatT* pSumOutputsByBin,
#ifdef IsWeighted
    _Inout_ FloatT2* pSumWeightsByBin,
#endif
    _Inout_ int32_t* pCountByBin, _In_ int32_t origSampleSize)
{
    bool high = true;
    for (int i = 0; i < origSampleSize; i++, high = !high)
    {
        int index = i / 2;
        //bool high = i % 2 == 0;

        int32_t featureValue = (int32_t)(pData[index]);
        if (high) featureValue >>= 4;
        else featureValue &= 0x0f;

        FloatT output = pSampleOutputs[i];
        pSumOutputsByBin[featureValue] += output;
#ifdef IsWeighted
        FloatT2 output2 = pSampleOutputWeights[i];
        pSumWeightsByBin[featureValue] += output2;
#endif
        ++pCountByBin[featureValue];
    }
}


#ifdef IsWeighted
template<class FloatT, class FloatT2, class T>
#else
template<class FloatT, class T>
#endif
void SumupNibbles
(_In_ T* pData, _In_reads_(origSampleSize) int32_t* pIndices, _In_reads_(origSampleSize) FloatT* pSampleOutputs,
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
#ifdef IsWeighted
        SumupNibbles_noindices<FloatT, FloatT2, T>
            (pData, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, origSampleSize);
#else
        SumupNibbles_noindices<FloatT, T>
            (pData, pSampleOutputs, pSumOutputsByBin, pCountByBin, origSampleSize);
#endif
        return;
    }
    for (int i = 0; i < origSampleSize; i++)
    {
        int32_t featureValue = (pData[pIndices[i] >> 1] >> ((~(pIndices[i] << 2)) & 4)) & 0xf;

        FloatT output = pSampleOutputs[i];
        pSumOutputsByBin[featureValue] += output;
#ifdef IsWeighted
        FloatT2 output2 = pSampleOutputWeights[i];
        pSumWeightsByBin[featureValue] += output2;
#endif
        ++pCountByBin[featureValue];
    }
}
