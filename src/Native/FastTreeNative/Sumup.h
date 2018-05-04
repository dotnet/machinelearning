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
void Sumup_noindices(_In_reads_(origSampleSize) T* pData,
    _In_reads_(origSampleSize) FloatT* pSampleOutputs,
#ifdef IsWeighted
    _In_reads_(origSampleSize) FloatT2* pSampleOutputWeights,
#endif
    _Inout_ FloatT* pSumOutputsByBin,
#ifdef IsWeighted
    _Inout_ FloatT2* pSumWeightsByBin,
#endif
    _Inout_ int32_t* pCountByBin, _In_ int32_t origSampleSize)
{
    for (int i = 0; i < origSampleSize; i++)
    {
        T featureValue = pData[i];

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
void Sumup(_In_ T* pData, _In_reads_(origSampleSize) int32_t* pIndices, _In_reads_(origSampleSize) FloatT* pSampleOutputs,
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
        Sumup_noindices<FloatT, FloatT2, T>
            (pData, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, origSampleSize);
#else
        Sumup_noindices<FloatT, T>
            (pData, pSampleOutputs, pSumOutputsByBin, pCountByBin, origSampleSize);
#endif
        return;
    }
    for (int i = 0; i < origSampleSize; i++)
    {
        T featureValue = pData[pIndices[i]];

        FloatT output = pSampleOutputs[i];
        pSumOutputsByBin[featureValue] += output;
#ifdef IsWeighted
        FloatT2 output2 = pSampleOutputWeights[i];
        pSumWeightsByBin[featureValue] += output2;
#endif
        ++pCountByBin[featureValue];
    }
}

////////////////////// Delta Sparse

#ifdef IsWeighted
template<class FloatT, class FloatT2, class T>
#else
template<class FloatT, class T>
#endif
void SumupDeltaSparse_noindices(_In_reads_(numDeltas) T* pValues, _In_reads_(numDeltas) uint8_t* pDeltas,
    _In_ int32_t numDeltas, _In_reads_(numDeltas) FloatT* pSampleOutputs,
#ifdef IsWeighted
    _In_reads_(numDeltas) FloatT2* pSampleOutputWeights,
#endif
    _Inout_ FloatT* pSumOutputsByBin,
#ifdef IsWeighted
    _Inout_ FloatT2* pSumWeightsByBin,
#endif
    _Inout_ int32_t* pCountByBin, _In_ int32_t totalCount, _In_ double totalSampleOutputs
#ifdef IsWeighted
    , _In_ double totalSampleOutputWeights
#endif
    )
{
    double totalOutput = 0;
    double totalOutput2 = 0;
    int currentPos = 0;

    for (int i = 0; i < numDeltas; i++, pDeltas++, pValues++)
    {
        currentPos += *pDeltas;
        T featureValue = *pValues;
        FloatT output = pSampleOutputs[currentPos];
        pSumOutputsByBin[featureValue] += output;
        totalOutput += output;
#ifdef IsWeighted
        FloatT2 output2 = pSampleOutputWeights[currentPos];
        pSumWeightsByBin[featureValue] += output2;
        totalOutput2 += output2;
#endif
        ++pCountByBin[featureValue];
    }
    // Fixup the zeros. There were some zero items already placed in the zero-th entry, just add the remainder
    pSumOutputsByBin[0] += (FloatT)(totalSampleOutputs - totalOutput);
#ifdef IsWeighted
    pSumWeightsByBin[0] += (FloatT2)(totalSampleOutputWeights - totalOutput2);
#endif
    pCountByBin[0] += (totalCount)-numDeltas;
}

#ifdef IsWeighted
template<class FloatT, class FloatT2, class T>
#else
template<class FloatT, class T>
#endif
void SumupDeltaSparse(_In_ T* pValues, _In_ uint8_t* pDeltas, _In_ int32_t numDeltas,
    _In_reads_(totalCount) int32_t* pIndices, _In_ FloatT* pSampleOutputs,
#ifdef IsWeighted
    _In_ FloatT2* pSampleOutputWeights,
#endif
    _Inout_ FloatT* pSumOutputsByBin,
#ifdef IsWeighted
    _Inout_ FloatT2* pSumWeightsByBin,
#endif
    _Inout_ int32_t* pCountByBin, _In_ int32_t totalCount, _In_ double totalSampleOutputs
#ifdef IsWeighted
    , _In_ double totalSampleOutputWeights
#endif
    )
{
    if (pIndices == nullptr)
    {
#ifdef IsWeighted
        SumupDeltaSparse_noindices<FloatT, FloatT2, T>
            (pValues, pDeltas, numDeltas, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin,
            pCountByBin, totalCount, totalSampleOutputs, totalSampleOutputWeights);
#else
        SumupDeltaSparse_noindices<FloatT, T>
            (pValues, pDeltas, numDeltas, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount, totalSampleOutputs);
#endif
        return;
    }
    // In the corner case where numDeltas == 0, the corresponding delta array in the managed code is empty,
    // so pDeltas here will be a nullptr. Assign - 1 to currentPos so the following loop will have no effect.
    int currentPos = numDeltas > 0 ? pDeltas[0] : -1;

    int iDocIndices = 0, iSparse = 0, nonZeroCount = 0;
    FloatT totalOutput = 0;
#ifdef IsWeighted
    FloatT2 totalOutput2 = 0;
#endif

    for (;;)
    {
        if (currentPos < pIndices[iDocIndices])
        {
            if (++iSparse >= numDeltas)
                break;
            currentPos += pDeltas[iSparse];
        }
        else if (currentPos > pIndices[iDocIndices])
        {
            if (++iDocIndices >= totalCount)
                break;
        }
        else
        {
            // a nonzero entry matched one of the docs in the leaf, add it to the histogram
            T featureValue = pValues[iSparse];
            FloatT output = pSampleOutputs[iDocIndices];
            pSumOutputsByBin[featureValue] += output;
            totalOutput += output;
#ifdef IsWeighted
            FloatT2 output2 = pSampleOutputWeights[iDocIndices];
            pSumWeightsByBin[featureValue] += output2;
            totalOutput2 += output2;
#endif
            ++pCountByBin[featureValue];
            nonZeroCount++;

            if (++iSparse >= numDeltas)
                break;

            if (pDeltas[iSparse] > 0)
            {
                // If pDeltas was 0, we do not continue on.
                currentPos += pDeltas[iSparse];
                if (++iDocIndices >= totalCount)
                    break;
            }
        }
    }
    // Fixup the zeros. There were some zero items already placed in the zero-th entry, just add the remainder
    pSumOutputsByBin[0] += (FloatT)(totalSampleOutputs - totalOutput);
#ifdef IsWeighted
    pSumWeightsByBin[0] += (FloatT)(totalSampleOutputWeights - totalOutput2);
#endif
    pCountByBin[0] += totalCount - nonZeroCount;
}
