//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

// This file is expanded 2 times in ExpandFloatType.cpp  (once for floats and once for doubles)
#define CONCAT(x,y) x##y
#define EXPAND_SUMUP(suffix) CONCAT(C_Sumup_, suffix)
#define EXPAND_SUMUPDELTASPARSE(suffix) CONCAT(C_SumupDeltaSparse_, suffix)
#define EXPAND_SUMUPSEGMENT(suffix) CONCAT(C_SumupSegment_, suffix)

EXPORT_API(int) EXPAND_SUMUP(FloatType) (_In_ int32_t numBits, _In_ uint8_t* pData, _In_reads_(totalCount) int32_t* pIndices, _In_reads_(totalCount) FloatType* pSampleOutputs, _In_reads_opt_(totalCount) double* pSampleOutputWeights,
    _Inout_ FloatType* pSumOutputsByBin, _Inout_ double* pSumWeightsByBin, _Inout_  int32_t* pCountByBin,
    _In_ int32_t totalCount, _In_ double totalSampleOutputs, _In_ double totalSampleOutputWeights)
{
    if (pSampleOutputWeights == nullptr)
    {
        switch (numBits)
        {
        case 1:  SumupOneBit <FloatType, uint8_t>
            (pData, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount, totalCount, totalSampleOutputs); break;
        case 4:  SumupNibbles<FloatType, uint8_t>
            (pData, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount); break;
        case 8:  Sumup       <FloatType, uint8_t>
            (pData, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount); break;
        case 16: Sumup       <FloatType, uint16_t>
            ((uint16_t*)pData, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount); break;
        case 32: Sumup       <FloatType, int32_t>
            ((int32_t*)pData, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount); break;
        default: return -1; //Handle exception in C#
        }
    }
    else
    {
        switch (numBits)
        {
            //case 1:  SumupOneBit <FloatType, uint8_t>
            //             (pData, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount, totalCount, totalSampleOutputs); break;
        case 4:  SumupNibbles<FloatType, double, uint8_t>
            (pData, pIndices, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, totalCount); break;
        case 8:  Sumup       <FloatType, double, uint8_t>
            (pData, pIndices, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, totalCount); break;
        case 16: Sumup       <FloatType, double, uint16_t>
            ((uint16_t*)pData, pIndices, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, totalCount); break;
        case 32: Sumup       <FloatType, double, int32_t>
            ((int32_t*)pData, pIndices, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, totalCount); break;
        default: return -1; //Handle exception in C#
        }
    }
    return 0;
}

EXPORT_API(int) EXPAND_SUMUPDELTASPARSE(FloatType) (_In_ int32_t numBits, _In_ uint8_t* pValues, _In_ uint8_t* pDeltas,
    _In_ int32_t numDeltas, _In_reads_(totalCount) int32_t* pIndices, _In_ FloatType* pSampleOutputs, _In_opt_ double* pSampleOutputWeights,
    _Inout_ FloatType* pSumOutputsByBin, _Inout_ double* pSumWeightsByBin, _Inout_ int32_t* pCountByBin,
    _In_ int32_t totalCount, _In_ double totalSampleOutputs, _In_ double totalSampleOutputWeights)
{
    if (pSampleOutputWeights == nullptr)
    {
        switch (numBits)
        {
        case 8:  SumupDeltaSparse<FloatType, uint8_t>
            (pValues, pDeltas, numDeltas, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount, totalSampleOutputs); break;
        case 16: SumupDeltaSparse<FloatType, uint16_t>
            ((uint16_t*)pValues, pDeltas, numDeltas, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount, totalSampleOutputs); break;
        case 32: SumupDeltaSparse<FloatType, int32_t>
            ((int32_t*)pValues, pDeltas, numDeltas, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin,
                totalCount, totalSampleOutputs); break;
        default: return -1; //Handle exception in C#
        } return 0;
    }
    else
    {
        switch (numBits)
        {
        case 8:  SumupDeltaSparse<FloatType, double, uint8_t>
            (pValues, pDeltas, numDeltas, pIndices, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, totalCount, totalSampleOutputs, totalSampleOutputWeights); break;
        case 16: SumupDeltaSparse<FloatType, double, uint16_t>
            ((uint16_t*)pValues, pDeltas, numDeltas, pIndices, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, totalCount, totalSampleOutputs, totalSampleOutputWeights); break;
        case 32: SumupDeltaSparse<FloatType, double, int32_t>
            ((int32_t*)pValues, pDeltas, numDeltas, pIndices, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin,
                totalCount, totalSampleOutputs, totalSampleOutputWeights); break;
        default: return -1; //Handle exception in C#
        } return 0;
    }
}

EXPORT_API(int) EXPAND_SUMUPSEGMENT(FloatType) (_In_ uint32_t* pData, _In_ uint8_t* pSegType, _In_ int32_t* pSegLength,
    _In_reads_(totalCount) int32_t* pIndices, _In_reads_(totalCount) FloatType* pSampleOutputs, _In_reads_opt_(totalCount) double* pSampleOutputWeights, _Inout_ FloatType* pSumOutputsByBin,
    _Inout_ double* pSumWeightsByBin, _Inout_ int32_t* pCountByBin, _In_ int32_t totalCount, _In_ double totalSampleOutputs)
{
    if (pSampleOutputWeights == nullptr)
    {
        SumupSegment<FloatType, double>(pData, pSegType, pSegLength, pIndices, pSampleOutputs, pSumOutputsByBin, pCountByBin, totalCount);
    }
    else
    {
        SumupSegment<FloatType, double>(pData, pSegType, pSegLength, pIndices, pSampleOutputs, pSampleOutputWeights, pSumOutputsByBin, pSumWeightsByBin, pCountByBin, totalCount);
    }
    return 0;
}
