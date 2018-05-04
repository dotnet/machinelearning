//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

typedef unsigned char byte;

template<class FloatT, class T>
void SumupOneBit_noindices(_In_reads_(origSampleSize / 8) T* pData, _In_ FloatT* pSampleOutputs,
    _Inout_updates_all_(2) FloatT* pSumOutputsByBin, _Inout_updates_all_(2) int32_t* pCountByBin,
    _In_ int32_t origSampleSize, _In_ int32_t totalCount, _In_ double totalSampleOutputs)
{
    int numData = origSampleSize / 8;
    int remainingData = origSampleSize % 8;

    FloatT output1 = 0;
    int count1 = 0;

    int docIndex = 0;
    for (int i = 0; i < numData; i++)
    {
        byte v = pData[i];
        if (v == 0)
        {
            docIndex += 8;
            continue;
        }

        for (int j = 0; j < 8; j++)
        {
            FloatT output = pSampleOutputs[docIndex];
            if (v & 1)
            {
                output1 += output;
                count1++;
            }
            v >>= 1; docIndex++;
        }
        //if (v & 1){FloatT output = pSampleOutputs[docIndex]; output1 += output; outputSquared1 += output*output; count1++;}
        //if (v & 2){FloatT output = pSampleOutputs[docIndex+1]; output1 += output; outputSquared1 += output*output; count1++;}
        //if (v & 4){FloatT output = pSampleOutputs[docIndex+2]; output1 += output; outputSquared1 += output*output; count1++;}
        //if (v & 8){FloatT output = pSampleOutputs[docIndex+3]; output1 += output; outputSquared1 += output*output; count1++;}
        //if (v & 16){FloatT output = pSampleOutputs[docIndex+4]; output1 += output; outputSquared1 += output*output; count1++;}
        //if (v & 32){FloatT output = pSampleOutputs[docIndex+5]; output1 += output; outputSquared1 += output*output; count1++;}
        //if (v & 64){FloatT output = pSampleOutputs[docIndex+6]; output1 += output; outputSquared1 += output*output; count1++;}
        //if (v & 128){FloatT output = pSampleOutputs[docIndex+7]; output1 += output; outputSquared1 += output*output; count1++;}
        //docIndex += 8;
    }

    if (remainingData > 0)
    {
        byte v = pData[numData];
        for (int j = 0; j < remainingData; j++)
        {
            FloatT output = pSampleOutputs[docIndex];
            if (v & 1)
            {
                output1 += output;
                count1++;
            }
            v >>= 1; docIndex++;
        }
    }

    pSumOutputsByBin[0] = (FloatT)(totalSampleOutputs - output1);
    pCountByBin[0] = totalCount - count1;
    pSumOutputsByBin[1] = output1;
    pCountByBin[1] = count1;
}

template<class FloatT, class T>
void SumupOneBit(_In_ T* pData, _In_reads_(origSampleSize) int32_t* pIndices, _In_reads_(origSampleSize) FloatT* pSampleOutputs,
    _Inout_updates_all_(2) FloatT* pSumOutputsByBin, _Inout_updates_all_(2) int32_t* pCountByBin,
    _In_ int32_t origSampleSize, _In_ int32_t totalCount, _In_ double totalSampleOutputs)
{
    if (pIndices == nullptr)
    {
        SumupOneBit_noindices<FloatT, T>
            (pData, pSampleOutputs, pSumOutputsByBin, pCountByBin, origSampleSize, totalCount, totalSampleOutputs);
        return;
    }

    FloatT output1 = 0;
    int count1 = 0;
    for (int i = 0; i < origSampleSize; i++)
    {
        int dataIndex = pIndices[i] / 8;
        int bitIndex = pIndices[i] % 8;
        byte v = pData[dataIndex];
        v >>= bitIndex;
        if (v & 1)
        {
            FloatT output = pSampleOutputs[i];
            output1 += output;
            count1++;
        }
    }

    pSumOutputsByBin[0] = (FloatT)(totalSampleOutputs - output1);
    pCountByBin[0] = (totalCount)-count1;
    pSumOutputsByBin[1] = output1;
    pCountByBin[1] = count1;
}
