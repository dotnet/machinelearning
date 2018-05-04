//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#include "stdafx.h"
#include <math.h>
#include <iostream> // for qsort
#include <stdlib.h>    // for qsort
#include <search.h> // for qsort

// C++ Version of GetDerivatives

EXPORT_API(void) C_GetSurplusDerivatives(
    _In_ int numDocuments, _In_ int begin,
    _In_reads_(numDocuments) int *pPermutation, _In_ short *pLabels,
    _In_ double *pScores, _Inout_ double *pLambdas, _Inout_ double *pWeights, _In_ double *pDiscount,
    _In_ double *pGainLabels,
    _In_reads_(sigmoidTableLength) double *sigmoidTable, _In_ double minScore, _In_ double maxScore, _In_ int sigmoidTableLength, _In_ double scoreToSigmoidTableFactor,
    _In_ char costFunctionParam, _In_ bool distanceWeight2, _Inout_ double *pLambdaSum, _In_ double minDoubleValue
    )
{
    // These arrays are shared among many threads, "begin" is the offset by which all arrays are indexed. 
    //  So we shift them all here to avoid having to add 'begin' to every pointer below.
    pLabels += begin;
    pScores += begin;
    pLambdas += begin;
    pWeights += begin;
    pGainLabels += begin;

    if (costFunctionParam != 's')
        return;

    double bestScore = pScores[pPermutation[0]];

    int worstIndexToConsider = numDocuments - 1;
    while (worstIndexToConsider > 0 && pScores[pPermutation[worstIndexToConsider]] == minDoubleValue)
    {
        worstIndexToConsider--;
    }
    double worstScore = pScores[pPermutation[worstIndexToConsider]];

    *pLambdaSum = 0;

    int maxWinLossSurplusCutoff = 0;
    //Stuff specific to SurplusMART:
    int currentWinLossSurplus = 0;
    int maxWinLossSurplus = 0;
    //For WinLossSurplus cost we need to calculate the max WinLosSurplus cutoff document.
    for (int i = 0; i < numDocuments; ++i)
    {
        if (pLabels[pPermutation[i]] > 0 ||
            pLabels[pPermutation[i]] < 0)
            currentWinLossSurplus += pLabels[pPermutation[i]];
        else
            currentWinLossSurplus--;

        if (currentWinLossSurplus > maxWinLossSurplus)
        {
            maxWinLossSurplus = currentWinLossSurplus;
            //save 'i', not 'pPermutation[i]' since we want the index into the sorted array
            maxWinLossSurplusCutoff = i;
        }
    }
    //Special case when the surplus cutoff is very near either end
    //The trainer needs to bootstrap the scoring
    //So use a lower cutoff
    if (maxWinLossSurplusCutoff <= numDocuments / 20)
    {
        maxWinLossSurplusCutoff = numDocuments / 20 + 1;
    }
    if (maxWinLossSurplusCutoff >= 19 * numDocuments / 20)
    {
        maxWinLossSurplusCutoff = 19 * numDocuments / 20;
    }
    //end of SurplusMART specific

    // Did not help to use pointer match on pPermutation[i]
    for (int i = 0; i < numDocuments; ++i)
    {
        int high = pPermutation[i];
        // We are going to loop through all pairs where label[high] > label[low]. If label[high] is 0, it can't be larger
        // If score[high] is Double.MinValue, it's being discarded by shifted WinLossSurplus
        if (pLabels[high] == 0 || pScores[high] == minDoubleValue) continue;
        // These variables are all looked up just once per loop of 'i', so do it here.
        short labelHigh = pLabels[high];
        double scoreHigh = pScores[high];
        // These variables will store the accumulated lambda and weight difference for high, which saves time
        double deltaLambdasHigh = 0;
        double deltaWeightsHigh = 0;

        //The below is effectively: for (int j = 0; j < numDocuments; ++j)
        double *ppDiscountJ = pDiscount;
        for (int *ppPermutationJ = pPermutation;
        ppPermutationJ < pPermutation + numDocuments;
            ppPermutationJ++, ppDiscountJ++)
        {
            // only consider pairs with different labels, where "high" has a higher label than "low"
            // If score[low] is Double.MinValue, it's being discarded by shifted WinLossSurplus
            int low = *ppPermutationJ;
            if (labelHigh <= pLabels[low] || pScores[low] == minDoubleValue) continue;

            double scoreHighMinusLow = scoreHigh - pScores[low];

            // calculate the lambdaP for this pair by looking it up in the lambdaTable (computed in LambdaMart.FillLambdaTable)
            double lambdaP;
            if (scoreHighMinusLow <= minScore)
                lambdaP = sigmoidTable[0];
            else if (scoreHighMinusLow >= maxScore)
                lambdaP = sigmoidTable[sigmoidTableLength - 1];
            else
                lambdaP = sigmoidTable[(int)((scoreHighMinusLow - minScore) * scoreToSigmoidTableFactor)];

            double weightP = lambdaP * (2.0 - lambdaP);

            // calculate the SurplusMART lambdas for this pair
            double deltaWinLossSurplusP;
            int orderj;
            int quartilei;
            int quartilej;

            //Use the WinLossSurplus cost function
            //so Delta WinLossSurplus is actually Delta WinLossSurplus
            orderj = (int)(pPermutation - ppPermutationJ);
            deltaWinLossSurplusP = 0;
            quartilei = 4 * i / numDocuments;
            quartilej = 4 * orderj / numDocuments;
            //WinLossSurplus only changes is we are moving something across the maxWinLossSurplus cutoff point
            if ((maxWinLossSurplusCutoff >= i && maxWinLossSurplusCutoff < orderj) ||
                (maxWinLossSurplusCutoff < i && maxWinLossSurplusCutoff >= orderj))
                deltaWinLossSurplusP = labelHigh - pLabels[low];
            else if (quartilei != quartilej)
                deltaWinLossSurplusP = (labelHigh - pLabels[low])*0.75;

            if (distanceWeight2 && bestScore != worstScore)
            {
                deltaWinLossSurplusP /= (.01 + fabs(pScores[high] - pScores[low]));
            }

            // update lambdas and weights
            deltaLambdasHigh += lambdaP * deltaWinLossSurplusP;
            deltaWeightsHigh += weightP * deltaWinLossSurplusP;
            pLambdas[low] -= lambdaP * deltaWinLossSurplusP;
            pWeights[low] += weightP * deltaWinLossSurplusP;

            *pLambdaSum += 2 * lambdaP * deltaWinLossSurplusP;
        }
        // Finally, add the values for the high part of the pair that we accumulated across all the low parts
        pLambdas[high] += deltaLambdasHigh;
        pWeights[high] += deltaWeightsHigh;
    }
}




