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

EXPORT_API(void) C_GetDerivatives(
    _In_ int numDocuments, _In_ int begin,
    _In_reads_(numDocuments) int *pPermutation, _In_ short *pLabels,
    _In_ double *pScores, _Inout_ double *pLambdas, _Inout_ double *pWeights, _In_ double *pDiscount,
    _In_ double inverseMaxDCG, _In_ double *pGainLabels,
    _In_ double secondaryMetricShare, _In_ bool secondaryExclusive, _In_ double secondaryInverseMaxDCG, _In_ double *pSecondaryGains,
    _In_reads_(sigmoidTableLength) double *sigmoidTable, _In_ double minScore, _In_ double maxScore, _In_ int sigmoidTableLength, _In_ double scoreToSigmoidTableFactor,
    _In_ char costFunctionParam, _In_ bool distanceWeight2, _In_ int numActualDocuments, _Inout_ double *pLambdaSum, _In_ double minDoubleValue,
    _In_ double alphaRisk, _In_ double baselineVersusCurrentDcg
    )
{
    // These arrays are shared among many threads, "begin" is the offset by which all arrays are indexed. 
    //  So we shift them all here to avoid having to add 'begin' to every pointer below.
    pLabels += begin;
    pScores += begin;
    pLambdas += begin;
    pWeights += begin;
    pGainLabels += begin;

    if (secondaryMetricShare != 0)
    {
        pSecondaryGains += begin;
    }

    double bestScore = pScores[pPermutation[0]];

    int worstIndexToConsider = numDocuments - 1;
    while (worstIndexToConsider > 0 && pScores[pPermutation[worstIndexToConsider]] == minDoubleValue)
    {
        worstIndexToConsider--;
    }
    double worstScore = pScores[pPermutation[worstIndexToConsider]];

    *pLambdaSum = 0;

    // Should we still run the calculation on those pairs which are ostensibly the same?
    bool pairSame = secondaryMetricShare != 0.0;

    // Did not help to use pointer match on pPermutation[i]
    for (int i = 0; i < numDocuments; ++i)
    {
        int high = pPermutation[i];
        // We are going to loop through all pairs where label[high] > label[low]. If label[high] is 0, it can't be larger
        // If score[high] is Double.MinValue, it's being discarded by shifted NDCG
        if ((pLabels[high] == 0 && !pairSame) || pScores[high] == minDoubleValue) continue;
        // These variables are all looked up just once per loop of 'i', so do it here.
        double gainLabelHigh = pGainLabels[high];
        short labelHigh = pLabels[high];
        double scoreHigh = pScores[high];
        double discountI = pDiscount[i];
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
            // If score[low] is Double.MinValue, it's being discarded by shifted NDCG
            int low = *ppPermutationJ;
            if ((pairSame ? labelHigh < pLabels[low] : labelHigh <= pLabels[low]) || pScores[low] == minDoubleValue) continue;

            double scoreHighMinusLow = scoreHigh - pScores[low];
            if (secondaryMetricShare == 0.0 && labelHigh == pLabels[low] && scoreHighMinusLow <= 0) continue;
            double dcgGap = gainLabelHigh - pGainLabels[low];
            double currentInverseMaxDCG = inverseMaxDCG * (1.0 - secondaryMetricShare);

            // Handle risk w.r.t. baseline.
            double pairedDiscount = fabs((discountI - *ppDiscountJ));
            if (alphaRisk > 0)
            {
                double risk = 0.0, baselineDenorm = baselineVersusCurrentDcg / pairedDiscount;
                if (baselineVersusCurrentDcg > 0)
                {
                    // The baseline is currently higher than the model.
                    // If we're ranked incorrectly, we can only reduce risk only as much as the baseline current DCG.
                    risk = scoreHighMinusLow <= 0 && dcgGap > baselineDenorm ? baselineDenorm : dcgGap;
                }
                else if (scoreHighMinusLow > 0)
                {
                    // The baseline is currently lower, but this pair is ranked correctly.
                    risk = baselineDenorm + dcgGap;
                }
                if (risk > 0)
                {
                    dcgGap += alphaRisk * risk;
                }
            }

            bool sameLabel = labelHigh == pLabels[low];

            // calculate the lambdaP for this pair by looking it up in the sigmoidTable (for example, computed in FastTreeRanking.FillSigmoidTable)
            double lambdaP;
            if (scoreHighMinusLow <= minScore) lambdaP = sigmoidTable[0];
            else if (scoreHighMinusLow >= maxScore) lambdaP = sigmoidTable[sigmoidTableLength - 1];
            else lambdaP = sigmoidTable[(int)((scoreHighMinusLow - minScore) * scoreToSigmoidTableFactor)];

            double weightP = lambdaP * (2.0 - lambdaP);

            if (secondaryMetricShare != 0.0)
            {
                if (sameLabel || currentInverseMaxDCG == 0.0)
                {
                    if (pSecondaryGains[high] <= pSecondaryGains[low]) continue;
                    // We should use the secondary metric this time.
                    dcgGap = pSecondaryGains[high] - pSecondaryGains[low];
                    currentInverseMaxDCG = secondaryInverseMaxDCG * secondaryMetricShare;
                    sameLabel = false;
                }
                else if (!secondaryExclusive && pSecondaryGains[high] > pSecondaryGains[low])
                {
                    double sIDCG = secondaryInverseMaxDCG * secondaryMetricShare;
                    dcgGap = dcgGap / sIDCG + (pSecondaryGains[high] - pSecondaryGains[low]) / currentInverseMaxDCG;
                    currentInverseMaxDCG *= sIDCG;
                }
            }

            //printf("%d-%d : gap %g, currentinv %g\n", high, low, (float)dcgGap, (float)currentInverseMaxDCG); fflush(stdout);

            // calculate the deltaNDCGP for this pair
            double deltaNDCGP = dcgGap * pairedDiscount * currentInverseMaxDCG;

            // apply distanceWeight2 only to regular pairs
            if (!sameLabel && distanceWeight2 && bestScore != worstScore)
            {
                deltaNDCGP /= (.01 + fabs(pScores[high] - pScores[low]));
            }

            // update lambdas and weights
            deltaLambdasHigh += lambdaP * deltaNDCGP;
            deltaWeightsHigh += weightP * deltaNDCGP;
            pLambdas[low] -= lambdaP * deltaNDCGP;
            pWeights[low] += weightP * deltaNDCGP;

            *pLambdaSum += 2 * lambdaP * deltaNDCGP;
        }
        // Finally, add the values for the high part of the pair that we accumulated across all the low parts
        pLambdas[high] += deltaLambdasHigh;
        pWeights[high] += deltaWeightsHigh;
    }
}
