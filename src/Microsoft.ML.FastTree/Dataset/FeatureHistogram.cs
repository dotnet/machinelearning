// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
#if USE_SINGLE_PRECISION
    using FloatType = System.Single;
#else
    using FloatType = System.Double;
#endif

    /// <summary>
    /// Class to represent statistics of the feature used by LeastSquaresRegressionTreeLearner
    /// </summary>
    public sealed class FeatureHistogram
    {
        public readonly FloatType[] SumTargetsByBin;
        public readonly double[] SumWeightsByBin;
        public readonly int[] CountByBin;

        public readonly int NumFeatureValues;

        private readonly IntArray _bins;

        /// <summary>
        /// Make a new FeatureHistogram
        /// </summary>
        /// <param name="bins">The bins we will be calculating sumups over</param>
        /// <param name="numBins">The number of bins, should be at least as large as the number of bins</param>
        /// <param name="useWeights">Allocates weights array when true</param>
        public FeatureHistogram(IntArray bins, int numBins, bool useWeights)
        {
            Contracts.AssertValue(bins);
            Contracts.Assert(bins.Length == 0 || (0 <= numBins && bins.Max() < numBins));
            _bins = bins;

            NumFeatureValues = numBins;
            SumTargetsByBin = new FloatType[NumFeatureValues];
            CountByBin = new int[NumFeatureValues];
            if (useWeights)
                SumWeightsByBin = new double[NumFeatureValues];
        }

        /// <summary>
        /// This function returns the estimated memory used for a FeatureHistogram object according to given
        /// number of bins.
        /// </summary>
        /// <param name="numBins">number of bins</param>
        /// <param name="hasWeights">weights array is counted when true</param>
        /// <returns>estimated size of memory used for a feature histogram object</returns>
        public static int EstimateMemoryUsedForFeatureHistogram(int numBins, bool hasWeights)
        {
            return sizeof(int) // NumberFeatureValues
                + sizeof(int) // the IsSplittable boolean value. Although sizeof(bool) is 1,
                // but we just estimate it as 4 for alignment
                + 8 // size of reference to _feature in 64 bit machines.
                + sizeof(int) * numBins // CountByBin
                + sizeof(FloatType) * numBins // SumTargetsByBin
                + (hasWeights ? sizeof(double) * numBins : 0); // SumWeightsByBin
        }

        /// <summary>
        /// Subtract from myself the counts of the child histogram
        /// </summary>
        /// <param name="child">Another histogram to subtract</param>
        public unsafe void Subtract(FeatureHistogram child)
        {
            if (child.NumFeatureValues != NumFeatureValues)
                throw Contracts.Except("cannot subtract FeatureHistograms of different lengths");

            fixed (FloatType* pSumTargetsByBin = SumTargetsByBin)
            fixed (FloatType* pChildSumTargetsByBin = child.SumTargetsByBin)
            fixed (double* pSumWeightsByBin = SumWeightsByBin)
            fixed (double* pChildSumWeightsByBin = child.SumWeightsByBin)
            fixed (int* pTotalCountByBin = CountByBin)
            fixed (int* pChildTotalCountByBin = child.CountByBin)
            {
                if (pSumWeightsByBin == null)
                {
                    for (int i = 0; i < NumFeatureValues; i++)
                    {
                        pSumTargetsByBin[i] -= pChildSumTargetsByBin[i];
                        pTotalCountByBin[i] -= pChildTotalCountByBin[i];
                    }
                }
                else
                {
                    Contracts.Assert(pChildSumWeightsByBin != null);
                    for (int i = 0; i < NumFeatureValues; i++)
                    {
                        pSumTargetsByBin[i] -= pChildSumTargetsByBin[i];
                        pSumWeightsByBin[i] -= pChildSumWeightsByBin[i];
                        pTotalCountByBin[i] -= pChildTotalCountByBin[i];
                    }
                }
            }
        }

        public void Sumup(int numDocsInLeaf, double sumTargets, FloatType[] outputs, int[] docIndices)
        {
            SumupWeighted(numDocsInLeaf, sumTargets, 0.0, outputs, null, docIndices);
        }

        public void SumupWeighted(int numDocsInLeaf, double sumTargets, double sumWeights, FloatType[] outputs, double[] weights, int[] docIndices)
        {
            using (Timer.Time(TimerEvent.Sumup))
            {
#if TLC_REVISION
                Array.Clear(SumWeightedTargetsByBin, 0, SumWeightedTargetsByBin.Length);
#else
                Array.Clear(SumTargetsByBin, 0, SumTargetsByBin.Length);
#endif

                if (SumWeightsByBin != null)
                {
                    Array.Clear(SumWeightsByBin, 0, SumWeightsByBin.Length);
                }

                Array.Clear(CountByBin, 0, CountByBin.Length);

                if (numDocsInLeaf > 0)
                {
                    SumupInputData input = new SumupInputData(
                        numDocsInLeaf,
                        sumTargets,
                        sumWeights,
                        outputs,
                        weights,
                        docIndices);

                    _bins.Sumup(input, this);
                }
            }
        }
    }

    public sealed class SumupInputData
    {
        public int TotalCount;
        public double SumTargets;
        public readonly FloatType[] Outputs;
        public readonly int[] DocIndices;
        public double SumWeights;
        public readonly double[] Weights;

        public SumupInputData(int totalCount, double sumTargets, double sumWeights,
            FloatType[] outputs, double[] weights, int[] docIndices)
        {
            TotalCount = totalCount;
            SumTargets = sumTargets;
            Outputs = outputs;
            DocIndices = docIndices;
            SumWeights = sumWeights;
            Weights = weights;
        }
    }

}
