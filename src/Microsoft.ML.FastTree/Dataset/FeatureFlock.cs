// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#if USE_SINGLE_PRECISION
using FloatType = System.Single;
#else
using FloatType = System.Double;
#endif

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    /// <summary>
    /// Holds statistics per bin value for a feature. These are yielded by <see cref="SufficientStatsBase.GetBinStats"/>
    /// to indicate after a <see cref="SufficientStatsBase.Sumup"/> call over a subset of the dataset. These statistics
    /// are then used in <see cref="LeastSquaresRegressionTreeLearner"/> to find splitting on which bin will yield the
    /// best least squares solution
    /// </summary>
    public struct PerBinStats
    {
        /// <summary>Sum of all target values in a partition for the bin.</summary>
        public readonly Double SumTargets;
        /// <summary>Sum of all target weights in a partition. May be 0 if we are not doing weighted training.</summary>
        public readonly Double SumWeights;
        /// <summary>Count of the documents in this partition for the bin.</summary>
        public readonly int Count;

        public PerBinStats(Double sumTargets, Double sumWeights, int count)
        {
            Contracts.Assert(count >= 0);

            SumTargets = sumTargets;
            SumWeights = sumWeights;
            Count = count;
        }
    }

    /// <summary>
    /// These objects are stateful, reusable objects that enable the collection of sufficient
    /// stats per feature flock, per node or leaf of a tree, to enable it to find the "best"
    /// splits.
    ///
    /// Each instance of this corresponds to a single flock, but multiple of these will be created
    /// per flock. Note that feature indices, whenever present, refer to the feature within the
    /// particular flock the same as they do with <see cref="FeatureFlockBase"/>.
    /// </summary>
    public abstract class SufficientStatsBase
    {
        // REVIEW: Holdover from histogram. I really don't like this. Figure out if
        // there's a better way.
        /// <summary>
        /// An array as large as there are count of features in the corresponding flock. Used by
        /// <see cref="LeastSquaresRegressionTreeLearner"/> to indicate whether a particular
        /// feature has been judged to be potentially splittable or not.
        /// </summary>
        public readonly bool[] IsSplittable;

#if DEBUG
        /// <summary>
        /// One per feature in the corresponding flock. Tracks the last sumup, whether the feature
        /// was marked as "active" or not. Used in debug builds only, to track the invariants w.r.t.
        /// calls of <see cref="Sumup"/> and <see cref="Subtract"/>.
        /// </summary>
        private readonly bool[] _active;
#endif

        public abstract FeatureFlockBase Flock { get; }

        protected SufficientStatsBase(int features)
        {
            Contracts.Assert(features > 0);
            IsSplittable = new bool[features];
#if DEBUG
            _active = new bool[features];
#endif
        }

        /// <summary>
        /// Performs the accumulation of sufficient statistics for active features within a flock.
        /// </summary>
        /// <param name="featureOffset">Offset into <paramref name="active"/> where we should start querying active stats</param>
        /// <param name="active">The indicator array of whether features are active or not, logically starting for
        /// this flock at <paramref name="featureOffset"/>, where after this </param>
        /// <param name="numDocsInLeaf">Minimum documents total in this leaf</param>
        /// <param name="sumTargets">The sum of the targets for this leaf</param>
        /// <param name="sumWeights">The sum of the weights for this leaf</param>
        /// <param name="outputs">The target values, indexed by <paramref name="numDocsInLeaf"/></param>
        /// <param name="weights"></param>
        /// <param name="docIndices">The first <paramref name="numDocsInLeaf"/> entries indicate the row indices
        /// in this leaf, and these row indices are used to </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Sumup(
            int featureOffset,
            bool[] active,
            int numDocsInLeaf,
            double sumTargets,
            double sumWeights,
            FloatType[] outputs,
            double[] weights,
            int[] docIndices)
        {
#if DEBUG
            Contracts.AssertValueOrNull(active);
            Contracts.Assert(active == null || (0 <= featureOffset && featureOffset <= Utils.Size(active) - Flock.Count));
            Contracts.Assert(_active.Length == Flock.Count);
            // Preserve the active status of the sumup.
            if (active == null)
            {
                for (int f = 0; f < _active.Length; ++f)
                    _active[f] = true;
            }
            else
                Array.Copy(active, featureOffset, _active, 0, _active.Length);
#endif
            SumupCore(featureOffset, active, numDocsInLeaf, sumTargets, sumWeights, outputs, weights, docIndices);
        }

        /// <summary>
        /// The core implementation called from <see cref="Sumup"/>.
        /// </summary>
        protected abstract void SumupCore(
            int featureOffset,
            bool[] active,
            int numDocsInLeaf,
            double sumTargets,
            double sumWeights,
            FloatType[] outputs,
            double[] weights,
            int[] docIndices);

        /// <summary>
        /// Subtracts one sufficient statistics from another. Note that this other
        /// sufficient statistics object must be over the same feature flock in order
        /// to be meaningful, as well as have undergone <see cref="Sumup"/> under
        /// the same set of active features.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Subtract(SufficientStatsBase other)
        {
#if DEBUG
            Contracts.Assert(Flock == other.Flock);
            Contracts.Assert(_active.Length == other._active.Length);
            // One sufficient statistics object can contractually only be subtracted from
            // another, if both at some previous point were initialized with a sumup with
            // the same set of features active. Otherwise this operation is not meaningful.
            for (int f = 0; f < _active.Length; ++f)
                Contracts.Assert(_active[f] == other._active[f]);
#endif
            SubtractCore(other);
        }

        protected abstract void SubtractCore(SufficientStatsBase other);

        /// <summary>
        /// An approximation of the size in bytes used by this structure. Used for tracking
        /// and memory size estimation purposes.
        /// </summary>
        public abstract long SizeInBytes();

        // Returns first bin index for a given feature in histogram
        protected abstract int GetMinBorder(int featureIndexInFlock);

        // Returns last bin index for a given feature in histogram
        protected abstract int GetMaxBorder(int featureIndex);

        protected abstract PerBinStats GetBinStats(int featureIndex);

        protected abstract double GetBinGradient(int featureIndex, double bias);

        /// <summary>
        /// Get a fullcopy of histogram for one sub feature.
        /// </summary>
        public void CopyFeatureHistogram(int subfeatureIndex, ref PerBinStats[] hist)
        {
            int min = GetMinBorder(subfeatureIndex);
            int max = GetMaxBorder(subfeatureIndex);
            int len = max - min + 1;
            Utils.EnsureSize(ref hist, len);

            for (int i = 0; i < len; ++i)
                hist[i] = GetBinStats(min + i);

        }

        public void FillSplitCandidates(LeastSquaresRegressionTreeLearner learner, LeastSquaresRegressionTreeLearner.LeafSplitCandidates leafSplitCandidates,
            int flock, int[] featureUseCount, double featureFirstUsePenalty, double featureReusePenalty, double minDocsInLeaf,
            bool hasWeights, double gainConfidenceInSquaredStandardDeviations, double entropyCoefficient)
        {
            int featureMin = learner.TrainData.FlockToFirstFeature(flock);
            int featureLim = featureMin + learner.TrainData.Flocks[flock].Count;
            foreach (var feature in learner.GetActiveFeatures(featureMin, featureLim))
            {
                int subfeature = feature - featureMin;
                Contracts.Assert(0 <= subfeature && subfeature < Flock.Count);
                Contracts.Assert(subfeature <= feature);
                Contracts.Assert(learner.TrainData.FlockToFirstFeature(flock) == feature - subfeature);

                if (!IsSplittable[subfeature])
                    continue;

                Contracts.Assert(featureUseCount[feature] >= 0);

                double trust = learner.TrainData.Flocks[flock].Trust(subfeature);
                double usePenalty = (featureUseCount[feature] == 0) ?
                featureFirstUsePenalty : featureReusePenalty * Math.Log(featureUseCount[feature] + 1);
                int totalCount = leafSplitCandidates.NumDocsInLeaf;
                double sumTargets = leafSplitCandidates.SumTargets;
                double sumWeights = leafSplitCandidates.SumWeights;

                FindBestSplitForFeature(learner, leafSplitCandidates, totalCount, sumTargets, sumWeights,
                    feature, flock, subfeature, minDocsInLeaf,
                    hasWeights, gainConfidenceInSquaredStandardDeviations, entropyCoefficient,
                    trust, usePenalty);

                if (leafSplitCandidates.FlockToBestFeature != null)
                {
                    if (leafSplitCandidates.FlockToBestFeature[flock] == -1 ||
                        leafSplitCandidates.FeatureSplitInfo[leafSplitCandidates.FlockToBestFeature[flock]].Gain <
                        leafSplitCandidates.FeatureSplitInfo[feature].Gain)
                    {
                        leafSplitCandidates.FlockToBestFeature[flock] = feature;
                    }
                }
            }
        }

        internal void FindBestSplitForFeature(ILeafSplitStatisticsCalculator leafCalculator,
            LeastSquaresRegressionTreeLearner.LeafSplitCandidates leafSplitCandidates,
            int totalCount, double sumTargets, double sumWeights,
            int featureIndex, int flockIndex, int subfeatureIndex, double minDocsInLeaf,
            bool hasWeights, double gainConfidenceInSquaredStandardDeviations, double entropyCoefficient,
            double trust, double usePenalty)
        {
            double minDocsForThis = minDocsInLeaf / trust;
            double bestSumGTTargets = double.NaN;
            double bestSumGTWeights = double.NaN;
            double bestShiftedGain = double.NegativeInfinity;
            const double eps = 1e-10;
            int bestGTCount = -1;
            double sumGTTargets = 0.0;
            double sumGTWeights = eps;
            int gtCount = 0;
            sumWeights = 2 * eps;
            double gainShift = leafCalculator.GetLeafSplitGain(totalCount, sumTargets, sumWeights);

            // We get to this more explicit handling of the zero case since, under the influence of
            // numerical error, especially under single precision, the histogram computed values can
            // be wildly inaccurate even to the point where 0 unshifted gain may become a strong
            // criteria.
            double minShiftedGain = gainConfidenceInSquaredStandardDeviations <= 0 ? 0.0 :
               (gainConfidenceInSquaredStandardDeviations * leafSplitCandidates.VarianceTargets
               * totalCount / (totalCount - 1) + gainShift);

            // re-evaluate if the histogram is splittable
            IsSplittable[subfeatureIndex] = false;
            int t = Flock.BinCount(subfeatureIndex);
            uint bestThreshold = (uint)t;
            t--;
            int min = GetMinBorder(subfeatureIndex);
            int max = GetMaxBorder(subfeatureIndex);
            for (int b = max; b >= min; --b)
            {
                var binStats = GetBinStats(b);
                t--;
                sumGTTargets += binStats.SumTargets;
                if (hasWeights)
                    sumGTWeights += binStats.SumWeights;
                gtCount += binStats.Count;

                // Advance until GTCount is high enough.
                if (gtCount < minDocsForThis)
                    continue;
                int lteCount = totalCount - gtCount;

                // If LTECount is too small, we are finished.
                if (lteCount < minDocsForThis)
                    break;

                // Calculate the shifted gain, including the LTE child.
                double currentShiftedGain = leafCalculator.GetLeafSplitGain(gtCount, sumGTTargets, sumGTWeights)
                    + leafCalculator.GetLeafSplitGain(lteCount, sumTargets - sumGTTargets, sumWeights - sumGTWeights);

                // Test whether we are meeting the min shifted gain confidence criteria for this split.
                if (currentShiftedGain < minShiftedGain)
                    continue;

                // If this point in the code is reached, the histogram is splittable.
                IsSplittable[subfeatureIndex] = true;

                if (entropyCoefficient > 0)
                {
                    // Consider the entropy of the split.
                    double entropyGain = (totalCount * Math.Log(totalCount) - lteCount * Math.Log(lteCount) - gtCount * Math.Log(gtCount));
                    currentShiftedGain += entropyCoefficient * entropyGain;
                }

                // Is t the best threshold so far?
                if (currentShiftedGain > bestShiftedGain)
                {
                    bestGTCount = gtCount;
                    bestSumGTTargets = sumGTTargets;
                    bestSumGTWeights = sumGTWeights;
                    bestThreshold = (uint)t;
                    bestShiftedGain = currentShiftedGain;
                }
            }
            // set the appropriate place in the output vectors
            leafSplitCandidates.FeatureSplitInfo[featureIndex].CategoricalSplit = false;
            leafSplitCandidates.FeatureSplitInfo[featureIndex].Feature = featureIndex;
            leafSplitCandidates.FeatureSplitInfo[featureIndex].Threshold = bestThreshold;
            leafSplitCandidates.FeatureSplitInfo[featureIndex].LteOutput = leafCalculator.CalculateSplittedLeafOutput(totalCount - bestGTCount, sumTargets - bestSumGTTargets, sumWeights - bestSumGTWeights);
            leafSplitCandidates.FeatureSplitInfo[featureIndex].GTOutput = leafCalculator.CalculateSplittedLeafOutput(bestGTCount, bestSumGTTargets, bestSumGTWeights);
            leafSplitCandidates.FeatureSplitInfo[featureIndex].LteCount = totalCount - bestGTCount;
            leafSplitCandidates.FeatureSplitInfo[featureIndex].GTCount = bestGTCount;

            leafSplitCandidates.FeatureSplitInfo[featureIndex].Gain = (bestShiftedGain - gainShift) * trust - usePenalty;
            double erfcArg = Math.Sqrt((bestShiftedGain - gainShift) * (totalCount - 1) / (2 * leafSplitCandidates.VarianceTargets * totalCount));
            leafSplitCandidates.FeatureSplitInfo[featureIndex].GainPValue = ProbabilityFunctions.Erfc(erfcArg);
        }

        public void FillSplitCandidatesCategorical(LeastSquaresRegressionTreeLearner learner,
            LeastSquaresRegressionTreeLearner.LeafSplitCandidates leafSplitCandidates,
            int flock, int[] featureUseCount, double featureFirstUsePenalty, double featureReusePenalty,
            double minDocsInLeaf,
            bool hasWeights, double gainConfidenceInSquaredStandardDeviations, double entropyCoefficient)
        {
            int featureMin = learner.TrainData.FlockToFirstFeature(flock);
            int featureLim = featureMin + learner.TrainData.Flocks[flock].Count;

            //REVIEW: Should we not consider all the features if at least one feature is randomly chosen to not be active
            //when feature fraction is less than 1.0?
            List<int> features = learner.GetActiveFeatures(featureMin, featureLim).ToList();
            if (features.Count == 0 || !IsSplittable[0])
                return;
#if DEBUG
            foreach (var feature in features)
            {
                int subfeature = feature - featureMin;
                int min = GetMinBorder(subfeature);
                int max = GetMaxBorder(subfeature);
                Contracts.Assert(min == max);
            }
#endif
            //Use this feature to represent the flock.
            int firstFlockFeature = features[0];
            int lastFlockFeature = features[features.Count - 1];

            //REVIEW: What about the zero bin that contains missing values?
            features.Sort((a, b) =>
            {
                double gradientA = GetBinGradient(GetMinBorder(a - featureMin), learner.Bias);
                double gradientB = GetBinGradient(GetMinBorder(b - featureMin), learner.Bias);
                return gradientB.CompareTo(gradientA);
            });

            double usePenalty = (featureUseCount[firstFlockFeature] == 0)
                ? featureFirstUsePenalty
                : featureReusePenalty * Math.Log(featureUseCount[firstFlockFeature] + 1);

            double bestSumGTTargets = double.NaN;
            double bestSumGTWeights = double.NaN;
            double bestShiftedGain = double.NegativeInfinity;
            const double eps = 1e-10;
            int bestGTCount = -1;
            double sumGTTargets = 0.0;
            double sumGTWeights = eps;
            int gtCount = 0;
            int totalCount = leafSplitCandidates.NumDocsInLeaf;
            double sumTargets = leafSplitCandidates.SumTargets;
            double sumWeights = leafSplitCandidates.SumWeights + 2 * eps;
            double gainShift = learner.GetLeafSplitGain(totalCount, sumTargets, sumWeights);
            double trust = learner.TrainData.Flocks[flock].Trust(0);
            int bestThreshold = -1;
            double minDocsForThis = minDocsInLeaf / trust;
            // We get to this more explicit handling of the zero case since, under the influence of
            // numerical error, especially under single precision, the histogram computed values can
            // be wildly inaccurate even to the point where 0 unshifted gain may become a strong
            // criteria.
            double minShiftedGain = gainConfidenceInSquaredStandardDeviations <= 0
                ? 0.0
                : (gainConfidenceInSquaredStandardDeviations * leafSplitCandidates.VarianceTargets
                   * totalCount / (totalCount - 1) + gainShift);

            int maxGroups = learner.MaxCategoricalGroupsPerNode;
            int remainingGroups = maxGroups - 1;
            int minDocsPerGroup = (totalCount - gtCount) / maxGroups;
            int docsInCurrentGroup = 0;
            IsSplittable[0] = false;
            for (int i = 0; i < features.Count && i < learner.MaxCategoricalSplitPointsPerNode; ++i)
            {
                int feature = features[i];
                int subfeature = feature - featureMin;
                Contracts.Assert(0 <= subfeature && subfeature < Flock.Count);
                Contracts.Assert(subfeature <= feature);
                Contracts.Assert(learner.TrainData.FlockToFirstFeature(flock) == feature - subfeature);
                Contracts.Assert(featureUseCount[feature] >= 0);
                Contracts.Assert(Flock.BinCount(subfeature) == 2);
                Contracts.Assert(GetMaxBorder(subfeature) == GetMinBorder(subfeature));

                var binStats = GetBinStats(GetMinBorder(subfeature));
                sumGTTargets += binStats.SumTargets;
                if (hasWeights)
                    sumGTWeights += binStats.SumWeights;
                gtCount += binStats.Count;
                docsInCurrentGroup += binStats.Count;

                // Advance until GTCount is high enough.
                if (gtCount < minDocsForThis || docsInCurrentGroup < minDocsPerGroup)
                    continue;

                docsInCurrentGroup = 0;

                minDocsPerGroup = remainingGroups > 1 ? (totalCount - gtCount) / --remainingGroups : 0;

                int lteCount = totalCount - gtCount;

                // If LTECount is too small, we are finished.
                if (lteCount < minDocsForThis)
                    continue;

                // Calculate the shifted gain, including the LTE child.
                double currentShiftedGain = learner.GetLeafSplitGain(gtCount, sumGTTargets, sumGTWeights)
                                            +
                                            learner.GetLeafSplitGain(lteCount, sumTargets - sumGTTargets,
                                                sumWeights - sumGTWeights);

                // Test whether we are meeting the min shifted gain confidence criteria for this split.
                if (currentShiftedGain < minShiftedGain)
                    continue;

                // If this point in the code is reached, the flock is splittable.
                IsSplittable[0] = true;
                if (entropyCoefficient > 0)
                {
                    // Consider the entropy of the split.
                    double entropyGain = (totalCount * Math.Log(totalCount) - lteCount * Math.Log(lteCount) -
                                          gtCount * Math.Log(gtCount));

                    currentShiftedGain += entropyCoefficient * entropyGain;
                }

                // Is i the best threshold so far?
                if (currentShiftedGain > bestShiftedGain)
                {
                    bestGTCount = gtCount;
                    bestSumGTTargets = sumGTTargets;
                    bestSumGTWeights = sumGTWeights;
                    bestThreshold = i;
                    bestShiftedGain = currentShiftedGain;
                }
            }

            // set the appropriate place in the output vectors

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalFeatureIndices =
                features.GetRange(0, bestThreshold + 1).ToArray();

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalSplitRange = new[]
            {firstFlockFeature, lastFlockFeature};

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Feature = firstFlockFeature;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalSplit = true;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].LteOutput =
                learner.CalculateSplittedLeafOutput(totalCount - bestGTCount, sumTargets - bestSumGTTargets,
                    sumWeights - bestSumGTWeights);

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GTOutput =
                learner.CalculateSplittedLeafOutput(bestGTCount, bestSumGTTargets, bestSumGTWeights);

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].LteCount = totalCount - bestGTCount;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GTCount = bestGTCount;

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Gain = (bestShiftedGain - gainShift) * trust -
                                                                           usePenalty;

            double erfcArg = Math.Sqrt((bestShiftedGain - gainShift) * (totalCount - 1) / (2 * leafSplitCandidates.VarianceTargets * totalCount));

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GainPValue = ProbabilityFunctions.Erfc(erfcArg);
            if (leafSplitCandidates.FlockToBestFeature != null)
                leafSplitCandidates.FlockToBestFeature[flock] = firstFlockFeature;

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Flock = flock;
        }

        private class VirtualBin
        {
            public int FeatureIndex;
            public double SumTargets;
            public int Count;
            public int[] SubFeatures;
            public double Bias;

            public VirtualBin(double bias)
            {
                Bias = bias;
                SubFeatures = new int[0];
            }
            public double Gradient => (SumTargets / (Count + Bias));

        };

        public void FillSplitCandidatesCategoricalLowPopulation(LeastSquaresRegressionTreeLearner learner,
            LeastSquaresRegressionTreeLearner.LeafSplitCandidates leafSplitCandidates,
            int flock, int[] featureUseCount, double featureFirstUsePenalty, double featureReusePenalty,
            double minDocsInLeaf, bool hasWeights, double gainConfidenceInSquaredStandardDeviations, double entropyCoefficient)
        {
            int featureMin = learner.TrainData.FlockToFirstFeature(flock);
            int featureLim = featureMin + learner.TrainData.Flocks[flock].Count;

            //REVIEW: Should we not consider all the features if at least one feature is randomly chosen to not be active
            //when feature fraction is less than 1.0?
            List<int> features = learner.GetActiveFeatures(featureMin, featureLim).ToList();
            if (features.Count == 0 || !IsSplittable[0])
                return;
#if DEBUG
            foreach (var feature in features)
            {
                int subfeature = feature - featureMin;
                int min = GetMinBorder(subfeature);
                int max = GetMaxBorder(subfeature);
                Contracts.Assert(min == max);
            }
#endif
            //Use this feature to represent the flock.
            int firstFlockFeature = features[0];
            int lastFlockFeature = features[features.Count - 1];
            double minDocs = Math.Max(learner.MinDocsForCategoricalSplit,
                leafSplitCandidates.NumDocsInLeaf * learner.MinDocsPercentageForCategoricalSplit);

            List<VirtualBin> virtualBins = new List<VirtualBin>();
            List<int> lowPopulationFeatures = new List<int>();
            int lowPopulationCount = 0;
            double lowPopulationSumTargets = 0;
            foreach (var feature in features)
            {
                int subfeature = feature - featureMin;
                Contracts.Assert(0 <= subfeature && subfeature < Flock.Count);
                Contracts.Assert(subfeature <= feature);
                Contracts.Assert(learner.TrainData.FlockToFirstFeature(flock) == feature - subfeature);
                Contracts.Assert(featureUseCount[feature] >= 0);
                Contracts.Assert(Flock.BinCount(subfeature) == 2);
                Contracts.Assert(GetMaxBorder(subfeature) == GetMinBorder(subfeature));

                var binStats = GetBinStats(GetMinBorder(subfeature));

                if (binStats.Count >= minDocs)
                {
                    var vBin = new VirtualBin(learner.Bias);
                    vBin.FeatureIndex = feature;
                    vBin.Count = binStats.Count;
                    vBin.SumTargets = binStats.SumTargets;
                    virtualBins.Add(vBin);
                }
                else
                {
                    lowPopulationFeatures.Add(feature);
                    lowPopulationCount += binStats.Count;
                    lowPopulationSumTargets += binStats.SumTargets;
                }
            }

            if (lowPopulationFeatures.Count > 0)
            {
                virtualBins.Add(new VirtualBin(learner.Bias)
                {
                    Count = lowPopulationCount,
                    SumTargets = lowPopulationSumTargets,
                    FeatureIndex = lowPopulationFeatures[0],
                    SubFeatures = lowPopulationFeatures.GetRange(1, lowPopulationFeatures.Count - 1).ToArray()
                });
            }

            //REVIEW: What about the zero bin that contains missing values?
            virtualBins.Sort((a, b) => b.Gradient.CompareTo(a.Gradient));

            double usePenalty = (featureUseCount[firstFlockFeature] == 0)
                ? featureFirstUsePenalty
                : featureReusePenalty * Math.Log(featureUseCount[firstFlockFeature] + 1);

            double bestSumGTTargets = double.NaN;
            double bestSumGTWeights = double.NaN;
            double bestShiftedGain = double.NegativeInfinity;
            const double eps = 1e-10;
            int bestGTCount = -1;
            double sumGTTargets = 0.0;
            double sumGTWeights = eps;
            int gtCount = 0;
            int totalCount = leafSplitCandidates.NumDocsInLeaf;
            double sumTargets = leafSplitCandidates.SumTargets;
            double sumWeights = leafSplitCandidates.SumWeights + 2 * eps;
            double gainShift = learner.GetLeafSplitGain(totalCount, sumTargets, sumWeights);
            double trust = learner.TrainData.Flocks[flock].Trust(0);
            int bestThreshold = -1;
            double minDocsForThis = minDocsInLeaf / trust;
            // We get to this more explicit handling of the zero case since, under the influence of
            // numerical error, especially under single precision, the histogram computed values can
            // be wildly inaccurate even to the point where 0 unshifted gain may become a strong
            // criteria.
            double minShiftedGain = gainConfidenceInSquaredStandardDeviations <= 0
                ? 0.0
                : (gainConfidenceInSquaredStandardDeviations * leafSplitCandidates.VarianceTargets
                   * totalCount / (totalCount - 1) + gainShift);

            int docsInCurrentGroup = 0;
            IsSplittable[0] = false;
            int catFeatureCount = 0;
            for (int i = 0; i < virtualBins.Count && catFeatureCount < learner.MaxCategoricalSplitPointsPerNode; ++i)
            {
                var binStats = virtualBins[i];
                catFeatureCount += 1 + binStats.SubFeatures.Length;

                sumGTTargets += binStats.SumTargets;
                gtCount += binStats.Count;
                docsInCurrentGroup += binStats.Count;

                // Advance until GTCount is high enough.
                if (gtCount < minDocsForThis)
                    continue;

                int lteCount = totalCount - gtCount;

                // If LTECount is too small, we are finished.
                if (lteCount < minDocsForThis)
                    continue;

                // Calculate the shifted gain, including the LTE child.
                double currentShiftedGain = learner.GetLeafSplitGain(gtCount, sumGTTargets, sumGTWeights)
                                            +
                                            learner.GetLeafSplitGain(lteCount, sumTargets - sumGTTargets,
                                                sumWeights - sumGTWeights);

                // Test whether we are meeting the min shifted gain confidence criteria for this split.
                if (currentShiftedGain < minShiftedGain)
                    continue;

                // If this point in the code is reached, the flock is splittable.
                IsSplittable[0] = true;
                if (entropyCoefficient > 0)
                {
                    // Consider the entropy of the split.
                    double entropyGain = (totalCount * Math.Log(totalCount) - lteCount * Math.Log(lteCount) -
                                          gtCount * Math.Log(gtCount));

                    currentShiftedGain += entropyCoefficient * entropyGain;
                }

                // Is i the best threshold so far?
                if (currentShiftedGain > bestShiftedGain)
                {
                    bestGTCount = gtCount;
                    bestSumGTTargets = sumGTTargets;
                    bestSumGTWeights = sumGTWeights;
                    bestThreshold = i;
                    bestShiftedGain = currentShiftedGain;
                }
            }

            // set the appropriate place in the output vectors

            List<int> catFeatureIndices = new List<int>();
            for (int index = 0; index <= bestThreshold; index++)
            {
                catFeatureIndices.Add(virtualBins[index].FeatureIndex);
                catFeatureIndices.AddRange(virtualBins[index].SubFeatures);
            }

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalFeatureIndices =
                catFeatureIndices.ToArray();

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalSplitRange = new[]
            {firstFlockFeature, lastFlockFeature};

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Feature = firstFlockFeature;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalSplit = true;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].LteOutput =
                learner.CalculateSplittedLeafOutput(totalCount - bestGTCount, sumTargets - bestSumGTTargets,
                    sumWeights - bestSumGTWeights);

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GTOutput =
                learner.CalculateSplittedLeafOutput(bestGTCount, bestSumGTTargets, bestSumGTWeights);

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].LteCount = totalCount - bestGTCount;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GTCount = bestGTCount;

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Gain = (bestShiftedGain - gainShift) * trust -
                                                                           usePenalty;

            double erfcArg = Math.Sqrt((bestShiftedGain - gainShift) * (totalCount - 1) / (2 * leafSplitCandidates.VarianceTargets * totalCount));

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GainPValue = ProbabilityFunctions.Erfc(erfcArg);
            if (leafSplitCandidates.FlockToBestFeature != null)
                leafSplitCandidates.FlockToBestFeature[flock] = firstFlockFeature;

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Flock = flock;
        }

        public void FillSplitCandidatesCategoricalNeighborBundling(LeastSquaresRegressionTreeLearner learner,
            LeastSquaresRegressionTreeLearner.LeafSplitCandidates leafSplitCandidates,
            int flock, int[] featureUseCount, double featureFirstUsePenalty, double featureReusePenalty,
            double minDocsInLeaf,
            bool hasWeights, double gainConfidenceInSquaredStandardDeviations, double entropyCoefficient)
        {
            int featureMin = learner.TrainData.FlockToFirstFeature(flock);
            int featureLim = featureMin + learner.TrainData.Flocks[flock].Count;

            //REVIEW: Should we not consider all the features if at least one feature is randomly chosen to not be active
            //when feature fraction is less than 1.0?
            List<int> features = learner.GetActiveFeatures(featureMin, featureLim).ToList();
            if (features.Count == 0 || !IsSplittable[0])
                return;
#if DEBUG
            foreach (var feature in features)
            {
                int subfeature = feature - featureMin;
                int min = GetMinBorder(subfeature);
                int max = GetMaxBorder(subfeature);
                Contracts.Assert(min == max);
            }
#endif
            //Use this feature to represent the flock.
            int firstFlockFeature = features[0];
            int lastFlockFeature = features[features.Count - 1];
            double minDocs = Math.Max(learner.MinDocsForCategoricalSplit,
                leafSplitCandidates.NumDocsInLeaf * learner.MinDocsPercentageForCategoricalSplit);

            List<VirtualBin> virtualBins = new List<VirtualBin>();
            List<int> lowPopulationFeatures = new List<int>();
            int lowPopulationCount = 0;
            double lowPopulationSumTargets = 0;
            foreach (var feature in features)
            {
                int subfeature = feature - featureMin;
                Contracts.Assert(0 <= subfeature && subfeature < Flock.Count);
                Contracts.Assert(subfeature <= feature);
                Contracts.Assert(learner.TrainData.FlockToFirstFeature(flock) == feature - subfeature);
                Contracts.Assert(featureUseCount[feature] >= 0);
                Contracts.Assert(Flock.BinCount(subfeature) == 2);
                Contracts.Assert(GetMaxBorder(subfeature) == GetMinBorder(subfeature));

                var binStats = GetBinStats(GetMinBorder(subfeature));

                if (binStats.Count >= minDocs)
                {
                    var vBin = new VirtualBin(learner.Bias);
                    vBin.FeatureIndex = feature;
                    vBin.Count = binStats.Count;
                    vBin.SumTargets = binStats.SumTargets;
                    virtualBins.Add(vBin);

                    if (lowPopulationFeatures.Count > 0)
                    {
                        if (virtualBins.Count > 1)
                            vBin = virtualBins[virtualBins.Count - 2];

                        if (vBin.SubFeatures == null)
                            vBin.SubFeatures = lowPopulationFeatures.ToArray();
                        else
                        {
                            lowPopulationFeatures.AddRange(vBin.SubFeatures);
                            vBin.SubFeatures = lowPopulationFeatures.ToArray();
                        }

                        vBin.SumTargets += lowPopulationSumTargets;
                        vBin.Count += lowPopulationCount;

                        lowPopulationFeatures.Clear();
                        lowPopulationCount = 0;
                        lowPopulationSumTargets = 0;
                    }
                }
                else
                {
                    lowPopulationFeatures.Add(feature);
                    lowPopulationCount += binStats.Count;
                    lowPopulationSumTargets += binStats.SumTargets;

                    if (lowPopulationCount >= minDocs)
                    {
                        var vBin = new VirtualBin(learner.Bias);
                        vBin.FeatureIndex = lowPopulationFeatures[0];
                        vBin.SubFeatures =
                            lowPopulationFeatures.GetRange(1, lowPopulationFeatures.Count - 1).ToArray();

                        vBin.SumTargets = lowPopulationSumTargets;
                        vBin.Count = lowPopulationCount;
                        virtualBins.Add(vBin);
                        lowPopulationFeatures.Clear();
                        lowPopulationCount = 0;
                        lowPopulationSumTargets = 0;

                    }
                }
            }

            //REVIEW: What about the zero bin that contains missing values?
            virtualBins.Sort((a, b) => b.Gradient.CompareTo(a.Gradient));

            double usePenalty = (featureUseCount[firstFlockFeature] == 0)
                ? featureFirstUsePenalty
                : featureReusePenalty * Math.Log(featureUseCount[firstFlockFeature] + 1);

            double bestSumGTTargets = double.NaN;
            double bestSumGTWeights = double.NaN;
            double bestShiftedGain = double.NegativeInfinity;
            const double eps = 1e-10;
            int bestGTCount = -1;
            double sumGTTargets = 0.0;
            double sumGTWeights = eps;
            int gtCount = 0;
            int totalCount = leafSplitCandidates.NumDocsInLeaf;
            double sumTargets = leafSplitCandidates.SumTargets;
            double sumWeights = leafSplitCandidates.SumWeights + 2 * eps;
            double gainShift = learner.GetLeafSplitGain(totalCount, sumTargets, sumWeights);
            double trust = learner.TrainData.Flocks[flock].Trust(0);
            int bestThreshold = -1;
            double minDocsForThis = minDocsInLeaf / trust;
            // We get to this more explicit handling of the zero case since, under the influence of
            // numerical error, especially under single precision, the histogram computed values can
            // be wildly inaccurate even to the point where 0 unshifted gain may become a strong
            // criteria.
            double minShiftedGain = gainConfidenceInSquaredStandardDeviations <= 0
                ? 0.0
                : (gainConfidenceInSquaredStandardDeviations * leafSplitCandidates.VarianceTargets
                   * totalCount / (totalCount - 1) + gainShift);

            int docsInCurrentGroup = 0;
            IsSplittable[0] = false;
            int catFeatureCount = 0;
            for (int i = 0; i < virtualBins.Count && catFeatureCount < learner.MaxCategoricalSplitPointsPerNode; ++i)
            {
                var binStats = virtualBins[i];
                catFeatureCount += 1 + binStats.SubFeatures.Length;

                sumGTTargets += binStats.SumTargets;
                gtCount += binStats.Count;
                docsInCurrentGroup += binStats.Count;

                // Advance until GTCount is high enough.
                if (gtCount < minDocsForThis)
                    continue;

                int lteCount = totalCount - gtCount;

                // If LTECount is too small, we are finished.
                if (lteCount < minDocsForThis)
                    continue;

                // Calculate the shifted gain, including the LTE child.
                double currentShiftedGain = learner.GetLeafSplitGain(gtCount, sumGTTargets, sumGTWeights)
                                            +
                                            learner.GetLeafSplitGain(lteCount, sumTargets - sumGTTargets,
                                                sumWeights - sumGTWeights);

                // Test whether we are meeting the min shifted gain confidence criteria for this split.
                if (currentShiftedGain < minShiftedGain)
                    continue;

                // If this point in the code is reached, the flock is splittable.
                IsSplittable[0] = true;
                if (entropyCoefficient > 0)
                {
                    // Consider the entropy of the split.
                    double entropyGain = (totalCount * Math.Log(totalCount) - lteCount * Math.Log(lteCount) -
                                          gtCount * Math.Log(gtCount));

                    currentShiftedGain += entropyCoefficient * entropyGain;
                }

                // Is i the best threshold so far?
                if (currentShiftedGain > bestShiftedGain)
                {
                    bestGTCount = gtCount;
                    bestSumGTTargets = sumGTTargets;
                    bestSumGTWeights = sumGTWeights;
                    bestThreshold = i;
                    bestShiftedGain = currentShiftedGain;
                }
            }

            // set the appropriate place in the output vectors

            List<int> catFeatureIndices = new List<int>();
            for (int index = 0; index <= bestThreshold; index++)
            {
                catFeatureIndices.Add(virtualBins[index].FeatureIndex);
                catFeatureIndices.AddRange(virtualBins[index].SubFeatures);
            }

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalFeatureIndices =
                catFeatureIndices.ToArray();

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalSplitRange = new[]
            {firstFlockFeature, lastFlockFeature};

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Feature = firstFlockFeature;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].CategoricalSplit = true;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].LteOutput =
                learner.CalculateSplittedLeafOutput(totalCount - bestGTCount, sumTargets - bestSumGTTargets,
                    sumWeights - bestSumGTWeights);

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GTOutput =
                learner.CalculateSplittedLeafOutput(bestGTCount, bestSumGTTargets, bestSumGTWeights);

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].LteCount = totalCount - bestGTCount;
            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GTCount = bestGTCount;

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Gain = (bestShiftedGain - gainShift) * trust -
                                                                           usePenalty;

            double erfcArg = Math.Sqrt((bestShiftedGain - gainShift) * (totalCount - 1) / (2 * leafSplitCandidates.VarianceTargets * totalCount));

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].GainPValue = ProbabilityFunctions.Erfc(erfcArg);
            if (leafSplitCandidates.FlockToBestFeature != null)
                leafSplitCandidates.FlockToBestFeature[flock] = firstFlockFeature;

            leafSplitCandidates.FeatureSplitInfo[firstFlockFeature].Flock = flock;
        }
    }

    /// <summary>
    /// Type specific implementation of sufficient stats.
    /// </summary>
    /// <typeparam name="TSuffStats">The type of sufficient stats that we will be able to do
    /// "peer" operations against, like subtract. This will always be the derived class itself.</typeparam>
    public abstract class SufficientStatsBase<TSuffStats> : SufficientStatsBase
        where TSuffStats : SufficientStatsBase<TSuffStats>
    {
        protected SufficientStatsBase(int features)
            : base(features)
        {
        }

        protected sealed override void SubtractCore(SufficientStatsBase other)
        {
            Contracts.Assert(other is TSuffStats);
            Contracts.Assert(Flock == other.Flock);
            SubtractCore((TSuffStats)other);
        }

        /// <summary>
        /// Actual core implementation of subtraction. By the time this is called, the
        /// base class has ensured that the feature flock of this is the same as the
        /// feature flock in the other.
        /// </summary>
        /// <param name="other">The sufficient statistics we are subtracting</param>
        protected abstract void SubtractCore(TSuffStats other);
    }

    /// <summary>
    /// A feature flock is a collection of features, grouped together because storing the
    /// features and performing the key operations on them in a collection can be done
    /// more efficiently than if they were stored as separate features.
    ///
    /// Since this is a collection of features, feature specific quantities and methods
    /// will have a feature index parameter. Note that this index is always, for every
    /// flock, from 0 up to but not including <see cref="FeatureFlockBase.Count"/>. Now,
    /// in the larger context of a <see cref="Dataset"/> holding many flocks, the
    /// individual features might have some sort of "dataset-wide" index, but this is
    /// considered the business of the dataset, not the flocks themselves. See
    /// <see cref="Dataset.MapFeatureToFlockAndSubFeature"/> to see some details of this
    /// dataset-wide versus flock-wide feature index.
    /// </summary>
    public abstract class FeatureFlockBase
    {
        /// <summary>
        /// The number of features contained within this flock.
        /// </summary>
        public readonly int Count;

        /// <summary>
        /// The number of training examples represented by the features within this flock.
        /// This should be the same for all flocks within a dataset.
        /// </summary>
        public abstract int Examples { get; }

        /// <summary>
        /// Flock is a categorical feature.
        /// </summary>
        public bool Categorical;

        protected FeatureFlockBase(int count, bool categorical = false)
        {
            Contracts.Assert(0 < count);
            Count = count;
            Categorical = categorical;
        }

        /// <summary>
        /// An approximation of the size in bytes used by this structure. Used for estimating
        /// memory usage of the tree learner.
        /// </summary>
        public abstract long SizeInBytes();

        /// <summary>
        /// A reusable structure for tracking the sufficient statistics for tree learning
        /// of the features in this flock.
        /// </summary>
        /// <param name="hasWeights">Whether structures related to tracking
        /// example weights should be allocated</param>
        /// <returns>A sufficient statistics object</returns>
        public abstract SufficientStatsBase CreateSufficientStats(bool hasWeights);

        /// <summary>
        /// Returns a forward indexer for a single feature. This has a default implementation that
        /// relies on <see cref="GetFlockIndexer"/>, but base classes may find it beneficial from
        /// a performance perspective to provide their own implementation.
        /// </summary>
        public virtual IIntArrayForwardIndexer GetIndexer(int featureIndex)
        {
            Contracts.Assert(0 <= featureIndex && featureIndex < Count);
            return new GenericIntArrayForwardIndexer(GetFlockIndexer(), featureIndex);
        }

        /// <summary>
        /// Returns a forward indexer for all features within the flock.
        /// </summary>
        public abstract FlockForwardIndexerBase GetFlockIndexer();

        public abstract FeatureFlockBase[] Split(int[][] assignment);

        /// <summary>
        /// Given a feature index, return the number of distinct bins there are for that feature.
        /// This will be the length of <see cref="BinUpperBounds"/> for this feature. This is also
        /// the upper exclusive limit on the binned value seen for this feature.
        /// </summary>
        /// <param name="featureIndex">The index of the feature within the flock</param>
        /// <returns>The number of distinct bins for that feature within the flock</returns>
        public abstract int BinCount(int featureIndex);

        /// <summary>
        /// The multiplier on the gain for any particular feature. This can be used to make
        /// features appear more or less attractive. The default value should be considered
        /// to be 1.
        /// </summary>
        public abstract double Trust(int featureIndex);

        /// <summary>
        /// An array of increasing values, forming the boundaries of all the binned values.
        /// </summary>
        /// <param name="featureIndex"></param>
        /// <returns>The bin upper bounds for a feature. This array will have the same
        /// length as <see cref="BinCount"/>.</returns>
        public abstract double[] BinUpperBounds(int featureIndex);

        /// <summary>
        /// If you need to implement <see cref="GetIndexer"/> you can use
        /// <see cref="GenericIntArrayForwardIndexer"/>. This will be slower than a
        /// specialized implementation but is at least a useful shim.
        /// </summary>
        private sealed class GenericIntArrayForwardIndexer : IIntArrayForwardIndexer
        {
            private readonly FlockForwardIndexerBase _flockIndexer;
            private readonly int _feature;

            public GenericIntArrayForwardIndexer(FlockForwardIndexerBase flockIndexer, int feature)
            {
                Contracts.AssertValue(flockIndexer);
                Contracts.Assert(0 <= feature && feature < flockIndexer.Flock.Count);
                _flockIndexer = flockIndexer;
                _feature = feature;
            }

            public int this[int index]
            {
                get { return _flockIndexer[_feature, index]; }
            }
        }

        /// <summary>
        /// Interface for objects that can index into a flock, but only with a nondecreasing sequence of row
        /// indices from access to access. It is fine for feature indices to appear in any order.
        /// A feature group analogy to <see cref="IIntArrayForwardIndexer"/> but for feature flocks instead of
        /// <see cref="IntArray"/> instances.
        /// </summary>
        public abstract class FlockForwardIndexerBase
        {
            /// <summary>
            /// The flock over which this feature flock was built.
            /// </summary>
            public abstract FeatureFlockBase Flock { get; }

            /// <summary>
            /// Gets the element at the given position.
            /// </summary>
            /// <param name="featureIndex">The index of the feature within the flock</param>
            /// <param name="rowIndex">Index of the row to get, should be non-decreasing from any previous
            /// access on this indexer</param>
            /// <returns>The value at the index</returns>
            public abstract int this[int featureIndex, int rowIndex] { get; }
        }
    }

    /// <summary>
    /// A base class for a feature flock that wraps a single <see cref="IntArray"/> that contains multiple
    /// feature values using a concatentation of the non-zero ranges of each feature, and also in some way
    /// that doing a <see cref="IntArray.Sumup"/> will accumulate sufficient statistics correctly for all
    /// except the first (zero) bin.
    /// </summary>
    /// <typeparam name="TIntArray">The type of <see>IntArray</see> this implementation wraps</typeparam>
    internal abstract class SinglePartitionedIntArrayFlockBase<TIntArray> : FeatureFlockBase
        where TIntArray : IntArray
    {
        public readonly TIntArray Bins;
        protected readonly int[] HotFeatureStarts;
        protected readonly double[][] AllBinUpperBounds;

        public override int Examples => Bins.Length;

        /// <summary>
        /// Constructor for the <see cref="SinglePartitionedIntArrayFlockBase{TIntArray}"/>.
        /// </summary>
        /// <param name="bins">The binned version of the features, stored collectively in one
        /// <see cref="IntArray"/>, where 0 indicates that all features are in the "cold" bin
        /// of 0, and a non-zero value indicates one of the features is "hot," where which
        /// feature is hot and what value it has is indicated by <paramref name="hotFeatureStarts"/>.
        /// The zero value is "shared" among all features, effectively, and the non-zero values
        /// are the result of a shifted concatenation of the range of the non-zero values, for
        /// each feature incorporated in the flock. See the example for more concrete information.
        /// </param>
        /// <param name="hotFeatureStarts">The ranges of values of <paramref name="bins"/>
        /// where features start and stop. This is a non-decreasing array of integers. For
        /// feature <c>f</c>, the elements at <c>f</c> and <c>f+1</c> indicate the minimum
        /// and limit of values in <paramref name="bins"/> that indicate that the corresponding
        /// feature is "hot" starting at a value of 1.</param>
        /// <param name="binUpperBounds">The bin upper bounds structure</param>
        /// <param name="categorical"></param>
        /// <example>
        /// Imagine we have a six row dataset, with two features, which if stored separately in,
        /// say, a <see cref="SingletonFeatureFlock"/>, would have bin values as follows.
        ///
        /// <c>f0 = { 0, 1, 0, 0, 2, 0}</c>
        /// <c>f1 = { 0, 0, 1, 0, 0, 1}</c>
        ///
        /// These two are a candidate for a <see cref="OneHotFeatureFlock"/>, because they never both
        /// have a non-zero bin value for any row. Then, in order to represent this in this feature,
        /// we would pass in this value for the <paramref name="bins"/>:
        /// <c><paramref name="bins"/> = { 0, 1, 3, 0, 2, 3 }</c>
        /// and this value for <paramref name="hotFeatureStarts"/>:
        /// <c><paramref name="hotFeatureStarts"/> = { 1, 3, 4 }</c>
        /// Note that the range of <paramref name="bins"/> is, aside from the zero, the concatenation
        /// of the non-zero range of all constituent input features, and where the reconstruction of
        /// what feature is which can be reconstructed from <paramref name="hotFeatureStarts"/>, which
        /// for each feature specifies the range in <paramref name="bins"/> corresponding to the
        /// "logical" bin value for that feature starting from 1.
        ///
        /// Note that it would also have been legal for <paramref name="hotFeatureStarts"/> to be
        /// larger than the actual observed range, e.g., it could have been:
        /// <c><paramref name="hotFeatureStarts"/> = { 1, 5, 8}</c>
        /// or something. This could happen if binning happened over a different dataset from the data
        /// being represented right now, for example, but this is a more complex case.
        ///
        /// The <paramref name="binUpperBounds"/> would contain the upper bounds for both of these features,
        /// which would be arrays large enough so that the maximum value of the logical bin for each feature
        /// in the flock could index it. (So in this example, the first bin upper bound would be at least
        /// length 3, and the second at least length 2.)
        ///
        /// The <paramref name="categorical"/> indicates if the flock is a categorical feature.
        /// </example>
        protected SinglePartitionedIntArrayFlockBase(TIntArray bins, int[] hotFeatureStarts, double[][] binUpperBounds, bool categorical = false)
            : base(Utils.Size(hotFeatureStarts) - 1, categorical)
        {
            Contracts.AssertValue(bins);
            Contracts.AssertValue(binUpperBounds);
            Contracts.Assert(Utils.Size(hotFeatureStarts) == binUpperBounds.Length + 1); // One more than number of features.
            Contracts.Assert(hotFeatureStarts[0] == 1);
            Contracts.Assert(Utils.IsSorted(hotFeatureStarts));
            Contracts.Assert(bins.Max() < hotFeatureStarts[hotFeatureStarts.Length - 1]);

            Bins = bins;
            HotFeatureStarts = hotFeatureStarts;
            AllBinUpperBounds = binUpperBounds;
            // Bin upper bounds must have some value.
            Contracts.Assert(AllBinUpperBounds.All(x => Utils.Size(x) >= 1));
            // The hot feature start ranges and the bin lengths must be consistent.
            Contracts.Assert(AllBinUpperBounds.Select((b, f) => HotFeatureStarts[f + 1] - HotFeatureStarts[f] + 1 == b.Length).All(i => i));
        }

        public sealed override double[] BinUpperBounds(int featureIndex)
        {
            Contracts.Assert(0 <= featureIndex && featureIndex < Count);
            return AllBinUpperBounds[featureIndex];
        }

        public sealed override double Trust(int featureIndex)
        {
            Contracts.Assert(0 <= featureIndex && featureIndex < Count);
            return 1;
        }

        public sealed override int BinCount(int featureIndex)
        {
            Contracts.Assert(0 <= featureIndex && featureIndex < Count);
            return AllBinUpperBounds[featureIndex].Length;
        }

        public override long SizeInBytes()
        {
            return Bins.SizeInBytes()
                   + sizeof(double) * (AllBinUpperBounds.Length - 1 + HotFeatureStarts[HotFeatureStarts.Length - 1])
                   + sizeof(int) * HotFeatureStarts.Length;
        }

        public override SufficientStatsBase CreateSufficientStats(bool hasWeights)
        {
            return new SufficientStats(this, hasWeights);
        }

        /// <summary>
        /// Stores the sufficient statistics for all features within this flock using a single
        /// histogram, where the range of what accumulated sufficient statistics are relevant
        /// to what feature can be inferred by <see cref="HotFeatureStarts"/>.
        /// </summary>
        private sealed class SufficientStats : SufficientStatsBase<SufficientStats>
        {
            private readonly SinglePartitionedIntArrayFlockBase<TIntArray> _flock;
            /// <summary>
            /// Stores the sufficient statistics for all features within this flock using a single
            /// histogram, where the range of what accumulated sufficient statistics correspond to
            /// what actual logical feature depends on <see cref="HotFeatureStarts"/>.
            /// </summary>
            public readonly FeatureHistogram Hist;

            public override FeatureFlockBase Flock => _flock;

            public SufficientStats(SinglePartitionedIntArrayFlockBase<TIntArray> flock, bool hasWeights)
                : base(flock.Count)
            {
                Contracts.AssertValue(flock);
                _flock = flock;
                Hist = new FeatureHistogram(_flock.Bins, _flock.HotFeatureStarts[_flock.Count], hasWeights);
            }

            protected override void SubtractCore(SufficientStats other)
            {
                Hist.Subtract(other.Hist);
            }

            protected override void SumupCore(int featureOffset, bool[] active,
                int numDocsInLeaf, double sumTargets, double sumWeights,
                double[] outputs, double[] weights, int[] docIndices)
            {
                Contracts.AssertValueOrNull(active);
                Contracts.Assert(active == null || (0 <= featureOffset && featureOffset <= Utils.Size(active) - Flock.Count));
                // The underlying histogram needs to have a sumup iff any features in the flock are active.
                if (active == null)
                {
                    Hist.SumupWeighted(numDocsInLeaf, sumTargets, sumWeights, outputs, weights, docIndices);
                    return;
                }

                for (int i = 0; i < _flock.Count; ++i)
                {
                    if (active[i + featureOffset])
                    {
                        Hist.SumupWeighted(numDocsInLeaf, sumTargets, sumWeights, outputs, weights, docIndices);
                        return;
                    }
                }
            }

            public override long SizeInBytes()
            {
                return FeatureHistogram.EstimateMemoryUsedForFeatureHistogram(Hist.NumFeatureValues,
                    Hist.SumWeightsByBin != null) + sizeof(int) + 2 * sizeof(double);
            }

            protected override int GetMaxBorder(int featureIndex)
            {
                return _flock.HotFeatureStarts[featureIndex + 1] - 1;
            }

            protected override int GetMinBorder(int featureIndex)
            {
                return _flock.HotFeatureStarts[featureIndex];
            }

            protected override PerBinStats GetBinStats(int featureIndex)
            {
                if (Hist.SumWeightsByBin != null)
                    return new PerBinStats(Hist.SumTargetsByBin[featureIndex], Hist.SumWeightsByBin[featureIndex], Hist.CountByBin[featureIndex]);
                else
                    return new PerBinStats(Hist.SumTargetsByBin[featureIndex], 0, Hist.CountByBin[featureIndex]);
            }

            protected override double GetBinGradient(int featureIndex, double bias)
            {
                if (Hist.SumWeightsByBin != null)
                    return Hist.SumTargetsByBin[featureIndex] / (Hist.SumWeightsByBin[featureIndex] + bias);
                else
                    return Hist.SumTargetsByBin[featureIndex] / (Hist.CountByBin[featureIndex] + bias);
            }
        }
    }
}
