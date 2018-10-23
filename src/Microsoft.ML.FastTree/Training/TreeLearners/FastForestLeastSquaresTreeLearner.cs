// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Trainers;
using System;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public class RandomForestLeastSquaresTreeLearner : LeastSquaresRegressionTreeLearner
    {
        private int _quantileSampleCount;
        private bool _quantileEnabled;

        public RandomForestLeastSquaresTreeLearner(Dataset trainData, int numLeaves, int minDocsInLeaf, Double entropyCoefficient, Double featureFirstUsePenalty,
            Double featureReusePenalty, Double softmaxTemperature, int histogramPoolSize, int randomSeed, Double splitFraction, bool allowEmptyTrees,
            Double gainConfidenceLevel, int maxCategoricalGroupsPerNode, int maxCategoricalSplitPointsPerNode, bool quantileEnabled, int quantileSampleCount, IParallelTraining parallelTraining,
            double minDocsPercentageForCategoricalSplit, Bundle bundling, int minDocsForCategoricalSplit, double bias)
            : base(trainData, numLeaves, minDocsInLeaf, entropyCoefficient, featureFirstUsePenalty, featureReusePenalty, softmaxTemperature, histogramPoolSize,
                randomSeed, splitFraction, false, allowEmptyTrees, gainConfidenceLevel, maxCategoricalGroupsPerNode, maxCategoricalSplitPointsPerNode, - 1, parallelTraining,
                minDocsPercentageForCategoricalSplit, bundling, minDocsForCategoricalSplit, bias)
        {
            _quantileSampleCount = quantileSampleCount;
            _quantileEnabled = quantileEnabled;
        }

        protected override RegressionTree NewTree()
        {
            return new QuantileRegressionTree(NumLeaves);
        }

        public RegressionTree FitTargets(IChannel ch, bool[] activeFeatures, Double[] weightedtargets, Double[] targets, Double[] weights)
        {
            var tree = (QuantileRegressionTree)FitTargets(ch, activeFeatures, weightedtargets);
            if (tree != null && _quantileEnabled)
            {
                Double[] distributionWeights = null;
                tree.SetLabelsDistribution(Partitioning.GetDistribution(
                    targets, weights, _quantileSampleCount, Rand, tree.NumLeaves, out distributionWeights), distributionWeights);
            }

            return tree;
        }

        protected override void FindAndSetBestFeatureForLeaf(LeafSplitCandidates leafSplitCandidates)
        {
            if (SoftmaxTemperature != 0 || SplitFraction == 1.0)
            {
                base.FindAndSetBestFeatureForLeaf(leafSplitCandidates);
                return;
            }

            // REVIEW: Stupid, but changing actually changes all
            // FastForeset baselines. Improve later.
            var infos = leafSplitCandidates.FeatureSplitInfo;
            int bestFeature = 0;
            double max = infos[0].Gain;
            for (int i = 1; i < infos.Length; ++i)
            {
                if (infos[i].Gain > max && Rand.NextDouble() < SplitFraction || Double.IsNegativeInfinity(max))
                    max = infos[bestFeature = i].Gain;
            }
            SetBestFeatureForLeaf(leafSplitCandidates, bestFeature);
        }
    }
}
