// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(typeof(Microsoft.ML.Runtime.FastTree.SingleTrainer),
    null, typeof(Microsoft.ML.Runtime.FastTree.SignatureParallelTrainer), "single")]

[assembly: EntryPointModule(typeof(SingleTrainerFactory))]

namespace Microsoft.ML.Runtime.FastTree
{
    using Microsoft.ML.Runtime.FastTree.Internal;
    using SplitInfo = Internal.LeastSquaresRegressionTreeLearner.SplitInfo;
    using LeafSplitCandidates = Internal.LeastSquaresRegressionTreeLearner.LeafSplitCandidates;

    public sealed class SingleTrainer : IParallelTraining
    {
        public void CacheHistogram(bool isSmallerLeaf, int featureIdx, int subfeature, SufficientStatsBase sufficientStatsBase, bool hasWeights)
        {
            return;
        }

        public bool IsNeedFindLocalBestSplit()
        {
            return true;
        }

        public void FindGlobalBestSplit(LeafSplitCandidates smallerChildSplitCandidates,
            LeafSplitCandidates largerChildSplitCandidates,
            FindBestThresholdFromRawArrayFun findFunction,
            SplitInfo[] bestSplits)
        {
            return;
        }

        public void GetGlobalDataCountInLeaf(int leafIdx, ref int cnt)
        {
            return;
        }

        public bool[] GetLocalBinConstructionFeatures(int numFeatures)
        {
            return Utils.CreateArray<bool>(numFeatures, true);
        }

        public double[] GlobalMean(Dataset dataset, RegressionTree tree, DocumentPartitioning partitioning, double[] weights, bool filterZeroLambdas)
        {
            double[] means = new double[tree.NumLeaves];
            for (int l = 0; l < tree.NumLeaves; ++l)
            {
                means[l] = partitioning.Mean(weights, dataset.SampleWeights, l, filterZeroLambdas);
            }
            return means;
        }

        public void PerformGlobalSplit(int leaf, int lteChild, int gtChild, SplitInfo splitInfo)
        {
            return;
        }

        public void InitIteration(ref bool[] activeFeatures)
        {
            return;
        }

        public void InitEnvironment()
        {
            return;
        }

        public void InitTreeLearner(Dataset trainData, int maxNumLeaves, int maxCatSplitPoints, ref int minDocInLeaf)
        {
            return;
        }

        public void SyncGlobalBoundary(int numFeatures, int maxBin, Double[][] binUpperBounds)
        {
            return;
        }

        public void FinalizeEnvironment()
        {
            return;
        }

        public void FinalizeTreeLearner()
        {
            return;
        }

        public void FinalizeIteration()
        {
            return;
        }

        public bool IsSkipNonSplittableHistogram()
        {
            return true;
        }
    }

    [TlcModule.Component(Name = "Single", Desc = "Single node machine learning process.")]
    public sealed class SingleTrainerFactory : ISupportParallelTraining
    {
        public IParallelTraining CreateComponent(IHostEnvironment env) => new SingleTrainer();
    }
}
