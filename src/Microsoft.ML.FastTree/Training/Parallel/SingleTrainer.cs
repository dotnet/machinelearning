// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;

[assembly: LoadableClass(typeof(SingleTrainer),
    null, typeof(SignatureParallelTrainer), "single")]

[assembly: EntryPointModule(typeof(SingleTrainerFactory))]

namespace Microsoft.ML.Trainers.FastTree
{
    using LeafSplitCandidates = LeastSquaresRegressionTreeLearner.LeafSplitCandidates;
    using SplitInfo = LeastSquaresRegressionTreeLearner.SplitInfo;

    internal sealed class SingleTrainer : IParallelTraining
    {
        void IParallelTraining.CacheHistogram(bool isSmallerLeaf, int featureIdx, int subfeature, SufficientStatsBase sufficientStatsBase, bool hasWeights)
        {
            return;
        }

        bool IParallelTraining.IsNeedFindLocalBestSplit()
        {
            return true;
        }

        void IParallelTraining.FindGlobalBestSplit(LeafSplitCandidates smallerChildSplitCandidates,
            LeafSplitCandidates largerChildSplitCandidates,
            FindBestThresholdFromRawArrayFun findFunction,
            SplitInfo[] bestSplits)
        {
            return;
        }

        void IParallelTraining.GetGlobalDataCountInLeaf(int leafIdx, ref int cnt)
        {
            return;
        }

        bool[] IParallelTraining.GetLocalBinConstructionFeatures(int numFeatures)
        {
            return Utils.CreateArray<bool>(numFeatures, true);
        }

        double[] IParallelTraining.GlobalMean(Dataset dataset, InternalRegressionTree tree, DocumentPartitioning partitioning, double[] weights, bool filterZeroLambdas)
        {
            double[] means = new double[tree.NumLeaves];
            for (int l = 0; l < tree.NumLeaves; ++l)
            {
                means[l] = partitioning.Mean(weights, dataset.SampleWeights, l, filterZeroLambdas);
            }
            return means;
        }

        void IParallelTraining.PerformGlobalSplit(int leaf, int lteChild, int gtChild, SplitInfo splitInfo)
        {
            return;
        }

        void IParallelTraining.InitIteration(ref bool[] activeFeatures)
        {
            return;
        }

        void IParallelTraining.InitEnvironment()
        {
            return;
        }

        void IParallelTraining.InitTreeLearner(Dataset trainData, int maxNumLeaves, int maxCatSplitPoints, ref int minDocInLeaf)
        {
            return;
        }

        void IParallelTraining.SyncGlobalBoundary(int numFeatures, int maxBin, Double[][] binUpperBounds)
        {
            return;
        }

        void IParallelTraining.FinalizeEnvironment()
        {
            return;
        }

        void IParallelTraining.FinalizeTreeLearner()
        {
            return;
        }

        void IParallelTraining.FinalizeIteration()
        {
            return;
        }

        bool IParallelTraining.IsSkipNonSplittableHistogram()
        {
            return true;
        }
    }

    [TlcModule.Component(Name = "Single", Desc = "Single node machine learning process.")]
    internal sealed class SingleTrainerFactory : ISupportParallelTraining
    {
        public IParallelTraining CreateComponent(IHostEnvironment env) => new SingleTrainer();
    }
}
