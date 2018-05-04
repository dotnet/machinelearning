// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree.Internal;

namespace Microsoft.ML.Runtime.FastTree
{
    using SplitInfo = Internal.LeastSquaresRegressionTreeLearner.SplitInfo;
    using LeafSplitCandidates = Internal.LeastSquaresRegressionTreeLearner.LeafSplitCandidates;

#if USE_SINGLE_PRECISION
    using FloatType = System.Single;
#else
    using FloatType = System.Double;
#endif

    /// <summary>
    /// Signature of Parallel trainer.
    /// </summary>
    public delegate void SignatureParallelTrainer();

    /// <summary>
    /// delegate function. This function is implemented in TLC, and called by TLC++. It will find best threshold
    /// from raw histogram data (countByBin, sumTargetsByBin, sumWeightsByBin, numDocsInLeaf, sumTargets, sumWeights)
    /// </summary>
    public delegate void FindBestThresholdFromRawArrayFun(LeafSplitCandidates leafSplitCandidates, int feature, int flock, int subfeature,
        int[] countByBin, FloatType[] sumTargetsByBin, FloatType[] sumWeightsByBin,
        int numDocsInLeaf, double sumTargets, double sumWeights, double varianceTargets, out SplitInfo bestSplit);

    /// <summary>
    /// Interface used for parallel training.
    /// Mainly contains three parts:
    /// 1. interactive with IO: <see href="GetLocalBinConstructionFeatures" />, <see href="SyncGlobalBoundary" />.
    ///    Data will be partitioned by rows in Data parallel and Voting Parallel.
    ///    To speed up the find bin process, it let different workers to find bins for different features.
    ///    Then perform global sync up.
    ///    In Feature parallel, every machines holds all data, so this is unneeded.
    /// 2. interactive with TreeLearner: <see href="InitIteration" />, <see href="CacheHistogram" />, <see href="IsNeedFindLocalBestSplit" />, 
    ///        <see href="IsSkipNonSplittableHistogram" />, <see href="FindGlobalBestSplit" />, <see href="GetGlobalDataCountInLeaf" />, <see href="PerformGlobalSplit" />.
    ///    A full process is:
    ///        Use <see href="InitIteration" /> to alter local active features.
    ///        Use <see href="GetGlobalDataCountInLeaf" /> to check smaller leaf and larger leaf.
    ///        Use <see href="CacheHistogram" />, <see href="IsNeedFindLocalBestSplit" /> and <see href="IsSkipNonSplittableHistogram" /> to interactive with Feature histograms.
    ///        Use <see href="FindGlobalBestSplit" /> to sync up global best split
    ///        Use <see href="PerformGlobalSplit" /> to record global num_data in leaves.
    /// 3. interactive with Application : <see href="GlobalMean" />.
    ///    Output of leaves is calculated by newton step ( - sum(first_order_gradients) / sum(second_order_gradients)).
    ///    If data is partitioned by row, it needs to a sync up for these sum result.
    ///    So It needs to call this to get the real output of leaves.
    /// </summary>
    public interface IParallelTraining
    {
        /// <summary>
        /// Initialize the network connection.
        /// </summary>
        void InitEnvironment();

        /// <summary>
        /// Finalize the network.
        /// </summary>
        void FinalizeEnvironment();

        /// <summary>
        /// Initialize once while construct tree learner.
        /// </summary>
        void InitTreeLearner(Dataset trainData, int maxNumLeaves, int maxCatSplitPoints, ref int minDocInLeaf);

        /// <summary>
        /// Finalize while tree learner is freed.
        /// </summary>
        void FinalizeTreeLearner();

        /// <summary>
        /// Initialize every time before training a tree.
        /// will alter activeFeatures in Feature parallel. 
        /// Because it only need to find threshold for part of features in feature parallel.
        /// </summary>
        void InitIteration(ref bool[] activeFeatures);

        /// <summary>
        /// Finalize after trained one tree.
        /// </summary>
        void FinalizeIteration();

        /// <summary>
        /// Cache Histogram, it will be used for global aggregate.
        /// Only used in Data parallel and Voting Parallel
        /// </summary>
        void CacheHistogram(bool isSmallerLeaf, int featureIdx, int subfeature, SufficientStatsBase sufficientStatsBase, bool hasWeights);

        /// <summary>
        /// Only return False in Data parallel.
        /// Data parallel find best threshold after merged global histograms.
        /// </summary>
        bool IsNeedFindLocalBestSplit();

        /// <summary>
        /// True if need to skip non-splittable histogram. 
        /// Only will return False in Voting parallel. 
        /// That is because local doesn't have global histograms in Voting parallel,
        /// So the information about NonSplittable is not correct, and we cannot skip it. 
        /// </summary>
        bool IsSkipNonSplittableHistogram();

        /// <summary>
        /// Find best split among machines.
        /// will save result in bestSplits.
        /// </summary>
        void FindGlobalBestSplit(LeafSplitCandidates smallerChildSplitCandidates,
           LeafSplitCandidates largerChildSplitCandidates,
           FindBestThresholdFromRawArrayFun findFunction,
           SplitInfo[] bestSplits);

        /// <summary>
        /// Get global num_data on specific leaf.
        /// </summary>
        void GetGlobalDataCountInLeaf(int leafIdx, ref int cnt);

        /// <summary>
        /// Used to record the global num_data on leaves.
        /// </summary>
        void PerformGlobalSplit(int leaf, int lteChild, int gtChild, SplitInfo splitInfo);

        /// <summary>
        /// Get Global mean on different machines for data partitioning in tree.
        /// Used for calculating leaf output value.
        /// will return a array this is the mean output of all leaves.
        /// </summary>
        double[] GlobalMean(Dataset dataset, RegressionTree tree, DocumentPartitioning partitioning, double[] weights, bool filterZeroLambdas);

        /// <summary>
        /// Get indices of features that should be find bin in local.
        /// After construct local boundary, should call <see href="SyncGlobalBoundary" /> 
        /// to get boundaries for all features.
        /// </summary>
        bool[] GetLocalBinConstructionFeatures(int numFeatures);

        /// <summary>
        /// Sync Global feature bucket.
        /// used in Data parallel and Voting parallel.
        /// Data are partitioned by row. To speed up the Global find bin process, 
        /// we can let different workers construct Bin Boundary for different features, 
        /// then perform a global sync up.
        /// </summary>
        void SyncGlobalBoundary(int numFeatures, int maxBin, Double[][] binUpperBounds);
    }

    [TlcModule.ComponentKind("ParallelTraining")]
    public interface ISupportParallelTraining : IComponentFactory<IParallelTraining>
    {
    }
}
