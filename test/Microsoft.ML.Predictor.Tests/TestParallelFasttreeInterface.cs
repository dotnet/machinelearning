// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

[assembly: LoadableClass(typeof(FastTreeParallelInterfaceChecker),
    null, typeof(Microsoft.ML.Runtime.FastTree.SignatureParallelTrainer), "FastTreeParallelInterfaceChecker")]

namespace Microsoft.ML.Runtime.RunTests
{
    using SplitInfo = Microsoft.ML.Runtime.FastTree.Internal.LeastSquaresRegressionTreeLearner.SplitInfo;
    using LeafSplitCandidates = Microsoft.ML.Runtime.FastTree.Internal.LeastSquaresRegressionTreeLearner.LeafSplitCandidates;

    public sealed class FastTreeParallelInterfaceChecker : Microsoft.ML.Runtime.FastTree.IParallelTraining
    {
        private bool _isInitEnv = false;
        private bool _isInitTreeLearner = false;
        private bool _isInitIteration = false;
        private bool _isCache = false;
        public void CacheHistogram(bool isSmallerLeaf, int featureIdx, int subfeature, SufficientStatsBase sufficientStatsBase, bool HasWeights)
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.True(_isInitIteration);
            Assert.NotNull(sufficientStatsBase);
            Assert.False(!_isCache);
            _isCache = true;
            return;
        }

        public bool IsNeedFindLocalBestSplit()
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.True(_isInitIteration);
            return true;
        }

        public void FindGlobalBestSplit(LeafSplitCandidates smallerChildSplitCandidates,
            LeafSplitCandidates largerChildSplitCandidates,
            Microsoft.ML.Runtime.FastTree.FindBestThresholdFromRawArrayFun findFunction,
            SplitInfo[] bestSplits)
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.True(_isInitIteration);
            Assert.True(_isCache);
            _isCache = false;
            Assert.NotNull(smallerChildSplitCandidates);
            Assert.NotNull(bestSplits);
            return;
        }

        public void GetGlobalDataCountInLeaf(int leafIdx, ref int cnt)
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.True(_isInitIteration);
            Assert.True(leafIdx >= 0);
            return;
        }

        public bool[] GetLocalBinConstructionFeatures(int numFeatures)
        {
            Assert.True(_isInitEnv);
            Assert.True(numFeatures >= 0);
            return Utils.CreateArray<bool>(numFeatures, true);
        }

        public double[] GlobalMean(Dataset dataset, RegressionTree tree, DocumentPartitioning partitioning, double[] weights, bool filterZeroLambdas)
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.NotNull(dataset);
            Assert.NotNull(tree);
            Assert.NotNull(partitioning);
            double[] means = new double[tree.NumLeaves];
            for (int l = 0; l < tree.NumLeaves; ++l)
            {
                means[l] = partitioning.Mean(weights, dataset.SampleWeights, l, filterZeroLambdas);
            }
            return means;
        }

        public void PerformGlobalSplit(int leaf, int lteChild, int gtChild, SplitInfo splitInfo)
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.True(_isInitIteration);
            return;
        }

        public void InitIteration(ref bool[] activeFeatures)
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.False(_isInitIteration);
            _isInitIteration = true;
            Assert.NotNull(activeFeatures);
            return;
        }

        public void InitEnvironment()
        {
            Assert.False(_isInitEnv);
            _isInitEnv = true;
            return;
        }

        public void InitTreeLearner(Dataset trainData, int maxNumLeaves, int maxCatSplitPoints, ref int minDocInLeaf)
        {
            Assert.True(_isInitEnv);
            Assert.False(_isInitTreeLearner);
            _isInitTreeLearner = true;
            Assert.NotNull(trainData);
            return;
        }

        public void SyncGlobalBoundary(int numFeatures, int maxBin, Double[][] binUpperBounds)
        {
            Assert.True(_isInitEnv);
            Assert.NotNull(binUpperBounds);
            return;
        }

        public void FinalizeEnvironment()
        {
            Assert.True(_isInitEnv);
            Assert.False(_isInitTreeLearner);
            Assert.False(_isInitIteration);
            _isInitEnv = false;
            return;
        }

        public void FinalizeTreeLearner()
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.False(_isInitIteration);
            _isInitTreeLearner = false;
            return;
        }

        public void FinalizeIteration()
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.True(_isInitIteration);
            _isInitIteration = false;
            return;
        }

        public bool IsSkipNonSplittableHistogram()
        {
            Assert.True(_isInitEnv);
            Assert.True(_isInitTreeLearner);
            Assert.True(_isInitIteration);
            return true;
        }
    }

    public class TestParallelFasttreeInterface : BaseTestBaseline
    {
        public TestParallelFasttreeInterface(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        [TestCategory("ParallelFasttree")]
        public void CheckFastTreeParallelInterface()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var outRoot = @"..\Common\CheckInterface";
            var modelOutPath = DeleteOutputPath(outRoot, "codegen-model.zip");
            var csOutPath = DeleteOutputPath(outRoot, "codegen-out.cs");

            var trainArgs = string.Format(
                "train data={{{0}}} loader=Text{{col=Label:0 col=F!1:1-5 col=F2:6-9}} xf=Concat{{col=Features:F!1,F2}}  tr=FastTreeBinaryClassification{{lr=0.1 nl=12 mil=10 iter=1 parag=checker}} out={{{1}}}",
                dataPath, modelOutPath);
            MainForTest(trainArgs);
        }
    }
}
