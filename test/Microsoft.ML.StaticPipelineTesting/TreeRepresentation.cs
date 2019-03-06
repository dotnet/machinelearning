using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.StaticPipelineTesting
{
    public sealed class TreeRepresentation : BaseTestClassWithConsole
    {
        public TreeRepresentation(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void FastTreeRegressionRepresentation()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var catalog = new RegressionCatalog(env);

            var reader = TextLoaderStatic.CreateLoader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            var opts = new FastTreeRegressionTrainer.Options()
            {
                NumberOfTrees = 10,
                NumberOfLeaves = 5,
                NumberOfThreads = 1
            };

            FastTreeRegressionModelParameters pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: catalog.Trainers.FastTree(r.label, r.features, null, opts,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            var treeCollection = pred.TrainedTreeEnsemble;
            Assert.Equal(0, treeCollection.Bias);
            Assert.Equal(10, treeCollection.Trees.Count);
            Assert.Equal(10, treeCollection.TreeWeights.Count);

            var trees = treeCollection.Trees;
            Assert.Equal(4, trees[0].NumberOfNodes);

            // Numerical split. There is no categorical split so the follwoing vector contains 0-element.
            var categoricalSplitFeatures = trees[0].GetCategoricalSplitFeaturesAt(0);
            Assert.Equal(0, categoricalSplitFeatures.Count);

            // Numerical split. There is no categorical split so the follwoing vector contains 0-element.
            var categoricalSplitFeatureRange = trees[0].GetCategoricalCategoricalSplitFeatureRangeAt(0);
            Assert.Equal(0, categoricalSplitFeatureRange.Count);

            var expectedGtChild = new int[] { 3, 2, -4, -5 };
            Assert.Equal(4, trees[0].RightChild.Count);
            Assert.Equal(expectedGtChild, trees[0].RightChild);

            var expectedLteChild = new int[] { 1, -1, -3, -2 };
            Assert.Equal(4, trees[0].LeftChild.Count);
            Assert.Equal(expectedLteChild, trees[0].LeftChild);

            var expectedCategoricalSplitFlags = new bool[] { false, false, false, false };
            Assert.Equal(4, trees[0].CategoricalSplitFlags.Count);
            Assert.Equal(expectedCategoricalSplitFlags, trees[0].CategoricalSplitFlags);

            var expectedNumericalSplitFeatureIndexes = new int[] { 0, 10, 2, 10 };
            Assert.Equal(4, trees[0].NumericalSplitFeatureIndexes.Count);
            Assert.Equal(expectedNumericalSplitFeatureIndexes, trees[0].NumericalSplitFeatureIndexes);

            var expectedNumericalSplitThresholds = new float[] { 0.14f, -0.645f, -0.095f, 0.31f };
            Assert.Equal(4, trees[0].NumericalSplitThresholds.Count);
            for (int i = 0; i < trees[0].NumericalSplitThresholds.Count; ++i)
                Assert.Equal(expectedNumericalSplitThresholds[i], trees[0].NumericalSplitThresholds[i], 6);

            Assert.Equal(5, trees[0].NumberOfLeaves);

            var expectedLeafValues = new double[] { 40.159015006449692, 80.434805844435061, 57.072130551545513, 82.898710076162757, 104.17547955322266 };
            Assert.Equal(5, trees[0].LeafValues.Count);
            for (int i = 0; i < trees[0].LeafValues.Count; ++i)
                Assert.Equal(expectedLeafValues[i], trees[0].LeafValues[i], 6);
        }

        [Fact]
        public void FastTreeRegressionRepresentationWithCategoricalSplit()
        {
            var env = new MLContext(seed: 0);
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataSource = new MultiFileSource(dataPath);

            var catalog = new RegressionCatalog(env);

            var reader = TextLoaderStatic.CreateLoader(env,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true);

            FastTreeRegressionModelParameters pred = null;

            var opts = new FastTreeRegressionTrainer.Options()
            {
                CategoricalSplit = true,
                NumberOfTrees = 3,
                NumberOfLeaves = 5,
                NumberOfThreads = 1,
                // This is the minimal samples to form a split (i.e., generating two extra nodes/leaves). For a small data set,
                // we should set a small value. Otherwise, the trained trees could be empty.
                MinimumExampleCountPerLeaf = 2
            };

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, features: r.features.OneHotEncoding()))
                .Append(r => (r.label, score: catalog.Trainers.FastTree(r.label, r.features, null, opts,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            var treeCollection = pred.TrainedTreeEnsemble;
            Assert.Equal(0, treeCollection.Bias);
            Assert.Equal(3, treeCollection.Trees.Count);
            Assert.Equal(3, treeCollection.TreeWeights.Count);

            var trees = treeCollection.Trees;
            Assert.Equal(4, trees[0].NumberOfNodes);

            var expectedGtChild = new int[] { 3, -3, -4, -5 };
            Assert.Equal(4, trees[0].RightChild.Count);
            Assert.Equal(expectedGtChild, trees[0].RightChild);

            var expectedLteChild = new int[] { 1, 2, -1, -2 };
            Assert.Equal(4, trees[0].LeftChild.Count);
            Assert.Equal(expectedLteChild, trees[0].LeftChild);

            var expectedCategoricalSplitFlags = new bool[] { true, true, true, true };
            Assert.Equal(4, trees[0].CategoricalSplitFlags.Count);
            Assert.Equal(expectedCategoricalSplitFlags, trees[0].CategoricalSplitFlags);

            var expectedNumericalSplitFeatureIndexes = new int[] { 5312, 2, 2126, 533 };
            Assert.Equal(4, trees[0].NumericalSplitFeatureIndexes.Count);
            Assert.Equal(expectedNumericalSplitFeatureIndexes, trees[0].NumericalSplitFeatureIndexes);

            var expectedNumericalSplitThresholds = new float[] { 0.5f, 0.5f, 0.5f, 0.5f };
            Assert.Equal(4, trees[0].NumericalSplitThresholds.Count);
            for (int i = 0; i < trees[0].NumericalSplitThresholds.Count; ++i)
                Assert.Equal(expectedNumericalSplitThresholds[i], trees[0].NumericalSplitThresholds[i], 6);

            var actualCategoricalRanges0 = trees[0].GetCategoricalCategoricalSplitFeatureRangeAt(0);
            Assert.Equal(actualCategoricalRanges0, new int[] { 5312, 5782 });

            var actualCategoricalRanges1 = trees[0].GetCategoricalCategoricalSplitFeatureRangeAt(1);
            Assert.Equal(actualCategoricalRanges1, new int[] { 2, 417 });

            var actualCategoricalRanges2 = trees[0].GetCategoricalCategoricalSplitFeatureRangeAt(2);
            Assert.Equal(actualCategoricalRanges2, new int[] { 2126, 2593 });

            var actualCategoricalRanges3 = trees[0].GetCategoricalCategoricalSplitFeatureRangeAt(3);
            Assert.Equal(actualCategoricalRanges3, new int[] { 533, 983 });

            int[] expectedCounts = { 62, 52, 54, 22 };
            int[] expectedStarts = { 5315, 10, 2141, 533 };
            int[] expectedEnds = { 5782, 401, 2558, 874 };
            for (int i = 0; i < trees[0].NumberOfNodes; ++i)
            {
                // Retrieve i-th node's split features.
                var actualCategoricalSplitFeatures = trees[0].GetCategoricalSplitFeaturesAt(i);
                Assert.Equal(expectedCounts[i], actualCategoricalSplitFeatures.Count);
                Assert.Equal(expectedStarts[i], actualCategoricalSplitFeatures[0]);
                Assert.Equal(expectedEnds[i], actualCategoricalSplitFeatures[expectedCounts[i] - 1]);
            }

            Assert.Equal(5, trees[0].NumberOfLeaves);

            var expectedLeafValues = new double[] { 48.456055413607892, 86.584156799316418, 87.017326642027, 76.381184971185391, 117.68872643673058 };
            Assert.Equal(5, trees[0].LeafValues.Count);
            for (int i = 0; i < trees[0].LeafValues.Count; ++i)
                Assert.Equal(expectedLeafValues[i], trees[0].LeafValues[i], 6);
        }
    }
}
