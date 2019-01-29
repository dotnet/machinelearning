using System;
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

            var reader = TextLoaderStatic.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true);

            FastTreeRegressionModelParameters pred = null;

            var est = reader.MakeNewEstimator()
                .Append(r => (r.label, score: catalog.Trainers.FastTree(r.label, r.features,
                    numTrees: 10,
                    numLeaves: 5,
                    onFit: (p) => { pred = p; })));

            var pipe = reader.Append(est);

            Assert.Null(pred);
            var model = pipe.Fit(dataSource);
            Assert.NotNull(pred);

            var treeCollection = pred.TrainedTreeCollection;
            Assert.Equal(0, treeCollection.Bias);
            Assert.Equal(10, treeCollection.Trees.Count);
            Assert.Equal(10, treeCollection.TreeWeights.Count);

            var trees = treeCollection.Trees;
            Assert.Equal(4, trees[0].NumNodes);

            var expectedGtChild = new int[] { 3, 2, -4, -5 };
            Assert.Equal(4, trees[0].GtChild.Length);
            Assert.Equal(expectedGtChild, trees[0].GtChild.ToArray());

            var expectedLteChild = new int[] { 1, -1, -3, -2 };
            Assert.Equal(4, trees[0].LteChild.Length);
            Assert.Equal(expectedLteChild, trees[0].LteChild.ToArray());

            var expectedCategoricalSplitFlags = new bool[] { false, false, false, false };
            Assert.Equal(4, trees[0].CategoricalSplitFlags.Length);
            Assert.Equal(expectedCategoricalSplitFlags, trees[0].CategoricalSplitFlags.ToArray());

            var expectedNumericalSplitFeatureIndexes = new int[] { 0, 10, 2, 10 };
            Assert.Equal(4, trees[0].NumericalSplitFeatureIndexes.Length);
            Assert.Equal(expectedNumericalSplitFeatureIndexes, trees[0].NumericalSplitFeatureIndexes.ToArray());

            var expectedNumericalSplitThresholds = new float[] { 0.14f, -0.645f, -0.095f, 0.31f };
            Assert.Equal(4, trees[0].NumericalSplitThresholds.Length);
            for (int i = 0; i < trees[0].NumericalSplitThresholds.Length; ++i)
                Assert.Equal(expectedNumericalSplitThresholds[i], trees[0].NumericalSplitThresholds[i], 6);

            Assert.Equal(5, trees[0].NumLeaves);

            var expectedLeafValues = new double[] { 40.159015006449692, 80.434805844435061, 57.072130551545513, 82.898710076162757, 104.17547955322266 };
            Assert.Equal(5, trees[0].LeafValues.Length);
            for (int i = 0; i < trees[0].LeafValues.Length; ++i)
                Assert.Equal(expectedLeafValues[i], trees[0].LeafValues[i], 6);

            var sampledLabels = trees[0].GetLeafSamplesAt(0);
            Assert.Equal(1, sampledLabels.Length);
            Assert.Equal(40.159015006449692, sampledLabels[0], 6);

            var sampledLabelWeights = trees[0].GetLeafSampleWeightsAt(0);
            Assert.Equal(1, sampledLabelWeights.Length);
            Assert.Equal(1, sampledLabelWeights[0], 6);
        }
    }
}
