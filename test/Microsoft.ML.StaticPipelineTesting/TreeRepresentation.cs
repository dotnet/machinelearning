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

            // GtChild is [3, 2, -4, -5].
            Assert.Equal(4, trees[0].GtChild.Length);
            Assert.Equal(3, trees[0].GtChild[0]);
            Assert.Equal(-5, trees[0].GtChild[3]);

            // LteChild is [1, -1, -3, -2].
            Assert.Equal(4, trees[0].LteChild.Length);
            Assert.Equal(1, trees[0].LteChild[0]);
            Assert.Equal(-2, trees[0].LteChild[3]);

            // CategoricalSplitFlags is [false, false, false, false].
            Assert.Equal(4, trees[0].CategoricalSplitFlags.Length);
            Assert.False(trees[0].CategoricalSplitFlags[0]);
            Assert.False(trees[0].CategoricalSplitFlags[3]);

            // NumericalSplitFeatureIndexes is [0, 10, 2, 10]
            Assert.Equal(4, trees[0].NumericalSplitFeatureIndexes.Length);
            Assert.Equal(0, trees[0].NumericalSplitFeatureIndexes[0]);
            Assert.Equal(10, trees[0].NumericalSplitFeatureIndexes[3]);

            // NumericalSplitThresholds is [0.14, -0.645, -0.095, 0.31].
            Assert.Equal(4, trees[0].NumericalSplitThresholds.Length);
            Assert.Equal(0.14, trees[0].NumericalSplitThresholds[0], 6);
            Assert.Equal(0.31, trees[0].NumericalSplitThresholds[3], 6);

            Assert.Equal(5, trees[0].NumLeaves);

            // Values in LeafValues:
            // [0]	40.159015006449692	double
            // [1]	80.434805844435061	double
            // [2]	57.072130551545513	double
            // [3]	82.898710076162757	double
            // [4]	104.17547955322266	double
            Assert.Equal(5, trees[0].LeafValues.Length);
            Assert.Equal(40.159015006449692, trees[0].LeafValues[0], 6);
            Assert.Equal(104.17547955322266, trees[0].LeafValues[4], 6);
        }
    }
}
