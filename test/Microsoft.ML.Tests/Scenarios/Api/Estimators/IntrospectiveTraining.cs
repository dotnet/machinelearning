// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.RunTests;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{

    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Introspective training: Models that produce outputs and are otherwise black boxes are of limited use;
        /// it is also necessary often to understand at least to some degree what was learnt. To outline critical
        /// scenarios that have come up multiple times:
        ///  *) When I train a linear model, I should be able to inspect coefficients.
        ///  *) The tree ensemble learners, I should be able to inspect the trees.
        ///  *) The LDA transform, I should be able to inspect the topics.
        ///  I view it as essential from a usability perspective that this be discoverable to someone without 
        ///  having to read documentation. For example, if I have var lda = new LdaTransform().Fit(data)(I don't insist on that
        ///  exact signature, just giving the idea), then if I were to type lda.
        ///  In Visual Studio, one of the auto-complete targets should be something like GetTopics.
        /// </summary>

        [Fact]
        public void IntrospectiveTraining()
        {
            var ml = new MLContext(seed: 1, conc: 1);
            var data = ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.trainFilename), hasHeader: true);

            var pipeline = ml.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(ml)
                .Append(ml.BinaryClassification.Trainers.StochasticDualCoordinateAscent(
                    new SdcaBinaryTrainer.Options { NumThreads = 1 }));

            // Train.
            var model = pipeline.Fit(data);

            // Get feature weights.
            VBuffer<float> weights = default;
            model.LastTransformer.Model.GetFeatureWeights(ref weights);
        }

        [Fact]
        public void FastTreeClassificationIntrospectiveTraining()
        {
            var ml = new MLContext(seed: 1, conc: 1);
            var data = ml.Data.ReadFromTextFile<SentimentData>(GetDataPath(TestDatasets.Sentiment.trainFilename), hasHeader: true);

            var trainer = ml.BinaryClassification.Trainers.FastTree(numLeaves: 5, numTrees: 3);

            BinaryPredictionTransformer<IPredictorWithFeatureWeights<float>> pred = null;

            var pipeline = ml.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(ml)
                .Append(trainer.WithOnFitDelegate(p => pred = p));

            // Train.
            var model = pipeline.Fit(data);

            // Extract the learned GBDT model.
            var treeCollection = ((FastTreeBinaryModelParameters)((Internal.Calibration.FeatureWeightsCalibratedPredictor)pred.Model).SubPredictor).TrainedTreeEnsemble;

            // Inspect properties in the extracted model.
            Assert.Equal(3, treeCollection.Trees.Count);
            Assert.Equal(3, treeCollection.TreeWeights.Count);
            Assert.Equal(0, treeCollection.Bias);
            Assert.All(treeCollection.TreeWeights, weight => Assert.Equal(1.0, weight));

            // Inspect the last tree.
            var tree = treeCollection.Trees[2];

            Assert.Equal(5, tree.NumLeaves);
            Assert.Equal(4, tree.NumNodes);
            Assert.Equal(tree.LteChild, new int[] { 2, -2, -1, -3 });
            Assert.Equal(tree.GtChild, new int[] { 1, 3, -4, -5 });
            Assert.Equal(tree.NumericalSplitFeatureIndexes, new int[] { 14, 294, 633, 266 });
            var expectedThresholds = new float[] { 0.0911167f, 0.06509889f, 0.019873254f, 0.0361835f };
            for (int i = 0; i < tree.NumNodes; ++i)
                Assert.Equal(expectedThresholds[i], tree.NumericalSplitThresholds[i], 6);
            Assert.All(tree.CategoricalSplitFlags, flag => Assert.False(flag));

            Assert.Equal(0, tree.GetCategoricalSplitFeaturesAt(0).Count);
            Assert.Equal(0, tree.GetCategoricalCategoricalSplitFeatureRangeAt(0).Count);
        }

        [Fact]
        public void FastForestRegressionIntrospectiveTraining()
        {
            var ml = new MLContext(seed: 1, conc: 1);
            var data = DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(1000);
            var dataView = ml.Data.ReadFromEnumerable(data);

            RegressionPredictionTransformer<FastForestRegressionModelParameters> pred = null;
            var trainer = ml.Regression.Trainers.FastForest(numLeaves: 5, numTrees: 3).WithOnFitDelegate(p => pred = p);

            // Train.
            var model = trainer.Fit(dataView);

            // Extract the learned RF model.
            var treeCollection = pred.Model.TrainedTreeEnsemble;

            // Inspect properties in the extracted model.
            Assert.Equal(3, treeCollection.Trees.Count);
            Assert.Equal(3, treeCollection.TreeWeights.Count);
            Assert.Equal(0, treeCollection.Bias);
            Assert.All(treeCollection.TreeWeights, weight => Assert.Equal(1.0, weight));

            // Inspect the last tree.
            var tree = treeCollection.Trees[2];

            Assert.Equal(5, tree.NumLeaves);
            Assert.Equal(4, tree.NumNodes);
            Assert.Equal(tree.LteChild, new int[] { -1, -2, -3, -4 });
            Assert.Equal(tree.GtChild, new int[] { 1, 2, 3, -5 });
            Assert.Equal(tree.NumericalSplitFeatureIndexes, new int[] { 9, 0, 1, 8 });
            var expectedThresholds = new float[] { 0.208134219f, 0.198336035f, 0.202952743f, 0.205061346f };
            for (int i = 0; i < tree.NumNodes; ++i)
                Assert.Equal(expectedThresholds[i], tree.NumericalSplitThresholds[i], 6);
            Assert.All(tree.CategoricalSplitFlags, flag => Assert.False(flag));

            Assert.Equal(0, tree.GetCategoricalSplitFeaturesAt(0).Count);
            Assert.Equal(0, tree.GetCategoricalCategoricalSplitFeatureRangeAt(0).Count);

            var samples = new double[] { 0.97468354430379744, 1.0, 0.97727272727272729, 0.972972972972973, 0.26124197002141325 };
            for (int i = 0; i < tree.NumLeaves; ++i)
            {
                var sample = tree.GetLeafSamplesAt(i);
                Assert.Single(sample);
                Assert.Equal(samples[i], sample[0], 6);
                var weight = tree.GetLeafSampleWeightsAt(i);
                Assert.Single(weight);
                Assert.Equal(1, weight[0]);
            }
        }
    }
}
