// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class IntrospectiveTraining : BaseTestClass
    {
        public IntrospectiveTraining(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// Introspective Training: Tree ensembles learned from FastForest can be inspected.
        /// </summary>
        [Fact]
        public void InspectFastForestRegresionTrees()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastForest(
                    new FastForestRegressionTrainer.Options { NumberOfLeaves = 5, NumberOfTrees = 3, NumberOfThreads = 1 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Extract the boosted tree model.
            var fastForestModel = model.LastTransformer.Model;

            // Extract the learned Random Forests model.
            var treeCollection = fastForestModel.TrainedTreeEnsemble;

            // Inspect properties in the extracted model.
            Assert.Equal(3, treeCollection.Trees.Count);
            Assert.Equal(3, treeCollection.TreeWeights.Count);
            Assert.All(treeCollection.TreeWeights, weight => Assert.Equal(1.0, weight));
            Assert.All(treeCollection.Trees, tree =>
            {
                Assert.Equal(5, tree.NumberOfLeaves);
                Assert.Equal(4, tree.NumberOfNodes);
                Assert.Equal(tree.SplitGains.Count, tree.NumberOfNodes);
                Assert.Equal(tree.NumericalSplitThresholds.Count, tree.NumberOfNodes);
                Assert.All(tree.CategoricalSplitFlags, flag => Assert.False(flag));
                Assert.Equal(0, tree.GetCategoricalSplitFeaturesAt(0).Count);
                Assert.Equal(0, tree.GetCategoricalCategoricalSplitFeatureRangeAt(0).Count);
            });
        }

        /// <summary>
        /// Introspective Training: Tree ensembles learned from FastTree can be inspected.
        /// </summary>
        [Fact]
        public void InspectFastTreeModelParameters()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.FastTree(
                    new FastTreeBinaryTrainer.Options{ NumberOfLeaves = 5, NumberOfTrees= 3, NumberOfThreads = 1 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Extract the boosted tree model.
            var fastTreeModel = model.LastTransformer.Model.SubModel;

            // Extract the learned GBDT model.
            var treeCollection = fastTreeModel.TrainedTreeEnsemble;

            // Make sure the tree models were formed as expected.
            Assert.Equal(3, treeCollection.Trees.Count);
            Assert.Equal(3, treeCollection.TreeWeights.Count);
            Assert.All(treeCollection.TreeWeights, weight => Assert.Equal(1.0, weight));
            Assert.All(treeCollection.Trees, tree =>
            {
                Assert.Equal(5, tree.NumberOfLeaves);
                Assert.Equal(4, tree.NumberOfNodes);
                Assert.Equal(tree.SplitGains.Count, tree.NumberOfNodes);
                Assert.Equal(tree.NumericalSplitThresholds.Count, tree.NumberOfNodes);
                Assert.All(tree.CategoricalSplitFlags, flag => Assert.False(flag));
                Assert.Equal(0, tree.GetCategoricalSplitFeaturesAt(0).Count);
                Assert.Equal(0, tree.GetCategoricalCategoricalSplitFeatureRangeAt(0).Count);
            });

            // Add baselines for the model.
            // Verify that there is no bias.
            Assert.Equal(0, treeCollection.Bias);
            // Check the parameters of the final tree.
            var finalTree = treeCollection.Trees[2];
            Assert.Equal(finalTree.LeftChild, new int[] { 2, -2, -1, -3 });
            Assert.Equal(finalTree.RightChild, new int[] { 1, 3, -4, -5 });
            Assert.Equal(finalTree.NumericalSplitFeatureIndexes, new int[] { 14, 294, 633, 266 });
            var expectedSplitGains = new double[] { 0.52634223978445616, 0.45899249367725858, 0.44142707650267105, 0.38348634823264854 };
            var expectedThresholds = new float[] { 0.0911167f, 0.06509889f, 0.019873254f, 0.0361835f };
            for (int i = 0; i < finalTree.NumberOfNodes; ++i)
            {
                Assert.Equal(expectedSplitGains[i], finalTree.SplitGains[i], 6);
                Assert.Equal(expectedThresholds[i], finalTree.NumericalSplitThresholds[i], 6);
            }
        }

        /// <summary>
        /// Introspective Training: GAM Shape Functions are easily accessed.
        /// </summary>
        [Fact]
        void IntrospectGamShapeFunctions()
        {
            // Concurrency must be 1 to assure that the mapping is done sequentially.
            var mlContext = new MLContext(seed: 1);

            // Load the Iris dataset.
            var data = mlContext.Data.LoadFromTextFile<Iris>(
                GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Compose the transformation.
            var pipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .Append(mlContext.Regression.Trainers.Gam(
                    new GamRegressionTrainer.Options { NumberOfIterations = 100, NumberOfThreads = 1 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Extract the normalizer from the trained pipeline.
            var gamModel = model.LastTransformer.Model;

            // Take look at the shape functions.
            for (int i = 0; i < gamModel.NumberOfShapeFunctions; i++)
            {
                var shapeFunctionBins = gamModel.GetBinUpperBounds(i);
                var shapeFunctionValues = gamModel.GetBinEffects(i);

                // Validate that the shape functions lengths match.
                Assert.Equal(shapeFunctionBins.Count, shapeFunctionValues.Count);
                Common.AssertFiniteNumbers(shapeFunctionBins as IList<double>, shapeFunctionBins.Count - 1);
                Common.AssertFiniteNumbers(shapeFunctionValues as IList<double>);
            }
        }

        /// <summary>
        /// Introspective Training: LDA models can be easily inspected.
        /// </summary>
        [Fact]
        public void InspectLdaModelParameters()
        {
            // Test Parameters
            int numTopics = 10;

            var mlContext = new MLContext(seed: 1);

            // Load the dataset.
            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);

            // Define the pipeline.
            var pipeline = mlContext.Transforms.Text.ProduceWordBags("SentimentBag", "SentimentText")
                .Append(mlContext.Transforms.Text.LatentDirichletAllocation("Features", "SentimentBag", numberOfTopics: numTopics, maximumNumberOfIterations: 10));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Get the trained LDA model.
            // TODO #2197: Get the topics and summaries from the model.
            var ldaTransform = model.LastTransformer;

            // Transform the data.
            var transformedData = model.Transform(data);

            // Make sure the model weights array is the same length as the features array.
            var numFeatures = (transformedData.Schema["Features"].Type as VectorType).Size;
            Assert.Equal(numFeatures, numTopics);
        }

        /// <summary>
        /// Introspective Training: Linear model parameters may be inspected.
        /// </summary>
        [Fact]
        public void InpsectLinearModelParameters()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(
                    new SdcaNonCalibratedBinaryTrainer.Options { NumberOfThreads = 1 }));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Transform the data.
            var transformedData = model.Transform(data);

            // Extract the linear model from the pipeline.
            var linearModel = model.LastTransformer.Model;

            // Get the model bias and weights.
            var bias = linearModel.Bias;
            var weights = linearModel.Weights;

            // Make sure the model weights array is the same length as the features array.
            var numFeatures = (transformedData.Schema["Features"].Type as VectorType).Size;
            Assert.Equal(numFeatures, weights.Count);
        }

        /// <summary>
        /// Introspectable Training: Parameters of a trained Normalizer are easily accessed.
        /// </summary>
        [Fact]
        void IntrospectNormalization()
        {
            // Concurrency must be 1 to assure that the mapping is done sequentially.
            var mlContext = new MLContext(seed: 1);

            // Load the Iris dataset.
            var data = mlContext.Data.LoadFromTextFile<Iris>(
                GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Compose the transformation.
            var pipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .Append(mlContext.Transforms.Normalize("Features", mode: NormalizingEstimator.NormalizationMode.MinMax));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Extract the normalizer from the trained pipeline.
            var normalizer = model.LastTransformer;

            // Extract the normalizer parameters.
            // TODO #2854: Normalizer parameters are easy to find via intellisense.
            var config = normalizer.GetNormalizerModelParameters(0) as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<float>>;
            Assert.NotNull(config);
            Common.AssertFiniteNumbers(config.Offset);
            Common.AssertFiniteNumbers(config.Scale);
        }
        /// <summary>
        /// Introspective Training: I can inspect a pipeline to determine which transformers were included. 	 
        /// </summary>
        [Fact]
        public void InspectPipelineContents()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastForest(numberOfLeaves: 5, numberOfTrees: 3));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Inspect the transforms in the trained pipeline.
            var expectedTypes = new Type[] {typeof(ColumnConcatenatingTransformer),
                typeof(RegressionPredictionTransformer<FastForestRegressionModelParameters>)};
            var expectedColumns = new string[][] {
                new string[] { "Features" },
                new string[] { "Score" },
            };
            int i = 0;
            var currentSchema = data.Schema;
            foreach (var transformer in model)
            {
                // It is possible to get the type at runtime.
                Assert.IsType(expectedTypes[i], transformer);
                
                // It's also possible to inspect the schema output from the transform.
                currentSchema = transformer.GetOutputSchema(currentSchema);
                foreach (var expectedColumn in expectedColumns[i])
                {
                    var column = currentSchema.GetColumnOrNull(expectedColumn);
                    Assert.NotNull(column);
                }
                i++;
            }
        }

        /// <summary>
        /// Introspective Training: Hashed values can be mapped back to the original column and value.
        /// </summary>
        [Fact]
        public void InspectSlotNamesForReversibleHash()
        {
            var mlContext = new MLContext(seed: 1);

            // Load the Adult dataset.
            var data = mlContext.Data.LoadFromTextFile<Adult>(GetDataPath(TestDatasets.adult.trainFilename),
                hasHeader: TestDatasets.adult.fileHasHeader,
                separatorChar: TestDatasets.adult.fileSeparator);

            // Create the learning pipeline.
            var pipeline = mlContext.Transforms.Concatenate("NumericalFeatures", Adult.NumericalFeatures)
                .Append(mlContext.Transforms.Concatenate("CategoricalFeatures", Adult.CategoricalFeatures))
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("CategoricalFeatures", numberOfBits: 8, // get collisions!
                    maximumNumberOfInverts: -1, outputKind: OneHotEncodingEstimator.OutputKind.Bag));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Transform the data.
            var transformedData = model.Transform(data);

            // Verify that the slotnames can be used to backtrack to the original values by confirming that 
            // all unique values in the input data are in the output data slot names.
            // First get a list of the unique values.
            VBuffer<ReadOnlyMemory<char>> categoricalSlotNames = new VBuffer<ReadOnlyMemory<char>>();
            transformedData.Schema["CategoricalFeatures"].GetSlotNames(ref categoricalSlotNames);
            var uniqueValues = new HashSet<string>();
            foreach (var slotName in categoricalSlotNames.GetValues())
            {
                var slotNameString = slotName.ToString();
                if (slotNameString.StartsWith("{"))
                {
                    // Values look like this: {3:Exec-managerial,2:Widowed}.
                    slotNameString = slotNameString.Substring(1, slotNameString.Length - 2);
                    foreach (var name in slotNameString.Split(','))
                        uniqueValues.Add(name);
                }
                else
                    uniqueValues.Add(slotNameString);
            }

            // Now validate that all values in the dataset are there.
            var transformedRows = mlContext.Data.CreateEnumerable<Adult>(data, false);
            foreach (var row in transformedRows)
            {
                for (int i = 0; i < Adult.CategoricalFeatures.Length; i++)
                {
                    // Fetch the categorical value.
                    string value = (string) row.GetType().GetProperty(Adult.CategoricalFeatures[i]).GetValue(row, null);
                    Assert.Contains($"{i}:{value}", uniqueValues);
                }
            }
        }
        
        /// <summary>
        /// Introspective Training: I can create nested pipelines, and extract individual components.
        /// </summary>
        [Fact]
        public void InspectNestedPipeline()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<Iris>(GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Create a training pipeline.
            var pipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .Append(StepOne(mlContext))
                .Append(StepTwo(mlContext));

            // Fit the pipeline.
            var model = pipeline.Fit(data);

            // Extract the trained models.
            var modelComponents = model.ToList();
            var kMeansModel = (modelComponents[1] as TransformerChain<ClusteringPredictionTransformer<KMeansModelParameters>>).LastTransformer;
            var mcLrModel = (modelComponents[2] as TransformerChain<MulticlassPredictionTransformer<MaximumEntropyModelParameters>>).LastTransformer;

            // Validate the k-means model.
            VBuffer<float>[] centroids = default;
            kMeansModel.Model.GetClusterCentroids(ref centroids, out int nCentroids);
            Assert.Equal(4, centroids.Length);

            // Validate the MulticlassLogisticRegressionModel.
            VBuffer<float>[] weights = default;
            mcLrModel.Model.GetWeights(ref weights, out int classes);
            Assert.Equal(3, weights.Length);
        }

        private IEstimator<TransformerChain<ClusteringPredictionTransformer<KMeansModelParameters>>> StepOne(MLContext mlContext)
        {
            return mlContext.Transforms.Concatenate("LabelAndFeatures", "Label", "Features")
                .Append(mlContext.Clustering.Trainers.KMeans(
                    new KMeansTrainer.Options
                    {
                        InitializationAlgorithm = KMeansTrainer.InitializationAlgorithm.Random,
                        NumberOfClusters = 4,
                        MaximumNumberOfIterations = 10,
                        NumberOfThreads = 1
                    }));
        }

        private IEstimator<TransformerChain<MulticlassPredictionTransformer<MaximumEntropyModelParameters>>> StepTwo(MLContext mlContext)
        {
            return mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.MulticlassClassification.Trainers.SdcaCalibrated(
                new SdcaCalibratedMulticlassTrainer.Options {
                    MaximumNumberOfIterations = 10,
                    NumberOfThreads = 1 }));
        }
    }
}