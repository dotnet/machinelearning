// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    /// <summary>
    /// Test explainability features.
    /// </summary>
    public class Explainability : BaseTestClass
    {
        public Explainability(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// GlobalFeatureImportance: PFI can be used to compute global feature importance.
        /// </summary>
        [Fact]
        public void GlobalFeatureImportanceWithPermutationFeatureImportance()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.Sdca());

            // Fit the pipeline and transform the data.
            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            // Compute the permutation feature importance to look at global feature importance.
            var permutationMetrics = mlContext.Regression.PermutationFeatureImportance(model.LastTransformer, transformedData);

            // Make sure the correct number of features came back.
            Assert.Equal(HousingRegression.Features.Length, permutationMetrics.Length);
            foreach (var metricsStatistics in permutationMetrics)
                Common.AssertMetricsStatistics(metricsStatistics);
        }

        /// <summary>
        /// GlobalFeatureImportance: A linear model's feature importance can be viewed through its weight coefficients.
        /// </summary>
        /// <remarks>
        /// Note that this isn't recommended, as there are quite a few statistical issues with interpreting coefficients
        /// as weights, but it's common practice, so it's a supported scenario.
        /// </remarks>
        [Fact]
        public void GlobalFeatureImportanceForLinearModelThroughWeights()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.Sdca());

            // Fit the pipeline and transform the data.
            var model = pipeline.Fit(data);
            var linearModel = model.LastTransformer.Model;

            // Make sure the number of model weights returned matches the length of the input feature vector.
            var weights = linearModel.Weights;
            Assert.Equal(HousingRegression.Features.Length, weights.Count);
        }

        /// <summary>
        /// GlobalFeatureImportance: A FastTree model can give back global feature importance through feature gain.
        /// </summary>
        [Fact]
        public void GlobalFeatureImportanceForFastTreeThroughFeatureGain()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastTree());

            // Fit the pipeline and transform the data.
            var model = pipeline.Fit(data);
            var treeModel = model.LastTransformer.Model;

            // Get the feature gain.
            var weights = new VBuffer<float>();
            treeModel.GetFeatureWeights(ref weights);

            // Make sure the number of feature gains returned matches the length of the input feature vector.
            Assert.Equal(HousingRegression.Features.Length, weights.Length);
        }

        /// <summary>
        /// GlobalFeatureImportance: A FastForest model can give back global feature importance through feature gain.
        /// </summary>
        [Fact]
        public void GlobalFeatureImportanceForFastForestThroughFeatureGain()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastForest());

            // Fit the pipeline and transform the data.
            var model = pipeline.Fit(data);
            var treeModel = model.LastTransformer.Model;

            // Get the feature gain
            var weights = new VBuffer<float>();
            treeModel.GetFeatureWeights(ref weights);

            // Make sure the number of feature gains returned matches the length of the input feature vector.
            Assert.Equal(HousingRegression.Features.Length, weights.Length);
        }

        /// <summary>
        /// LocalFeatureImportance: Per-row feature importance can be computed through FeatureContributionCalculator for a linear model.
        /// </summary>
        [Fact]
        public void LocalFeatureImportanceForLinearModel()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.Sdca());

            // Fit the pipeline and transform the data.
            var model = pipeline.Fit(data);
            var scoredData = model.Transform(data);

            // Create a Feature Contribution Calculator.
            var predictor = model.LastTransformer;
            var featureContributions = mlContext.Transforms.CalculateFeatureContribution(predictor, normalize: false);

            // Compute the contributions
            var outputData = featureContributions.Fit(scoredData).Transform(scoredData);

            // Validate that the contributions are there
            var shuffledSubset = mlContext.Data.TakeRows(mlContext.Data.ShuffleRows(outputData), 10);
            var scoringEnumerator = mlContext.Data.CreateEnumerable<FeatureContributionOutput>(shuffledSubset, true);

            // Make sure the number of feature contributions returned matches the length of the input feature vector.
            foreach (var row in scoringEnumerator)
            {
                Assert.Equal(HousingRegression.Features.Length, row.FeatureContributions.Length);
            }
        }

        /// <summary>
        /// LocalFeatureImportance: Per-row feature importance can be computed through FeatureContributionCalculator for a <see cref="FastTree"/> model.
        /// </summary>
        [Fact]
        public void LocalFeatureImportanceForFastTreeModel()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastTree());

            // Fit the pipeline and transform the data.
            var model = pipeline.Fit(data);
            var scoredData = model.Transform(data);

            // Create a Feature Contribution Calculator.
            var predictor = model.LastTransformer;
            var featureContributions = mlContext.Transforms.CalculateFeatureContribution(predictor, normalize: false);

            // Compute the contributions
            var outputData = featureContributions.Fit(scoredData).Transform(scoredData);

            // Validate that the contributions are there
            var shuffledSubset = mlContext.Data.TakeRows(mlContext.Data.ShuffleRows(outputData), 10);
            var scoringEnumerator = mlContext.Data.CreateEnumerable<FeatureContributionOutput>(shuffledSubset, true);

            // Make sure the number of feature contributions returned matches the length of the input feature vector.
            foreach (var row in scoringEnumerator)
            {
                Assert.Equal(HousingRegression.Features.Length, row.FeatureContributions.Length);
            }
        }

        /// <summary>
        /// LocalFeatureImportance: Per-row feature importance can be computed through FeatureContributionCalculator for a <see cref="FastForest"/>model.
        /// </summary>
        [Fact]
        public void LocalFeatureImportanceForFastForestModel()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.FastForest());

            // Fit the pipeline and transform the data.
            var model = pipeline.Fit(data);
            var scoredData = model.Transform(data);

            // Create a Feature Contribution Calculator.
            var predictor = model.LastTransformer;
            var featureContributions = mlContext.Transforms.CalculateFeatureContribution(predictor, normalize: false);

            // Compute the contributions
            var outputData = featureContributions.Fit(scoredData).Transform(scoredData);

            // Validate that the contributions are there
            var shuffledSubset = mlContext.Data.TakeRows(mlContext.Data.ShuffleRows(outputData), 10);
            var scoringEnumerator = mlContext.Data.CreateEnumerable<FeatureContributionOutput>(shuffledSubset, true);

            // Make sure the number of feature contributions returned matches the length of the input feature vector.
            foreach (var row in scoringEnumerator)
            {
                Assert.Equal(HousingRegression.Features.Length, row.FeatureContributions.Length);
            }
        }

        /// <summary>
        /// LocalFeatureImportance: Per-row feature importance can be computed through FeatureContributionCalculator for a <see cref="GamModelParametersBase" />
        /// (Generalized Additive Model) model.
        /// </summary>
        [Fact]
        public void LocalFeatureImportanceForGamModel()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.Gam(numberOfIterations: 2));

            // Fit the pipeline and transform the data.
            var model = pipeline.Fit(data);
            var scoredData = model.Transform(data);

            // Create a Feature Contribution Calculator.
            var predictor = model.LastTransformer;
            var featureContributions = mlContext.Transforms.CalculateFeatureContribution(predictor, normalize: false);

            // Compute the contributions
            var outputData = featureContributions.Fit(scoredData).Transform(scoredData);

            // Validate that the contributions are there
            var shuffledSubset = mlContext.Data.TakeRows(mlContext.Data.ShuffleRows(outputData), 10);
            var scoringEnumerator = mlContext.Data.CreateEnumerable<FeatureContributionOutput>(shuffledSubset, true);

            // Make sure the number of feature contributions returned matches the length of the input feature vector.
            foreach (var row in scoringEnumerator)
            {
                Assert.Equal(HousingRegression.Features.Length, row.FeatureContributions.Length);
            }
        }
    }
}
