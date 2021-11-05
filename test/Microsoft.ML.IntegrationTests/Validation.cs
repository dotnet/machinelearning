// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.IntegrationTests.Datasets;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.IntegrationTests
{
    public class Validation : IntegrationTestBaseClass
    {
        public Validation(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// Cross-validation: Have a mechanism to do cross validation, that is, you come up with
        /// a data source (optionally with stratification column), come up with an instantiable transform
        /// and trainer pipeline, and it will handle (1) splitting up the data, (2) training the separate
        /// pipelines on in-fold data, (3) scoring on the out-fold data, (4) returning the set of
        /// metrics, trained pipelines, and scored test data for each fold.
        /// </summary>
        [NativeDependencyFact("MklImports")]
        public void CrossValidation()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(TestCommon.GetDataPath(DataDir, TestDatasets.housing.trainFilename), hasHeader: true);

            // Create a pipeline to train on the housing data.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Regression.Trainers.Ols());

            // Compute the CV result.
            var cvResult = mlContext.Regression.CrossValidate(data, pipeline, numberOfFolds: 5);

            // Check that the results are valid
            Assert.IsType<RegressionMetrics>(cvResult[0].Metrics);
            Assert.IsType<TransformerChain<RegressionPredictionTransformer<OlsModelParameters>>>(cvResult[0].Model);
            Assert.True(cvResult[0].ScoredHoldOutSet is IDataView);
            Assert.Equal(5, cvResult.Count);

            // And validate the metrics.
            foreach (var result in cvResult)
                Common.AssertMetrics(result.Metrics);
        }

        [Fact]
        public void RankingCVTest()
        {
            string labelColumnName = "Label";
            string groupIdColumnName = "GroupId";
            string featuresColumnVectorNameA = "FeatureVectorA";
            string featuresColumnVectorNameB = "FeatureVectorB";
            int numFolds = 3;

            var mlContext = new MLContext(1);
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "FeatureVectorA", "FeatureVectorB" }).Append(
                mlContext.Transforms.Conversion.Hash("GroupId", "GroupId"));

            var trainer = mlContext.Ranking.Trainers.FastTree(new FastTreeRankingTrainer.Options()
            { RowGroupColumnName = "GroupId", LabelColumnName = "Label", FeatureColumnName = "Features" });
            var reader = mlContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Separators = new[] { '\t' },
                HasHeader = true,
                Columns = new[]
                {
                    new TextLoader.Column(labelColumnName, DataKind.Single, 0),
                    new TextLoader.Column(groupIdColumnName, DataKind.Int32, 1),
                    new TextLoader.Column(featuresColumnVectorNameA, DataKind.Single, 2, 9),
                    new TextLoader.Column(featuresColumnVectorNameB, DataKind.Single, 10, 137)
                }
            });
            var trainDataView = reader.Load(TestCommon.GetDataPath(DataDir, "MSLRWeb1K-tiny.tsv"));
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var result = mlContext.Ranking.CrossValidate(trainDataView, trainingPipeline, numberOfFolds: numFolds);
            for (int i = 0; i < numFolds; i++)
            {
                Assert.True(result[i].Metrics.NormalizedDiscountedCumulativeGains.Max() > .4);
                Assert.True(result[i].Metrics.DiscountedCumulativeGains.Max() > 16);
            }
        }

        /// <summary>
        /// Train with validation set.
        /// </summary>
        [Fact]
        public void TrainWithValidationSet()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(TestCommon.GetDataPath(DataDir, TestDatasets.housing.trainFilename), hasHeader: true);

            // Create the train and validation set.
            var dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = dataSplit.TrainSet;
            var validData = dataSplit.TestSet;

            // Create a pipeline to featurize the dataset.
            var pipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .AppendCacheCheckpoint(mlContext) as IEstimator<ITransformer>;

            // Preprocess the datasets.
            var preprocessor = pipeline.Fit(trainData);
            var preprocessedTrainData = preprocessor.Transform(trainData);
            var preprocessedValidData = preprocessor.Transform(validData);

            // Train the model with a validation set.
            var trainedModel = mlContext.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options
            {
                NumberOfTrees = 2,
                EarlyStoppingMetric = EarlyStoppingMetric.L2Norm,
                EarlyStoppingRule = new GeneralityLossRule()
            })
                .Fit(trainData: preprocessedTrainData, validationData: preprocessedValidData);

            // Combine the model.
            var model = preprocessor.Append(trainedModel);

            // Score the data sets.
            var scoredTrainData = model.Transform(trainData);
            var scoredValidData = model.Transform(validData);

            var trainMetrics = mlContext.Regression.Evaluate(scoredTrainData);
            var validMetrics = mlContext.Regression.Evaluate(scoredValidData);

            Common.AssertMetrics(trainMetrics);
            Common.AssertMetrics(validMetrics);
        }

        /// <summary>
        /// Test cross validation R^2 metric to return NaN when given fewer data
        /// than needed to infer metric calculation. R^2 is NaN when given folds
        /// with less than 2 rows of training data.
        /// </summary>
        [Fact]
        public void TestCrossValidationResultsWithNotEnoughData()
        {
            var mlContext = new MLContext(1);
            // Get data and set up sample regression pipeline.
            var data = mlContext.Data.LoadFromTextFile<Iris>(TestCommon.GetDataPath(DataDir, TestDatasets.iris.trainFilename), hasHeader: true);
            var pipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .Append(mlContext.Regression.Trainers.OnlineGradientDescent());
            // Train model with full dataset
            var model = pipeline.Fit(data);

            // Check that R^2 is NaN when given 1 row of scoring data.
            var scoredDataOneRow = model.Transform(mlContext.Data.TakeRows(data, 1));
            var evalResultOneRow = mlContext.Regression.Evaluate(scoredDataOneRow);
            Assert.Equal(double.NaN, evalResultOneRow.RSquared);

            // Check that R^2 is 0 when given 0 rows of scoring data.
            // Obtain empty IDataView with Iris schema as there are no rows of data with labels between -2 and -1.
            var scoredDataZeroRows = mlContext.Data.FilterRowsByColumn(scoredDataOneRow, "Label", lowerBound: -2, upperBound: -1);
            var evalResultZeroRows = mlContext.Regression.Evaluate(scoredDataZeroRows);
            Assert.Equal(0, evalResultZeroRows.RSquared);
        }
    }
}
