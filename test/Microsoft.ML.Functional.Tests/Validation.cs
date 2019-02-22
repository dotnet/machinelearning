// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers.HalLearners;
using Xunit;

namespace Microsoft.ML.Functional.Tests
{
    public class ValidationScenarios
    {
        /// <summary>
        /// Cross-validation: Have a mechanism to do cross validation, that is, you come up with
        /// a data source (optionally with stratification column), come up with an instantiable transform
        /// and trainer pipeline, and it will handle (1) splitting up the data, (2) training the separate
        /// pipelines on in-fold data, (3) scoring on the out-fold data, (4) returning the set of
        /// metrics, trained pipelines, and scored test data for each fold.
        /// </summary>
        [Fact]
        void CrossValidation()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Get the dataset.
            var data = mlContext.Data.CreateTextLoader(TestDatasets.housing.GetLoaderColumns(),
                    hasHeader: TestDatasets.housing.fileHasHeader, separatorChar: TestDatasets.housing.fileSeparator)
                .Read(BaseTestClass.GetDataPath(TestDatasets.housing.trainFilename));

            // Create a pipeline to train on the sentiment data.
            var pipeline = mlContext.Transforms.Concatenate("Features", new string[] {
                    "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling",
                    "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio"})
                .Append(mlContext.Transforms.CopyColumns("Label", "MedianHomeValue"))
                .Append(mlContext.Regression.Trainers.OrdinaryLeastSquares());

            // Compute the CV result.
            var cvResult = mlContext.Regression.CrossValidate(data, pipeline, numFolds: 5);

            // Check that the results are valid
            Assert.IsType<RegressionMetrics>(cvResult[0].Metrics);
            Assert.IsType<TransformerChain<RegressionPredictionTransformer<OlsLinearRegressionModelParameters>>>(cvResult[0].Model);
            Assert.True(cvResult[0].ScoredHoldOutSet is IDataView);
            Assert.Equal(5, cvResult.Length);

            // And validate the metrics.
            foreach (var result in cvResult)
                Common.AssertMetrics(result.Metrics);
        }

        /// <summary>
        /// Train with validation set.
        /// </summary>
        [Fact]
        public void TrainWithValidationSet()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Get the dataset.
            var data = mlContext.Data.CreateTextLoader(TestDatasets.housing.GetLoaderColumns(),
                    hasHeader: TestDatasets.housing.fileHasHeader, separatorChar: TestDatasets.housing.fileSeparator)
                .Read(BaseTestClass.GetDataPath(TestDatasets.housing.trainFilename));
            var dataSplit = mlContext.Regression.TrainTestSplit(data, testFraction: 0.2);
            var trainData = dataSplit.TrainSet;
            var validData = dataSplit.TestSet;

            // Create a pipeline to featurize the dataset.
            var pipeline = mlContext.Transforms.Concatenate("Features", new string[] {
                    "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling",
                    "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio"})
                .Append(mlContext.Transforms.CopyColumns("Label", "MedianHomeValue"))
                .AppendCacheCheckpoint(mlContext) as IEstimator<ITransformer>;

            // Preprocess the datasets.
            var preprocessor = pipeline.Fit(trainData);
            var preprocessedTrainData = preprocessor.Transform(trainData);
            var preprocessedValidData = preprocessor.Transform(validData);

            // Train the model with a validation set.
            var trainedModel = mlContext.Regression.Trainers.FastTree(new Trainers.FastTree.FastTreeRegressionTrainer.Options {
                    NumTrees = 2,
                    EarlyStoppingMetrics = 2,
                    EarlyStoppingRule = new GLEarlyStoppingCriterion.Options()
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
    }
}
