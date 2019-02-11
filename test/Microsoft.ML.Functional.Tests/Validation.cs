// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers.HalLearners;
using Xunit;
using static Microsoft.ML.RunTests.TestDataViewBase;

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

            // Get the dataset
            var data = mlContext.Data.CreateTextLoader(TestDatasets.housing.GetLoaderColumns(), hasHeader: true)
                .Read(BaseTestClass.GetDataPath(TestDatasets.housing.trainFilename));

            // Create a pipeline to train on the sentiment data
            var pipeline = mlContext.Transforms.Concatenate("Features", new string[] {
                    "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling",
                    "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio"})
                .Append(mlContext.Transforms.CopyColumns("Label", "MedianHomeValue"))
                .Append(mlContext.Regression.Trainers.OrdinaryLeastSquares());

            // Compute the CV result
            var cvResult = mlContext.Regression.CrossValidate(data, pipeline, numFolds: 5);

            // Check that the results are valid
            Assert.IsType<RegressionMetrics>(cvResult[0].metrics);
            Assert.IsType<TransformerChain<RegressionPredictionTransformer<OlsLinearRegressionModelParameters>>>(cvResult[0].model);
            Assert.True(cvResult[0].scoredTestData is IDataView);
            Assert.Equal(5, cvResult.Length);

            // And validate the metrics
            foreach (var result in cvResult)
                Common.CheckMetrics(result.metrics);
        }

        /// <summary>
        /// Train with validation set.
        /// </summary>
        [Fact]
        public void TrainWithValidationSet()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Get the dataset
            var data = mlContext.Data.CreateTextLoader(TestDatasets.housing.GetLoaderColumns(), hasHeader: true)
                .Read(BaseTestClass.GetDataPath(TestDatasets.housing.trainFilename));
            (var trainData, var validData) = mlContext.Regression.TrainTestSplit(data, testFraction: 0.2);

            // Create a pipeline to train on the sentiment data
            var pipeline = mlContext.Transforms.Concatenate("Features", new string[] {
                    "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling",
                    "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio"})
                .Append(mlContext.Transforms.CopyColumns("Label", "MedianHomeValue"))
                .AppendCacheCheckpoint(mlContext) as IEstimator<ITransformer>;
            var preprocessor = pipeline.Fit(trainData);
            var transformedValidData = preprocessor.Transform(validData);

            // Todo #2502: Allow fitting a model with a validation set
            // Train model with validation set.
            // Note for #2502: There is no way below to specify a validation set for the learner
            pipeline = pipeline.Append(mlContext.Regression.Trainers.FastTree(numTrees: 2));
            // Note for #2502: Nor is there a way to specify a validation set in the Fit
            var model = pipeline.Fit(trainData);

            // Score the data sets
            var scoredTrainData = model.Transform(trainData);
            var scoredValidData = model.Transform(validData);

            var trainMetrics = mlContext.Regression.Evaluate(scoredTrainData);
            var validMetrics = mlContext.Regression.Evaluate(scoredValidData);

            Common.CheckMetrics(trainMetrics);
            Common.CheckMetrics(validMetrics);
        }
    }
}
