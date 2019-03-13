// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Xunit;

namespace Microsoft.ML.Functional.Tests
{
    public class PredictionScenarios
    {
        /// <summary>
        /// Reconfigurable predictions: The following should be possible: A user trains a binary classifier,
        /// and through the test evaluator gets a PR curve, the based on the PR curve picks a new threshold
        /// and configures the scorer (or more precisely instantiates a new scorer over the same model parameters)
        /// with some threshold derived from that.
        /// </summary>
        [Fact]
        public void ReconfigurablePrediction()
        {
            var mlContext = new MLContext(seed: 789);

            // Get the dataset, create a train and test
            var data = mlContext.Data.CreateTextLoader(TestDatasets.housing.GetLoaderColumns(),
                hasHeader: TestDatasets.housing.fileHasHeader, separatorChar: TestDatasets.housing.fileSeparator)
                .Load(BaseTestClass.GetDataPath(TestDatasets.housing.trainFilename));
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Create a pipeline to train on the housing data
            var pipeline = mlContext.Transforms.Concatenate("Features", new string[] {
                    "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling",
                    "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio"})
                .Append(mlContext.Transforms.CopyColumns("Label", "MedianHomeValue"))
                .Append(mlContext.Regression.Trainers.Ols());

            var model = pipeline.Fit(split.TrainSet);

            var scoredTest = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(scoredTest);

            Common.AssertMetrics(metrics);

            // Todo #2465: Allow the setting of threshold and thresholdColumn for scoring.
            // This is no longer possible in the API
            //var newModel = new BinaryPredictionTransformer<IPredictorProducing<float>>(ml, model.Model, trainData.Schema, model.FeatureColumnName, threshold: 0.01f, thresholdColumn: DefaultColumnNames.Probability);
            //var newScoredTest = newModel.Transform(pipeline.Transform(testData));
            //var newMetrics = mlContext.BinaryClassification.Evaluate(scoredTest);
            // And the Threshold and ThresholdColumn properties are not settable.
            //var predictor = model.LastTransformer;
            //predictor.Threshold = 0.01; // Not possible
        }
    }
}
