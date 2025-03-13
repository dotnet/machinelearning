// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using FluentAssertions;
using Microsoft.Data.Analysis;
using Microsoft.ML.AutoML;
using Microsoft.ML.Fairlearn.AutoML;
using Microsoft.ML.TestFramework.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Fairlearn.Tests
{
    public class GridSearchTest
    {
        private readonly ITestOutputHelper _output;
        public GridSearchTest(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Generate_binary_classification_lambda_search_space_test()
        {
            var context = new MLContext();
            var moment = new UtilityParity();
            var X = CreateDummyDataset();
            moment.LoadData(X, X["y_true"], X["sensitiveFeature"] as StringDataFrameColumn);

            var searchSpace = Utilities.GenerateBinaryClassificationLambdaSearchSpace(moment, 5);
            searchSpace.Keys.Should().BeEquivalentTo("a_pos", "a_neg", "b_pos", "b_neg");

        }

        private DataFrame CreateDummyDataset()
        {
            var df = new DataFrame();
            df["X"] = DataFrameColumn.Create("X", new[] { 0f, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            df["y_true"] = DataFrameColumn.Create("y_true", new[] { true, true, true, true, true, true, true, false, false, false });
            df["y_pred"] = DataFrameColumn.Create("y_pred", new[] { true, true, true, true, false, false, false, true, false, false });
            df["sensitiveFeature"] = DataFrameColumn.Create("sensitiveFeature", new[] { "a", "b", "a", "a", "b", "a", "b", "b", "a", "b" });

            return df;
        }

        // Data generated so it is identical from Binary_Classification.ipynb from Fairlearn.github on Github
        private DataFrame CreateGridSearchDataset()
        {
            float[] score_feature = new float[52];
            int index = 0;
            for (int i = 0; i < 31; i++)
            {
                score_feature[index] = (i * 1.0f) / 30;
                index++;
            }
            for (int j = 0; j < 21; j++)
            {
                score_feature[index] = (j * 1.0f) / 20;
                index++;
            }
            var df = new DataFrame();
            df["score_feature"] = DataFrameColumn.Create("score_feature", score_feature);
            df["y"] = DataFrameColumn.Create("y", new[] {
                false, false, false, false, false, false, false,  true,  true,
                true,  true,  true,  true,  true,  true,  true,  true,  true,
                true,  true,  true,  true,  true,  true,  true,  true,  true,
                true,  true,  true,  true, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false,
                true,  true,  true,  true,  true,  true,  true });
            df["sensitiveFeature"] = DataFrameColumn.Create("sensitiveFeature", new[] { "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3" });

            return df;
        }

        /// <summary>
        /// This trial runner run the tests from Grid searh for Binary Classification.ipynb
        /// </summary>
        [Fact(Skip = "Currently flaky on non x86/x64 devices. Disabling until we figure it out. See https://github.com/dotnet/machinelearning/issues/6684")]
        public void TestGridSearchTrialRunner2()
        {
            var context = new MLContext();
            context.Log += (o, e) =>
            {

                if (e.Source == "AutoMLExperiment")
                {
                    _output.WriteLine(e.Message);
                }
            };
            var experiment = context.Auto().CreateExperiment();
            var df = CreateGridSearchDataset();
            var shuffledDataset = context.Data.ShuffleRows(df);
            var trainTestSplit = context.Data.TrainTestSplit(shuffledDataset, 0.2);
            var pipeline = context.Transforms.Categorical.OneHotHashEncoding("sensitiveFeature_encode", "sensitiveFeature")
                                   .Append(context.Transforms.Concatenate("Features", "sensitiveFeature_encode", "score_feature"))
                                    .Append(context.Auto().BinaryClassification(labelColumnName: "y", exampleWeightColumnName: "signedWeight"));

            experiment.SetPipeline(pipeline)
                        .SetDataset(trainTestSplit)
                        .SetBinaryClassificationMetricWithFairLearn("y", "PredictedLabel", "sensitiveFeature", "signedWeight")
                        .SetTrainingTimeInSeconds(10);//100

            var bestResult = experiment.Run();
            var model = bestResult.Model;
            bestResult.Metric.Should().BeGreaterThanOrEqualTo(0.4);
        }
    }
}
