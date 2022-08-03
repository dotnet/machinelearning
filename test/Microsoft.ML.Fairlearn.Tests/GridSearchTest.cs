// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using FluentAssertions;
using Microsoft.Data.Analysis;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Fairlearn.AutoML;
using Microsoft.ML.Fairlearn.reductions;
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
            var X = this.CreateDummyDataset();
            moment.LoadData(X, X["y_true"], X["sensitiveFeature"] as StringDataFrameColumn);

            var searchSpace = Utilities.GenerateBinaryClassificationLambdaSearchSpace(context, moment, 5);
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

        [Fact]
        public void TestGridSearchTrialRunner()
        {
            var context = new MLContext();
            context.Log += (o, e) =>
            {

                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    _output.WriteLine(e.Message);
                }
            };

            var experiment = context.Auto().CreateExperiment();
            var df = this.CreateDummyDataset();
            var moment = new UtilityParity();
            moment.LoadData(df, df["y_true"], df["sensitiveFeature"] as StringDataFrameColumn);

            var pipeline = context.Transforms.Categorical.OneHotHashEncoding("sensitiveFeature_encode", "sensitiveFeature")
                                   .Append(context.Transforms.Concatenate("Features", "sensitiveFeature_encode", "X"))
                                    .Append(context.Auto().BinaryClassification(labelColumnName: "y_true", exampleWeightColumnName: "signedWeight"));
            var trialRunner = new GridSearchTrailRunner(context, this.CreateDummyDataset(), this.CreateDummyDataset(), "y_true");
            experiment.SetPipeline(pipeline)
                        .SetEvaluateMetric(BinaryClassificationMetric.Accuracy, "y_true", "PredictedLabel")
                        .SetTrialRunner(trialRunner)
                        .SetBinaryClassificationMoment(moment)
                        .SetTrainingTimeInSeconds(20);

            var bestResult = experiment.Run();
            bestResult.Metric.Should().BeGreaterOrEqualTo(0.8);
        }
        // Data generated so it is identical from Binary_Classification.ipynb from Fairlearn.github on Github
        private DataFrame CreateGridScearhDataset()
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
            df["y"] = DataFrameColumn.Create("y", new[] { false, false, false, false, false, false, false,  true,  true,
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
        [Fact]
        public void TestGridSearchTrialRunner2()
        {
            _output.WriteLine("Test");
            var context = new MLContext();
            context.Log += (o, e) =>
            {

                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    _output.WriteLine(e.Message);
                }
            };
            var experiment = context.Auto().CreateExperiment();
            var df = CreateGridScearhDataset();
            var shuffledDataset = context.Data.ShuffleRows(df);
            var trainTestSplit = context.Data.TrainTestSplit(shuffledDataset, 0.2);
            var moment = new UtilityParity();
            var dfTrainSet = trainTestSplit.TrainSet.ToDataFrame();
            moment.LoadData(dfTrainSet, dfTrainSet["y"], dfTrainSet["sensitiveFeature"] as StringDataFrameColumn);

            var pipeline = context.Transforms.Categorical.OneHotHashEncoding("sensitiveFeature_encode", "sensitiveFeature")
                                   .Append(context.Transforms.Concatenate("Features", "sensitiveFeature_encode", "score_feature"))
                                    .Append(context.Auto().BinaryClassification(labelColumnName: "y", exampleWeightColumnName: "signedWeight"));
            var trialRunner = new GridSearchTrailRunner(context, trainTestSplit.TrainSet, trainTestSplit.TestSet, "y");
            experiment.SetPipeline(pipeline)
                        .SetEvaluateMetric(BinaryClassificationMetric.Accuracy, "y", "PredictedLabel")
                        .SetTrialRunner(trialRunner)
                        .SetBinaryClassificationMoment(moment)
                        .SetTrainingTimeInSeconds(10);//100

            var bestResult = experiment.Run();
            var model = bestResult.Model;
            var df2 = CreateGridScearhDataset();
            //bestResult.Metric.Should().BeGreaterOrEqualTo(0.75);
            var eval = model.Transform(df2);
            //Consoel.WriteLine("Test")
            var predictedColumn = eval.GetColumn<bool>("PredictedLabel");
            foreach (var item in predictedColumn)
            {
                _output.WriteLine(item.ToString());
            }
        }
    }
}
