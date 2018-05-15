// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictIrisModelTest()
        {
            string dataPath = GetDataPath("iris.txt");

            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader<IrisData>(dataPath, useHeader: false, separator: "tab"));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            PredictionModel<IrisData, IrisPrediction> model = pipeline.Train<IrisData, IrisPrediction>();

            IrisPrediction prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth= 5.1f,
            });

            Assert.Equal(1, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.1f,
                SepalWidth = 5.5f,
                PetalLength = 2.2f,
                PetalWidth = 6.4f,
            });

            Assert.Equal(0, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(1, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.1f,
                SepalWidth = 2.5f,
                PetalLength = 1.2f,
                PetalWidth = 4.4f,
            });

            Assert.Equal(.2, prediction.PredictedLabels[0], 1);
            Assert.Equal(.8, prediction.PredictedLabels[1], 1);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            // Note: Testing against the same data set as a simple way to test evaluation.
            // This isn't appropriate in real-world scenarios.
            string testDataPath = GetDataPath("iris.txt");
            var testData = new TextLoader<IrisData>(testDataPath, useHeader: false, separator: "tab");

            var evaluator = new ClassificationEvaluator();
            evaluator.OutputTopKAcc = 3;
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Assert.Equal(.98, metrics.AccuracyMacro);
            Assert.Equal(.98, metrics.AccuracyMicro, 2);
            Assert.Equal(.06, metrics.LogLoss, 2);
            Assert.InRange(metrics.LogLossReduction, 94, 96);
            Assert.Equal(1, metrics.TopKAccuracy);

            Assert.Equal(3, metrics.PerClassLogLoss.Length);
            Assert.Equal(0, metrics.PerClassLogLoss[0], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[1], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[2], 1);

            ConfusionMatrix matrix = metrics.ConfusionMatrix;
            Assert.Equal(3, matrix.Order);
            Assert.Equal(3, matrix.ClassNames.Count);
            Assert.Equal("0", matrix.ClassNames[0]);
            Assert.Equal("1", matrix.ClassNames[1]);
            Assert.Equal("2", matrix.ClassNames[2]);

            Assert.Equal(50, matrix[0, 0]);
            Assert.Equal(50, matrix["0", "0"]);
            Assert.Equal(0, matrix[0, 1]);
            Assert.Equal(0, matrix["0", "1"]);
            Assert.Equal(0, matrix[0, 2]);
            Assert.Equal(0, matrix["0", "2"]);

            Assert.Equal(0, matrix[1, 0]);
            Assert.Equal(0, matrix["1", "0"]);
            Assert.Equal(48, matrix[1, 1]);
            Assert.Equal(48, matrix["1", "1"]);
            Assert.Equal(2, matrix[1, 2]);
            Assert.Equal(2, matrix["1", "2"]);

            Assert.Equal(0, matrix[2, 0]);
            Assert.Equal(0, matrix["2", "0"]);
            Assert.Equal(1, matrix[2, 1]);
            Assert.Equal(1, matrix["2", "1"]);
            Assert.Equal(49, matrix[2, 2]);
            Assert.Equal(49, matrix["2", "2"]);
        }

        public class IrisData
        {
            [Column("0")]
            public float Label;

            [Column("1")]
            public float SepalLength;

            [Column("2")]
            public float SepalWidth;

            [Column("3")]
            public float PetalLength;

            [Column("4")]
            public float PetalWidth;
        }

        public class IrisPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }
    }
}

