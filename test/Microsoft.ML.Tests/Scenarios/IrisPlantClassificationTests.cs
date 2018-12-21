// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Data;
using Xunit;
using TextLoader = Microsoft.ML.Legacy.Data.TextLoader;

namespace Microsoft.ML.Scenarios
{
#pragma warning disable 612, 618
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictIrisModelTest()
        {
            string dataPath = GetDataPath("iris.txt");

            var pipeline = new Legacy.LearningPipeline(seed: 1, conc: 1);

            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(useHeader: false));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            Legacy.PredictionModel<IrisData, IrisPrediction> model = pipeline.Train<IrisData, IrisPrediction>();

            IrisPrediction prediction = model.Predict(new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.3f,
                PetalLength = 1.6f,
                PetalWidth = 0.2f,
            });

            Assert.Equal(1, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 6.4f,
                SepalWidth = 3.1f,
                PetalLength = 5.5f,
                PetalWidth = 2.2f,
            });

            Assert.Equal(0, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(1, prediction.PredictedLabels[2], 2);

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 4.4f,
                SepalWidth = 3.1f,
                PetalLength = 2.5f,
                PetalWidth = 1.2f,
            });

            Assert.Equal(.2, prediction.PredictedLabels[0], 1);
            Assert.Equal(.8, prediction.PredictedLabels[1], 1);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            // Note: Testing against the same data set as a simple way to test evaluation.
            // This isn't appropriate in real-world scenarios.
            string testDataPath = GetDataPath("iris.txt");
            var testData = new TextLoader(testDataPath).CreateFrom<IrisData>(useHeader: false);

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
            [LoadColumn(0)]
            public float Label;

            [LoadColumn(1)]
            public float SepalLength;

            [LoadColumn(2)]
            public float SepalWidth;

            [LoadColumn(3)]
            public float PetalLength;

            [LoadColumn(4)]
            public float PetalWidth;
        }

        public class IrisPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }
    }
#pragma warning restore 612, 618
}

