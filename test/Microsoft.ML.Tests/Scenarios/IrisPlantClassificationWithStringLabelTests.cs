﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictIrisModelWithStringLabelTest()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            var reader = mlContext.Data.CreateTextLoader(columns: new[]
                {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                    new TextLoader.Column("IrisPlantType", DataKind.TX, 4),
                },
                separatorChar: ','
            );

            // Read training and test data sets
            string dataPath = GetDataPath("iris.data");
            string testDataPath = dataPath;
            var trainData = reader.Read(dataPath);
            var testData = reader.Read(testDataPath);

            // Create Estimator
            var pipe = mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Transforms.Normalize("Features"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("IrisPlantType", "Label"), TransformerScope.TrainTest)
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                    new SdcaMultiClassTrainer.Options { NumThreads = 1 }))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(("PredictedLabel", "Plant")));

            // Train the pipeline
            var trainedModel = pipe.Fit(trainData);

            // Make predictions
            var predictFunction = trainedModel.CreatePredictionEngine<IrisDataWithStringLabel, IrisPredictionWithStringLabel>(mlContext);
            IrisPredictionWithStringLabel prediction = predictFunction.Predict(new IrisDataWithStringLabel()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.3f,
                PetalLength = 1.6f,
                PetalWidth = 0.2f,
            });

            Assert.Equal(1, prediction.PredictedScores[0], 2);
            Assert.Equal(0, prediction.PredictedScores[1], 2);
            Assert.Equal(0, prediction.PredictedScores[2], 2);
            Assert.True(prediction.PredictedPlant == "Iris-setosa");

            prediction = predictFunction.Predict(new IrisDataWithStringLabel()
            {
                SepalLength = 6.4f,
                SepalWidth = 3.1f,
                PetalLength = 5.5f,
                PetalWidth = 2.2f,
            });

            Assert.Equal(0, prediction.PredictedScores[0], 2);
            Assert.Equal(0, prediction.PredictedScores[1], 2);
            Assert.Equal(1, prediction.PredictedScores[2], 2);
            Assert.True(prediction.PredictedPlant == "Iris-virginica");

            prediction = predictFunction.Predict(new IrisDataWithStringLabel()
            {
                SepalLength = 4.4f,
                SepalWidth = 3.1f,
                PetalLength = 2.5f,
                PetalWidth = 1.2f,
            });

            Assert.Equal(.2, prediction.PredictedScores[0], 1);
            Assert.Equal(.8, prediction.PredictedScores[1], 1);
            Assert.Equal(0, prediction.PredictedScores[2], 2);
            Assert.True(prediction.PredictedPlant == "Iris-versicolor");

            // Evaluate the trained pipeline
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted, topK: 3);

            Assert.Equal(.98, metrics.AccuracyMacro);
            Assert.Equal(.98, metrics.AccuracyMicro, 2);
            Assert.Equal(.06, metrics.LogLoss, 2);
            Assert.InRange(metrics.LogLossReduction, 94, 96);
            Assert.Equal(1, metrics.TopKAccuracy);

            Assert.Equal(3, metrics.PerClassLogLoss.Length);
            Assert.Equal(0, metrics.PerClassLogLoss[0], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[1], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[2], 1);
        }

        private class IrisDataWithStringLabel
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string IrisPlantType { get; set; }
        }

        private class IrisPredictionWithStringLabel
        {
            [ColumnName("Score")]
            public float[] PredictedScores { get; set; }

            [ColumnName("Plant")]
            public string PredictedPlant { get; set; }
        }
    }
}
