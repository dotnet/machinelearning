// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests : BaseTestClass
    {
        [Fact]
        public void TrainAndPredictIrisModelTest()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            var reader = mlContext.Data.CreateTextReader(columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 0),
                    new TextLoader.Column("SepalLength", DataKind.R4, 1),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 2),
                    new TextLoader.Column("PetalLength", DataKind.R4, 3),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 4)
                }
            );

            var pipe = mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                            .Append(mlContext.Transforms.Normalize("Features"))
                            .AppendCacheCheckpoint(mlContext)
                            .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features", advancedSettings: s => s.NumThreads = 1));

            // Read training and test data sets
            string dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            string testDataPath = dataPath;
            var trainData = reader.Read(dataPath);
            var testData = reader.Read(testDataPath);

            // Train the pipeline
            var trainedModel = pipe.Fit(trainData);

            // Make predictions
            var predictFunction = trainedModel.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext);
            IrisPrediction prediction = predictFunction.Predict(new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.3f,
                PetalLength = 1.6f,
                PetalWidth = 0.2f,
            });

            Assert.Equal(1, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            prediction = predictFunction.Predict(new IrisData()
            {
                SepalLength = 6.4f,
                SepalWidth = 3.1f,
                PetalLength = 5.5f,
                PetalWidth = 2.2f,
            });

            Assert.Equal(0, prediction.PredictedLabels[0], 2);
            Assert.Equal(0, prediction.PredictedLabels[1], 2);
            Assert.Equal(1, prediction.PredictedLabels[2], 2);

            prediction = predictFunction.Predict(new IrisData()
            {
                SepalLength = 4.4f,
                SepalWidth = 3.1f,
                PetalLength = 2.5f,
                PetalWidth = 1.2f,
            });

            Assert.Equal(.2, prediction.PredictedLabels[0], 1);
            Assert.Equal(.8, prediction.PredictedLabels[1], 1);
            Assert.Equal(0, prediction.PredictedLabels[2], 2);

            // Evaluate the trained pipeline
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted, topK: 3);

            Assert.Equal(.98, metrics.AccuracyMacro);
            Assert.Equal(.98, metrics.AccuracyMicro, 2);
            Assert.Equal(.06, metrics.LogLoss, 2);
            Assert.Equal(1, metrics.TopKAccuracy);

            Assert.Equal(3, metrics.PerClassLogLoss.Length);
            Assert.Equal(0, metrics.PerClassLogLoss[0], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[1], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[2], 1);
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

        public ScenariosTests(ITestOutputHelper output) : base(output)
        {
        }
    }
}

