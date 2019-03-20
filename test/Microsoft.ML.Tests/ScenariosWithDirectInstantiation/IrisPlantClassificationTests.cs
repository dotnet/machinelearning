// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictIrisModelUsingDirectInstantiationTest()
        {
            var mlContext = new MLContext(seed: 1);

            var reader = mlContext.Data.CreateTextLoader(columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("SepalLength", DataKind.Single, 1),
                    new TextLoader.Column("SepalWidth", DataKind.Single, 2),
                    new TextLoader.Column("PetalLength", DataKind.Single, 3),
                    new TextLoader.Column("PetalWidth", DataKind.Single, 4)
                }
            );

            var pipe = mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Transforms.Normalize("Features"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaCalibrated(
                    new SdcaCalibratedMulticlassTrainer.Options { NumberOfThreads = 1 }));

            // Read training and test data sets
            string dataPath = GetDataPath(TestDatasets.iris.trainFilename);
            string testDataPath = dataPath;
            var trainData = reader.Load(dataPath);
            var testData = reader.Load(testDataPath);

            // Train the pipeline
            var trainedModel = pipe.Fit(trainData);

            // Make prediction and then evaluate the trained pipeline
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);
            CompareMetrics(metrics);
            var predictFunction = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(trainedModel);
            ComparePredictions(predictFunction);
        }

        private void ComparePredictions(PredictionEngine<IrisData, IrisPrediction> model)
        {
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
        }

        private void CompareMetrics(MulticlassClassificationMetrics metrics)
        {
            Assert.Equal(.98, metrics.MacroAccuracy);
            Assert.Equal(.98, metrics.MicroAccuracy, 2);
            Assert.InRange(metrics.LogLoss, .05, .06);
            Assert.InRange(metrics.LogLossReduction, 94, 96);

            Assert.Equal(3, metrics.PerClassLogLoss.Count);
            Assert.Equal(0, metrics.PerClassLogLoss[0], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[1], 1);
            Assert.Equal(.1, metrics.PerClassLogLoss[2], 1);
        }
    }
}
