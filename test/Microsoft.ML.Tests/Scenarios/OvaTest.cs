// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection.Metadata;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void OvaLogisticRegression()
        {
            string dataPath = GetDataPath("iris.txt");

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);
            var reader = new TextLoader(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Single, 0),
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            var textData = reader.Load(GetDataPath(dataPath));
            var data = mlContext.Data.Cache(mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Fit(textData).Transform(textData));

            // Pipeline
            var logReg = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();
            var pipeline = mlContext.MulticlassClassification.Trainers.OneVersusAll(logReg, useProbabilities: false);
            var pipelineTyped = mlContext.MulticlassClassification.Trainers.OneVersusAllTyped(logReg, useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            var modelTyped = pipelineTyped.Fit(data);
            var predictionsTyped = modelTyped.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.94);

            var metricsTyped = mlContext.MulticlassClassification.Evaluate(predictionsTyped);
            Assert.True(metricsTyped.MicroAccuracy > 0.94);

            Assert.Equal(metrics.MicroAccuracy, metricsTyped.MicroAccuracy);
        }
                //.Append(ML.MulticlassClassification.Trainers.OneVersusAllUnCalibratedToCalibratedTyped<LinearBinaryModelParameters, PlattCalibrator>(sdcaTrainer))

        [Fact]
        public void OvaAveragedPerceptron()
        {
            string dataPath = GetDataPath("iris.txt");

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);
            var reader = new TextLoader(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Single, 0),
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            // Data
            var textData = reader.Load(GetDataPath(dataPath));
            var data = mlContext.Data.Cache(mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Fit(textData).Transform(textData));

            // Pipeline
            var ap = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                    new AveragedPerceptronTrainer.Options { Shuffle = true });
            var apTyped = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                    new AveragedPerceptronTrainer.Options { Shuffle = true });

            var pipeline = mlContext.MulticlassClassification.Trainers.OneVersusAll(ap, useProbabilities: false);
            var pipelineTyped = mlContext.MulticlassClassification.Trainers.OneVersusAllTyped(apTyped, useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            var modelTyped = pipelineTyped.Fit(data);
            var predictionsTyped = modelTyped.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.66);

            var metricsTyped = mlContext.MulticlassClassification.Evaluate(predictionsTyped);
            Assert.True(metricsTyped.MicroAccuracy > 0.66);

            Assert.Equal(metrics.MicroAccuracy, metricsTyped.MicroAccuracy);
        }

        [Fact]
        public void OvaCalibratedAveragedPerceptron()
        {
            string dataPath = GetDataPath("iris.txt");

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);
            var reader = new TextLoader(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Single, 0),
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            // Data
            var textData = reader.Load(GetDataPath(dataPath));
            var data = mlContext.Data.Cache(mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Fit(textData).Transform(textData));

            // Pipeline
            var ap = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                    new AveragedPerceptronTrainer.Options { Shuffle = true });
            var apTyped = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                    new AveragedPerceptronTrainer.Options { Shuffle = true });

            var pipeline = mlContext.MulticlassClassification.Trainers.OneVersusAll(ap);
            var pipelineTyped = mlContext.MulticlassClassification.Trainers.OneVersusAllUnCalibratedToCalibrated<LinearBinaryModelParameters, PlattCalibrator>(apTyped);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            var modelTyped = pipelineTyped.Fit(data);
            var predictionsTyped = modelTyped.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.95);

            var metricsTyped = mlContext.MulticlassClassification.Evaluate(predictionsTyped);
            Assert.True(metricsTyped.MicroAccuracy > 0.95);

            Assert.Equal(metrics.MicroAccuracy, metricsTyped.MicroAccuracy);
        }

        [Fact]
        public void OvaFastTree()
        {
            string dataPath = GetDataPath("iris.txt");

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);
            var reader = new TextLoader(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Single, 0),
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            // Data
            var textData = reader.Load(GetDataPath(dataPath));
            var data = mlContext.Data.Cache(mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Fit(textData).Transform(textData));

            // Pipeline
            var pipeline = mlContext.MulticlassClassification.Trainers.OneVersusAll(
                mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options { NumberOfThreads = 1 }),
                useProbabilities: false);

            var pipelineTyped = mlContext.MulticlassClassification.Trainers.OneVersusAllTyped(
                mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options { NumberOfThreads = 1 }),
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            var modelTyped = pipelineTyped.Fit(data);
            var predictionsTyped = modelTyped.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.99);

            var metricsTyped = mlContext.MulticlassClassification.Evaluate(predictionsTyped);
            Assert.True(metricsTyped.MicroAccuracy > 0.99);

            Assert.Equal(metrics.MicroAccuracy, metricsTyped.MicroAccuracy);
        }

        [Fact]
        public void OvaLinearSvm()
        {
            string dataPath = GetDataPath("iris.txt");

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);
            var mlContextTyped = new MLContext(seed: 1);

            var reader = new TextLoader(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Single, 0),
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            // REVIEW: readerTyped and dataTyped aren't used anywhere in this test, but if I take them out
            // the test will fail. It seems to me that something is changing state somewhere, maybe in the cache?
            var readerTyped = new TextLoader(mlContextTyped, new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Single, 0),
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 4) }),
                        }
            });
            // Data
            var textData = reader.Load(GetDataPath(dataPath));
            var data = mlContext.Data.Cache(mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Fit(textData).Transform(textData));
            var dataTyped = mlContextTyped.Data.Cache(mlContextTyped.Transforms.Conversion.MapValueToKey("Label")
                .Fit(textData).Transform(textData));

            // Pipeline
            var pipeline = mlContext.MulticlassClassification.Trainers.OneVersusAll(
                mlContext.BinaryClassification.Trainers.LinearSvm(new LinearSvmTrainer.Options { NumberOfIterations = 100 }),
                useProbabilities: false);

            var pipelineTyped = mlContextTyped.MulticlassClassification.Trainers.OneVersusAllTyped(
                mlContextTyped.BinaryClassification.Trainers.LinearSvm(new LinearSvmTrainer.Options { NumberOfIterations = 100 }),
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            var modelTyped = pipelineTyped.Fit(data);
            var predictionsTyped = modelTyped.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.95);

            var metricsTyped = mlContextTyped.MulticlassClassification.Evaluate(predictionsTyped);
            Assert.True(metricsTyped.MicroAccuracy > 0.95);

            Assert.Equal(metrics.MicroAccuracy, metricsTyped.MicroAccuracy);
        }
    }
}
