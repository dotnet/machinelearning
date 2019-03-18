// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
            var logReg = mlContext.BinaryClassification.Trainers.LogisticRegression();
            var pipeline = mlContext.MulticlassClassification.Trainers.OneVersusAll(logReg, useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.94);
        }

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

            var pipeline = mlContext.MulticlassClassification.Trainers.OneVersusAll(ap, useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.66);
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

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.99);
        }

        [Fact]
        public void OvaLinearSvm()
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
                mlContext.BinaryClassification.Trainers.LinearSvm(new LinearSvmTrainer.Options { NumberOfIterations = 100 }),
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.MicroAccuracy > 0.83);
        }
    }
}
