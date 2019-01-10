// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Learners;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
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
            var reader = new TextLoader(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.R4, 0),
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            // Data
            var data = reader.Read(GetDataPath(dataPath));

            // Pipeline
            var pipeline = new Ova(
                mlContext, 
                new LogisticRegression(mlContext, "Label", "Features"),
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.AccuracyMicro > 0.94);
        }

        [Fact]
        public void OvaAveragedPerceptron()
        {
            string dataPath = GetDataPath("iris.txt");

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);
            var reader = new TextLoader(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.R4, 0),
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            // Data
            var data = mlContext.Data.Cache(reader.Read(GetDataPath(dataPath)));

            // Pipeline
            var pipeline = new Ova(
                mlContext,
                new AveragedPerceptronTrainer(mlContext, "Label", "Features", advancedSettings: s => { s.Shuffle = true;  s.Calibrator = null; }),
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.AccuracyMicro > 0.71);
        }

        [Fact]
        public void OvaFastTree()
        {
            string dataPath = GetDataPath("iris.txt");

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);
            var reader = new TextLoader(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.R4, 0),
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            // Data
            var data = reader.Read(GetDataPath(dataPath));

            // Pipeline
            var pipeline = new Ova(
                mlContext,
                mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryClassificationTrainer.Options { NumThreads = 1 }),
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.AccuracyMicro > 0.99);
        }

        [Fact]
        public void OvaLinearSvm()
        {
            string dataPath = GetDataPath("iris.txt");

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);
            var reader = new TextLoader(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.R4, 0),
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(1, 4) }),
                        }
            });

            // Data
            var data = mlContext.Data.Cache(reader.Read(GetDataPath(dataPath)));

            // Pipeline
            var pipeline = new Ova(mlContext, new LinearSvmTrainer(mlContext, numIterations: 100),  useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Assert.True(metrics.AccuracyMicro > 0.83);
        }
    }
}
