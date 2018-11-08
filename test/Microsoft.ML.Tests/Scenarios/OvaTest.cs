// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.Online;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void OVA_LR()
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
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(3, 4) }),
                        }
            });

            // Data
            var data = reader.Read(GetDataPath(dataPath));

            // Pipeline
            var pipeline = new Ova(
                mlContext, 
                () => new LogisticRegression(mlContext, "Features", "Label"),
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Assert.True(metrics.AccuracyMicro > 0.96);
        }

        [Fact]
        public void OVA_AP()
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
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(3, 4) }),
                        }
            });

            // Data
            var data = reader.Read(GetDataPath(dataPath));

            var averagePerceptron = new AveragedPerceptronTrainer(mlContext, "Label", "Features", advancedSettings: s =>
            {
                s.Shuffle = true;
                s.Calibrator = null;
            });

            // Pipeline
            var pipeline = new Ova(
                mlContext,
                () => averagePerceptron,
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Assert.True(metrics.AccuracyMicro > 0.66);
        }

        [Fact]
        public void OVA_FT()
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
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(3, 4) }),
                        }
            });

            // Data
            var data = reader.Read(GetDataPath(dataPath));

            var ft = new FastTreeBinaryClassificationTrainer(mlContext, "Label", "Features", advancedSettings: s =>
            {
                s.NumThreads = 1;
            });

            // Pipeline
            var pipeline = new Ova(
                mlContext,
                () => ft,
                useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Assert.True(metrics.AccuracyMicro > 0.99);
        }

        [Fact]
        public void OVA_SVM()
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
                            new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(3, 4) }),
                        }
            });

            // Data
            var data = reader.Read(GetDataPath(dataPath));

            // Pipeline
            var pipeline = new Ova(mlContext, () => new LinearSvm(mlContext, new LinearSvm.Arguments()),  useProbabilities: false);

            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            // Metrics
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Assert.True(metrics.AccuracyMicro > 0.73);
        }
    }
}
