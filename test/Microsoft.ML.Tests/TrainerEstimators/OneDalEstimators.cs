// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.DotNet.RemoteExecutor;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers.FastTree;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {

        [NativeDependencyFact("OneDalNative")]
        public void OneDalFastTreeBinaryEstimator()
        {
            var originalBackend = Environment.GetEnvironmentVariable("MLNET_BACKEND");
            var trainDataPath = GetDataPath("binary_synth_data_train.csv");
            var testDataPath = GetDataPath("binary_synth_data_test.csv");

            // oneDAL has to be selected before static initialization of MLContext runs, which means using RemoteExecutor here.
            var options = new RemoteInvokeOptions();
            options.StartInfo.EnvironmentVariables["MLNET_BACKEND"] = "ONEDAL";
            // Enable verbose output for oneDAL, so that the output check below can verify that oneDAL is being used.
            options.StartInfo.EnvironmentVariables["MLNET_BACKEND_VERBOSE"] = "1";
            // oneDAL loads its threading runtime dynamically; test dependencies are beside the test assembly.
            options.StartInfo.EnvironmentVariables["PATH"] =
                Path.GetDirectoryName(typeof(TrainerEstimators).Assembly.Location) + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH");
            options.StartInfo.RedirectStandardOutput = true;
            var remote = RemoteExecutor.Invoke(TrainOneDalFastForest, trainDataPath, testDataPath, options);
            var output = remote.Process.StandardOutput.ReadToEndAsync();
            remote.Dispose();

            Assert.Contains("Decision Forest Classification parameters:", output.GetAwaiter().GetResult());
            Assert.Equal(originalBackend, Environment.GetEnvironmentVariable("MLNET_BACKEND"));
        }

        private static void TrainOneDalFastForest(string trainDataPath, string testDataPath)
        {
            var ml = new MLContext(seed: 1);
            Assert.True(MLContext.OneDalDispatchingEnabled);

            var loader = ml.Data.CreateTextLoader(columns: new[] {
              new TextLoader.Column("f0", DataKind.Single, 0),
              new TextLoader.Column("f1", DataKind.Single, 1),
              new TextLoader.Column("f2", DataKind.Single, 2),
              new TextLoader.Column("f3", DataKind.Single, 3),
              new TextLoader.Column("f4", DataKind.Single, 4),
              new TextLoader.Column("f5", DataKind.Single, 5),
              new TextLoader.Column("f6", DataKind.Single, 6),
              new TextLoader.Column("f7", DataKind.Single, 7),
              new TextLoader.Column("target", DataKind.Boolean,8)},
              separatorChar: ',',
              hasHeader: true
             );

            var trainingData = loader.Load(trainDataPath);
            var testingData = loader.Load(testDataPath);

            var preprocessingPipeline = ml.Transforms.Concatenate("Features", new string[] { "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7" });

            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);

            FastForestBinaryTrainer.Options options = new FastForestBinaryTrainer.Options();
            options.LabelColumnName = "target";
            options.FeatureColumnName = "Features";
            options.NumberOfTrees = 100;
            options.NumberOfLeaves = 128;
            options.MinimumExampleCountPerLeaf = 5;
            options.FeatureFraction = 1.0;

            var trainer = ml.BinaryClassification.Trainers.FastForest(options);
            var model = trainer.Fit(preprocessedTrainingData);
            var trainingPredictions = model.Transform(preprocessedTrainingData);
            var trainingMetrics = ml.BinaryClassification.EvaluateNonCalibrated(trainingPredictions, labelColumnName: "target");
            var testingPredictions = model.Transform(preprocessedTestingData);
            var testingMetrics = ml.BinaryClassification.EvaluateNonCalibrated(testingPredictions, labelColumnName: "target");

            Assert.True(testingMetrics.Accuracy > 0.8);
        }
    }
}
