// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.DotNet.PlatformAbstractions;
using Microsoft.Extensions.DependencyModel;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {

        // [Fact]
        [NativeDependencyFact("OneDalNative")]
        public void OneDalFastTreeBinaryEstimator()
        {
            Environment.SetEnvironmentVariable("MLNET_BACKEND", "ONEDAL");
            var trainDataPath = GetDataPath("binary_synth_data_train.csv");
            var testDataPath = GetDataPath("binary_synth_data_test.csv");

            var loader = ML.Data.CreateTextLoader(columns: new[] {
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

            var preprocessingPipeline = ML.Transforms.Concatenate("Features", new string[] { "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7" });

            var preprocessedTrainingData = preprocessingPipeline.Fit(trainingData).Transform(trainingData);
            var preprocessedTestingData = preprocessingPipeline.Fit(trainingData).Transform(testingData);

            // Output.WriteLine($"**** After preprocessing the data got {preprocessedTrainingData.Schema.Count} columns.");

            FastForestBinaryTrainer.Options options = new FastForestBinaryTrainer.Options();
            options.LabelColumnName = "target";
            options.FeatureColumnName = "Features";
            options.NumberOfTrees = 100;
            options.NumberOfLeaves = 128;
            options.MinimumExampleCountPerLeaf = 5;
            options.FeatureFraction = 1.0;

            var trainer = ML.BinaryClassification.Trainers.FastForest(options);
            var model = trainer.Fit(preprocessedTrainingData);
            var trainingPredictions = model.Transform(preprocessedTrainingData);
            var trainingMetrics = ML.BinaryClassification.EvaluateNonCalibrated(trainingPredictions, labelColumnName: "target");
            var testingPredictions = model.Transform(preprocessedTestingData);
            var testingMetrics = ML.BinaryClassification.EvaluateNonCalibrated(testingPredictions, labelColumnName: "target");

            Assert.True(testingMetrics.Accuracy > 0.8);
        }
    }
}
