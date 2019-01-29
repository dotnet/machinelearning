﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Benchmarks.Harness;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML.Benchmarks
{
    [CIBenchmark]
    public class RffTransformTrain
    {
        private string _dataPath_Digits;

        [GlobalSetup]
        public void SetupTrainingSpeedTests()
        {
            _dataPath_Digits = BaseTestClass.GetDataPath(TestDatasets.Digits.trainFilename);

            if (!File.Exists(_dataPath_Digits))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _dataPath_Digits));
        }

        [Benchmark]
        public void CV_Multiclass_Digits_RffTransform_OVAAveragedPerceptron()
        {
            var mlContext = new MLContext();
            var reader = mlContext.Data.CreateTextLoader(new TextLoader.Arguments
            {
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 64),
                    new TextLoader.Column("Features", DataKind.R4, new[] {new TextLoader.Range() {Min = 0, Max = 63}})
                },
                HasHeader = false,
                Separators = new[] {','}
            });

            var data = reader.Read(_dataPath_Digits);

            var pipeline = mlContext.Transforms.Projection.CreateRandomFourierFeatures("FeaturesRFF", "Features")
            .AppendCacheCheckpoint(mlContext)
            .Append(mlContext.Transforms.Concatenate("Features", "FeaturesRFF"))
            .Append(new ValueToKeyMappingEstimator(mlContext, "Label"))
            .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(numIterations: 10)));

            var cvResults = mlContext.MulticlassClassification.CrossValidate(data, pipeline, numFolds: 5);
        }
    }
}
