// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.PerformanceTests.Harness;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.PerformanceTests
{
    [CIBenchmark]
    public class RffTransformTrain : BenchmarkBase
    {
        private string _dataPathDigits;

        [GlobalSetup]
        public void SetupTrainingSpeedTests()
        {
            _dataPathDigits = GetBenchmarkDataPathAndEnsureData(TestDatasets.Digits.trainFilename, TestDatasets.Digits.path);

            if (!File.Exists(_dataPathDigits))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _dataPathDigits));
        }

        [Benchmark]
        public void CV_Multiclass_Digits_RffTransform_OVAAveragedPerceptron()
        {
            var mlContext = new MLContext(1);
            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 64),
                    new TextLoader.Column("Features", DataKind.Single, new[] {new TextLoader.Range() {Min = 0, Max = 63}})
                },
                HasHeader = false,
                Separators = new[] { ',' }
            });

            var data = loader.Load(_dataPathDigits);

            var pipeline = mlContext.Transforms.ApproximatedKernelMap("FeaturesRFF", "Features")
            .AppendCacheCheckpoint(mlContext)
            .Append(mlContext.Transforms.Concatenate("Features", "FeaturesRFF"))
            .Append(new ValueToKeyMappingEstimator(mlContext, "Label"))
            .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(numberOfIterations: 10)));

            var cvResults = mlContext.MulticlassClassification.CrossValidate(data, pipeline, numberOfFolds: 5);
        }
    }
}
